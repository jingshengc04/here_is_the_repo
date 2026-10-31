import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from . import pipeline, replay as replay_module
from .models import ErrorBody, IntakeInput, Run

log = logging.getLogger("app")
app = FastAPI(title="Portfolio Research API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
                   expose_headers=["*"])

# ponytail: in-memory, single instance (Cloud Run max-instances=1); lost on restart. Swap for Redis/DB to scale out.
RUNS: dict[str, dict] = {}  # runId -> {"run": Run, "events": list[dict], "cond": asyncio.Condition}
TERMINAL = ("run.completed", "run.failed")


class APIErr(Exception):
    def __init__(self, status: int, code: str, message: str, details=None):
        self.status, self.body = status, {"code": code, "message": message, "details": details}


def _err(status: int, code: str, message: str, details=None) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message, "details": details}}, status_code=status)


@app.exception_handler(APIErr)
async def _api_err(_: Request, e: APIErr):
    return JSONResponse({"error": e.body}, status_code=e.status)


@app.exception_handler(RequestValidationError)
async def _validation(_: Request, e: RequestValidationError):
    details = {}
    for x in e.errors():
        loc = [p for p in x["loc"] if p not in ("body", "query")]
        details[".".join(map(str, loc)) or "body"] = x["msg"]
    return _err(422, "VALIDATION_ERROR", "Invalid request", details)


@app.exception_handler(Exception)
async def _internal(_: Request, e: Exception):
    log.exception("unhandled")
    return _err(500, "INTERNAL", "Something went wrong")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _execute(run_id: str, intake: IntakeInput, replay_name: str | None = None) -> None:
    st = RUNS[run_id]
    run: Run = st["run"]

    async def emit(ev: dict) -> None:
        async with st["cond"]:
            st["events"].append({**ev, "runId": run_id, "seq": len(st["events"]) + 1, "ts": _now()})
            st["cond"].notify_all()

    run.status = "RUNNING"
    try:
        if replay_name:
            report = await replay_module.run_replay(run_id, intake, replay_name, emit)
        else:
            report = await pipeline.run_pipeline(run_id, intake, emit)
        run.report, run.status = report, "COMPLETED"
        await emit({"type": "run.completed", "report": report.model_dump()})
        try:
            replay_module.save_run(run_id, intake, st["events"])
        except Exception:
            log.exception("failed to save run")
    except Exception:
        log.exception("pipeline failed")
        run.status = "FAILED"
        run.error = ErrorBody(code="INTERNAL", message="The research run failed.")
        await emit({"type": "run.failed", "error": run.error.model_dump()})


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/runs", status_code=202)
async def create_run(intake: IntakeInput, replay: str | None = None) -> Run:
    if replay:
        if not replay_module.is_valid_name(replay):
            raise APIErr(422, "VALIDATION_ERROR", "Invalid replay name", {"replay": "Name must match ^[A-Za-z0-9_-]+$"})
        if not replay_module.find_replay_file(replay):
            raise APIErr(404, "RUN_NOT_FOUND", f"Replay '{replay}' not found")
    if any(s["run"].status in ("QUEUED", "RUNNING") for s in RUNS.values()):
        raise APIErr(429, "RATE_LIMITED", "A run is already in progress. Try again shortly.")
    run_id = "run_" + uuid.uuid4().hex[:12]
    run = Run(runId=run_id, status="QUEUED", intake=intake, createdAt=_now())
    RUNS[run_id] = {"run": run, "events": [], "cond": asyncio.Condition()}
    RUNS[run_id]["task"] = asyncio.create_task(_execute(run_id, intake, replay))  # keep a ref so it isn't GC'd
    return run


def _get(run_id: str) -> dict:
    if run_id not in RUNS:
        raise APIErr(404, "RUN_NOT_FOUND", f"No run {run_id}")
    return RUNS[run_id]


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> Run:
    return _get(run_id)["run"]


@app.get("/api/runs/{run_id}/events")
async def run_events(run_id: str, request: Request):
    st = _get(run_id)
    try:
        idx = int(request.headers.get("last-event-id", "").rsplit(":", 1)[-1])  # next index == last seq
    except ValueError:
        idx = 0

    async def gen():
        nonlocal idx
        while True:
            try:
                async with st["cond"]:
                    await asyncio.wait_for(st["cond"].wait_for(lambda: len(st["events"]) > idx), 15)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"
                continue
            while idx < len(st["events"]):
                ev = st["events"][idx]
                idx += 1
                yield f"id: {run_id}:{ev['seq']}\nevent: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
                if ev["type"] in TERMINAL:
                    return

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
