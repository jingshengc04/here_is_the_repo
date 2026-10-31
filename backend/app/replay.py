import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
from typing import Any

from .models import IntakeInput, Report

log = logging.getLogger("app.replay")

Emit = Callable[[dict], Awaitable[None]]

DATA_RUNS_DIR = Path(__file__).resolve().parent.parent / "data" / "runs"
FIXTURES_RUNS_DIR = Path(__file__).resolve().parent / "fixtures" / "runs"
REPLAY_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def is_valid_name(name: str) -> bool:
    return bool(REPLAY_NAME_PATTERN.match(name))


def find_replay_file(name: str) -> Path | None:
    p1 = DATA_RUNS_DIR / f"{name}.json"
    if p1.is_file():
        return p1
    p2 = FIXTURES_RUNS_DIR / f"{name}.json"
    if p2.is_file():
        return p2
    return None


def save_run(run_id: str, intake: Any, events: list[dict]) -> None:
    try:
        DATA_RUNS_DIR.mkdir(parents=True, exist_ok=True)
        intake_dict = intake.model_dump() if hasattr(intake, "model_dump") else intake
        payload = {
            "intake": intake_dict,
            "events": events,
        }
        file_path = DATA_RUNS_DIR / f"{run_id}.json"
        file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("Saved run %s to %s", run_id, file_path)
    except Exception:
        log.exception("Failed to save run %s", run_id)


def _parse_ts(ts_str: Any) -> datetime | None:
    if not ts_str or not isinstance(ts_str, str):
        return None
    try:
        s = ts_str[:-1] + "+00:00" if ts_str.endswith("Z") else ts_str
        return datetime.fromisoformat(s)
    except Exception:
        return None


async def run_replay(run_id: str, intake: IntakeInput, replay_name: str, emit: Emit) -> Report:
    file_path = find_replay_file(replay_name)
    if not file_path:
        raise FileNotFoundError(f"Replay '{replay_name}' not found")

    content = json.loads(file_path.read_text(encoding="utf-8"))
    saved_events = content.get("events", [])
    speed = float(os.environ.get("REPLAY_SPEED", "1.0"))

    last_dt = None
    final_report_dict = None

    for ev in saved_events:
        ev_type = ev.get("type")
        curr_dt = _parse_ts(ev.get("ts"))

        if last_dt is not None and curr_dt is not None:
            dt = (curr_dt - last_dt).total_seconds()
            if dt > 0 and speed > 0:
                await asyncio.sleep(dt * speed)
        if curr_dt is not None:
            last_dt = curr_dt

        if ev_type == "run.completed":
            final_report_dict = ev.get("report")
            break
        elif ev_type == "run.failed":
            error_body = ev.get("error", {})
            raise RuntimeError(error_body.get("message", "Replayed run failed"))
        else:
            clean_ev = {k: v for k, v in ev.items() if k not in ("runId", "seq", "ts")}
            await emit(clean_ev)

    if not final_report_dict:
        raise ValueError(f"Replay '{replay_name}' has no run.completed event with report")

    rep = dict(final_report_dict)
    rep["runId"] = run_id
    rep["intake"] = intake.model_dump() if hasattr(intake, "model_dump") else intake
    rep["generatedAt"] = datetime.now(timezone.utc).isoformat()
    return Report.model_validate(rep)
