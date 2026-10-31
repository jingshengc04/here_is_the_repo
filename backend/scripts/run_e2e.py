"""Run the full REAL pipeline in-process through the FastAPI app and print a compact log + summary.

uv run --env-file <.env> python scripts/run_e2e.py --amount 10000 --horizon 2 --risk MODERATE --theme "AI infrastructure"
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["PIPELINE"] = "real"

import httpx  # noqa: E402

from app import main, replay  # noqa: E402


class CallCounter(logging.Handler):
    n = 0

    def emit(self, record):  # one INFO record per llm.ask() call
        if record.name == "app.llm" and record.levelno == logging.INFO:
            CallCounter.n += 1


def detail(ev: dict) -> str:
    t = ev["type"]
    if t == "agent.progress":
        return ev["message"][:80]
    if t == "agent.finding":
        f = ev["finding"]
        return f"{f['symbol']} {f['sentiment']} {f['claim'][:60]}"
    if t == "agent.completed":
        return f"{ev['summary'][:50]} ({ev['durationMs']}ms)"
    if t == "agent.failed":
        return ev["message"][:80]
    if t == "verification.completed":
        v = ev["verification"]
        return f"approved={v['isApproved']} " + "; ".join(v["changesRequested"])[:80]
    if t == "run.failed":
        return str(ev["error"])
    return ""


async def tail_events(run_id: str, t0: float) -> None:
    """Live log: poll the in-process event list (httpx ASGITransport buffers streaming bodies)."""
    i = 0
    while True:
        evs = main.RUNS[run_id]["events"]
        while i < len(evs):
            ev = evs[i]
            i += 1
            who = ev.get("agent") or (f"r{ev['round']}" if "round" in ev else "")
            print(f"{time.monotonic() - t0:6.1f}s  {ev['type']:<24} {who:<13} {detail(ev)}", flush=True)
            if ev["type"] in ("run.completed", "run.failed"):
                return
        await asyncio.sleep(0.2)


async def amain(args) -> int:
    logging.getLogger("app.llm").addHandler(CallCounter())
    body = {"amountUsd": args.amount, "horizonYears": args.horizon, "riskTolerance": args.risk}
    if args.theme:
        body["theme"] = args.theme
    prefs = {}
    if args.must_include:
        prefs["mustInclude"] = args.must_include.split(",")
    if args.exclude:
        prefs["exclude"] = args.exclude.split(",")
    if args.styles:
        prefs["styles"] = args.styles.split(",")
    if args.targets:
        prefs["assetClassTargets"] = {k: float(v) for k, v in (t.split("=") for t in args.targets.split(","))}
    if args.notes:
        prefs["notes"] = args.notes
    if prefs:
        body["preferences"] = prefs
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://e2e", timeout=None) as c:
        t0 = time.monotonic()
        r = await c.post("/api/runs", json=body)
        if r.status_code != 202:
            print("POST failed", r.status_code, r.text)
            return 2
        run_id = r.json()["runId"]
        print(f"run {run_id} started")
        await tail_events(run_id, t0)
        elapsed = time.monotonic() - t0
        # also consume the real SSE endpoint (replays everything, then closes)
        sse = await c.get(f"/api/runs/{run_id}/events")
        types = [l[7:] for l in sse.text.splitlines() if l.startswith("event: ")]
        snap = (await c.get(f"/api/runs/{run_id}")).json()

    print("\n==== SUMMARY ====")
    print(f"status: {snap['status']} | SSE events: {len(types)} (last: {types[-1] if types else None})")
    print(f"elapsed: {elapsed:.1f}s | Gemini calls (llm.ask INFO records): {CallCounter.n}")
    if snap["status"] != "COMPLETED":
        print("error:", snap["error"])
        return 1
    rep = snap["report"]
    for rd in rep["rounds"]:
        print(f"round {rd['round']}: isApproved={rd['verification']['isApproved']}"
              + "".join(f"\n   - {x}" for x in rd["verification"]["changesRequested"]))
    print("candidates:", ", ".join(f"{h['symbol']}{'*' if h.get('isUserRequested') else ''}"
                                  for sl in rep["finalAllocation"]["slices"] for h in sl["holdings"]), "(* = user requested)")
    print("preference notes:", rep.get("preferenceNotes") or [])
    print("final weights:")
    for s in rep["finalAllocation"]["slices"]:
        print(f"  {s['assetClass']} {s['weightPct']}%: " + ", ".join(f"{h['symbol']} {h['weightPct']}" for h in s["holdings"]))
    last = rep["rounds"][-1]["backtest"]
    print("crossCheck:", json.dumps(last["crossCheck"]))
    print("sharpeCi:", json.dumps(last.get("sharpeCi")))
    print("sample finding sources:")
    for f in [f for f in rep["findings"] if f["sources"]][:6]:
        print(f"  {f['id']} {f['agent']} {f['symbol']}: " + " | ".join(x["url"][:60] for x in f["sources"]))
    print("distinct source sets across findings:", len({tuple(x["url"] for x in f["sources"]) for f in rep["findings"] if f["sources"]}),
          "of", sum(1 for f in rep["findings"] if f["sources"]), "findings with sources")
    print("saved run:", replay.DATA_RUNS_DIR / f"{run_id}.json")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--amount", type=float, default=10000)
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--risk", default="MODERATE", choices=["CONSERVATIVE", "MODERATE", "AGGRESSIVE"])
    ap.add_argument("--theme", default="AI infrastructure")
    ap.add_argument("--must-include", default="")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--styles", default="")
    ap.add_argument("--targets", default="", help="e.g. CRYPTO=30,STOCKS=50")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("app.llm").setLevel(logging.INFO)
    logging.getLogger("app.llm").propagate = False
    logging.getLogger("httpx").setLevel(logging.WARNING)
    sys.exit(asyncio.run(amain(a)))
