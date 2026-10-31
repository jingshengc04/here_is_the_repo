import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from app import main
from app.models import Report, Run, RunEvent

GOOD = {"amountUsd": 10000, "horizonYears": 2, "riskTolerance": "MODERATE", "theme": "AI"}
EVENT = TypeAdapter(RunEvent)
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("FAKE_DELAY_SCALE", "0")
    main.RUNS.clear()
    with TestClient(main.app) as c:  # one portal => background tasks survive between requests
        yield c


def frames(client, run_id, headers=None):
    with client.stream("GET", f"/api/runs/{run_id}/events", headers=headers or {}) as r:
        assert r.status_code == 200
        assert r.headers["x-accel-buffering"] == "no"
        text = "".join(r.iter_text())
    out = []
    for block in text.strip().split("\n\n"):
        if block.startswith(":"):
            continue
        f = dict(line.split(": ", 1) for line in block.split("\n"))
        out.append(f)
    return out


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_bad_intake_422(client):
    r = client.post("/api/runs", json={**GOOD, "amountUsd": 5})
    assert r.status_code == 422
    e = r.json()["error"]
    assert e["code"] == "VALIDATION_ERROR" and "amountUsd" in e["details"]


def test_post_get_and_404(client):
    r = client.post("/api/runs", json=GOOD)
    assert r.status_code == 202
    run = Run.model_validate(r.json())
    assert run.status in ("QUEUED", "RUNNING") and run.report is None
    assert client.get(f"/api/runs/{run.runId}").status_code == 200
    r = client.get("/api/runs/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "RUN_NOT_FOUND"


def test_second_run_429(client, monkeypatch):
    monkeypatch.setenv("FAKE_DELAY_SCALE", "1")
    assert client.post("/api/runs", json=GOOD).status_code == 202
    r = client.post("/api/runs", json=GOOD)
    assert r.status_code == 429 and r.json()["error"]["code"] == "RATE_LIMITED"


def test_full_stream(client):
    run_id = client.post("/api/runs", json=GOOD).json()["runId"]
    fs = frames(client, run_id)
    seqs = [int(f["id"].split(":")[1]) for f in fs]
    assert seqs == sorted(set(seqs)) and all(f["id"].startswith(run_id + ":") for f in fs)
    events = []
    for f in fs:
        ev = EVENT.validate_python(json.loads(f["data"]))
        assert ev.type == f["event"]
        events.append(ev)
    assert events[0].type == "run.started" and events[-1].type == "run.completed"
    assert [e.verification.isApproved for e in events if e.type == "verification.completed"] == [False, True]
    Report.model_validate(events[-1].report.model_dump())
    assert events[-1].report.runId == run_id
    snap = Run.model_validate(client.get(f"/api/runs/{run_id}").json())
    assert snap.status == "COMPLETED" and snap.report is not None


def test_last_event_id(client):
    run_id = client.post("/api/runs", json=GOOD).json()["runId"]
    full = frames(client, run_id)
    part = frames(client, run_id, {"Last-Event-ID": f"{run_id}:5"})
    assert [int(f["id"].split(":")[1]) for f in part] == [int(f["id"].split(":")[1]) for f in full if int(f["id"].split(":")[1]) > 5]
    assert part[0]["id"] == f"{run_id}:6"


def test_pipeline_crash_emits_run_failed(client, monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("secret traceback")

    monkeypatch.setattr(main.pipeline, "run_pipeline", boom)
    run_id = client.post("/api/runs", json=GOOD).json()["runId"]
    last = frames(client, run_id)[-1]
    assert last["event"] == "run.failed" and "secret" not in last["data"]
    assert client.get(f"/api/runs/{run_id}").json()["status"] == "FAILED"


def test_fixture_is_valid_run():
    Run.model_validate_json((ROOT / "docs/fixtures/report.example.json").read_text())
