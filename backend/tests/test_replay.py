import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app import main
from app.models import Report, Run

GOOD = {"amountUsd": 10000, "horizonYears": 2, "riskTolerance": "MODERATE", "theme": "AI"}
NEW_INTAKE = {"amountUsd": 50000, "horizonYears": 5, "riskTolerance": "AGGRESSIVE", "theme": "Crypto & Tech"}
RUNS_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "runs"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("FAKE_DELAY_SCALE", "0")
    monkeypatch.setenv("REPLAY_SPEED", "0")
    main.RUNS.clear()
    with TestClient(main.app) as c:
        yield c


@pytest.fixture(autouse=True)
def isolated_runs_dir(tmp_path, monkeypatch):
    """Tests write saved runs to a temp dir, never the real backend/data/runs."""
    global RUNS_DATA_DIR
    from app import replay
    monkeypatch.setattr(replay, "DATA_RUNS_DIR", tmp_path / "runs")
    RUNS_DATA_DIR = tmp_path / "runs"
    yield



def frames(client, run_id):
    with client.stream("GET", f"/api/runs/{run_id}/events") as r:
        assert r.status_code == 200
        text = "".join(r.iter_text())
    out = []
    for block in text.strip().split("\n\n"):
        if block.startswith(":"):
            continue
        f = dict(line.split(": ", 1) for line in block.split("\n"))
        out.append(f)
    return out


def test_saved_run_and_replay(client):
    # 1. Run a FAKE pipeline run with FAKE_DELAY_SCALE=0
    r = client.post("/api/runs", json=GOOD)
    assert r.status_code == 202
    orig_run_id = r.json()["runId"]

    # Stream events to completion
    orig_frames = frames(client, orig_run_id)
    assert orig_frames[-1]["event"] == "run.completed"

    # Confirm JSON file is saved in backend/data/runs/{orig_run_id}.json
    saved_file = RUNS_DATA_DIR / f"{orig_run_id}.json"
    assert saved_file.is_file(), f"Expected {saved_file} to exist"

    saved_data = json.loads(saved_file.read_text(encoding="utf-8"))
    assert "intake" in saved_data
    assert "events" in saved_data
    assert saved_data["intake"]["amountUsd"] == 10000
    assert saved_data["events"][-1]["type"] == "run.completed"

    # 2. Replay it with REPLAY_SPEED=0
    r_replay = client.post(f"/api/runs?replay={orig_run_id}", json=NEW_INTAKE)
    assert r_replay.status_code == 202
    replay_run_id = r_replay.json()["runId"]
    assert replay_run_id != orig_run_id

    replay_frames = frames(client, replay_run_id)

    # Confirm same event types in same order
    orig_types = [f["event"] for f in orig_frames]
    replay_types = [f["event"] for f in replay_frames]
    assert replay_types == orig_types

    # Confirm new runId and sequential seq numbers
    for idx, f in enumerate(replay_frames, start=1):
        assert f["id"] == f"{replay_run_id}:{idx}"
        data = json.loads(f["data"])
        assert data["runId"] == replay_run_id
        assert data["seq"] == idx

    # Confirm run.completed report validates with updated runId and intake
    last_event = json.loads(replay_frames[-1]["data"])
    assert last_event["type"] == "run.completed"
    report_dict = last_event["report"]
    report = Report.model_validate(report_dict)
    assert report.runId == replay_run_id
    assert report.intake.amountUsd == 50000
    assert report.intake.riskTolerance == "AGGRESSIVE"

    # Confirm snapshot matches
    snap = Run.model_validate(client.get(f"/api/runs/{replay_run_id}").json())
    assert snap.status == "COMPLETED"
    assert snap.report is not None
    assert snap.report.runId == replay_run_id


def test_replay_path_traversal_422(client):
    # replay=../etc -> 422
    r = client.post("/api/runs?replay=../etc", json=GOOD)
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "VALIDATION_ERROR"
    assert "replay" in err.get("details", {})

    # Other illegal characters
    for bad_name in ("foo/bar", "run 123", "run$1", "run;rm", "a" * 100 + "/"):
        r = client.post(f"/api/runs?replay={bad_name}", json=GOOD)
        assert r.status_code == 422


def test_replay_unknown_name_404(client):
    r = client.post("/api/runs?replay=non_existent_run_12345", json=GOOD)
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["code"] == "RUN_NOT_FOUND"


def test_replay_from_fixture_runs(client):
    fixtures_dir = Path(__file__).resolve().parents[1] / "app" / "fixtures" / "runs"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    fixture_file = fixtures_dir / "sample_demo.json"

    # Save a run payload to fixtures
    sample_report = json.loads((Path(__file__).resolve().parents[1] / "app" / "fixtures" / "report.example.json").read_text())["report"]
    payload = {
        "intake": GOOD,
        "events": [
            {"type": "run.started", "agents": ["DISCOVERY"], "ts": "2026-10-31T12:00:00+00:00"},
            {"type": "run.completed", "report": sample_report, "ts": "2026-10-31T12:00:01+00:00"},
        ],
    }
    fixture_file.write_text(json.dumps(payload), encoding="utf-8")

    try:
        r = client.post("/api/runs?replay=sample_demo", json=NEW_INTAKE)
        assert r.status_code == 202
        run_id = r.json()["runId"]

        rep_frames = frames(client, run_id)
        assert [f["event"] for f in rep_frames] == ["run.started", "run.completed"]
        last_ev = json.loads(rep_frames[-1]["data"])
        assert last_ev["report"]["runId"] == run_id
        assert last_ev["report"]["intake"]["amountUsd"] == 50000
    finally:
        if fixture_file.exists():
            fixture_file.unlink()


def test_save_run_failure_does_not_fail_run(client, monkeypatch):
    from app import replay as replay_module

    def boom(*args, **kwargs):
        raise OSError("Disk write error")

    monkeypatch.setattr(replay_module, "save_run", boom)

    r = client.post("/api/runs", json=GOOD)
    assert r.status_code == 202
    run_id = r.json()["runId"]

    evs = frames(client, run_id)
    assert evs[-1]["event"] == "run.completed"
    snap = client.get(f"/api/runs/{run_id}").json()
    assert snap["status"] == "COMPLETED"

