import pytest


@pytest.fixture(autouse=True)
def _no_real_saved_runs(tmp_path, monkeypatch):
    """Every test saves runs to a temp dir; backend/data/runs is never touched."""
    from app import replay
    monkeypatch.setattr(replay, "DATA_RUNS_DIR", tmp_path / "runs")
