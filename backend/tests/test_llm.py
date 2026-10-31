import base64
import io
import os
import pandas as pd
import pytest


def test_config_exports():
    from app import config

    assert config.MODEL_FLASH == "gemini-3.8-flash"
    assert config.MODEL_ARGON == "gemini-4-argon"
    assert config.DRAWDOWN_LIMITS == {
        "CONSERVATIVE": -20.0,
        "MODERATE": -35.0,
        "AGGRESSIVE": -55.0,
    }
    assert config.MAX_HOLDING_PCT == 25.0
    assert config.CROSSCHECK_TOL_PP == 0.5
    assert config.NUMBER_TOL_PCT == 2.0
    assert config.MAX_ROUNDS == 2
    assert isinstance(config.LESSONS, bool)
    assert hasattr(config, "GEMINI_API_KEY")
    assert hasattr(config, "SEC_USER_AGENT")


def test_csv_part_and_text_part():
    from app.llm import csv_part, text_part

    # test text_part
    tp = text_part("hello world")
    assert tp == {"type": "text", "text": "hello world"}

    # test csv_part round-trip
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4.5, 5.5, 6.5]})
    part = csv_part(df)
    assert part["type"] == "document"
    assert part["mime_type"] == "text/csv"
    assert "data" in part

    raw_csv = base64.b64decode(part["data"]).decode("utf-8")
    recovered = pd.read_csv(io.StringIO(raw_csv), index_col=0)
    pd.testing.assert_frame_equal(df, recovered)

