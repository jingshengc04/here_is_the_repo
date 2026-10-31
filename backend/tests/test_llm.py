import os
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
