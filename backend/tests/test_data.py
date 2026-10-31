import os
import pytest

from app.data import is_crypto, fundamentals


def test_is_crypto():
    assert is_crypto("BTC-USD") is True
    assert is_crypto("ETH-USD") is True
    assert is_crypto("sol-usd") is True
    assert is_crypto("NVDA") is False
    assert is_crypto("GLD") is False
    assert is_crypto("SPY") is False
    assert is_crypto("AGG") is False


def test_fundamentals_crypto_and_etfs_return_empty():
    assert fundamentals("BTC-USD") == {}
    assert fundamentals("ETH-USD") == {}
    assert fundamentals("GLD") == {}
    assert fundamentals("SPY") == {}
    assert fundamentals("AGG") == {}
    assert fundamentals("gld") == {}
