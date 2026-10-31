"""ponytail: stub for Vibe-Trading's config. Disables their opt-in loader cache (our data.py caches)."""
import os
from types import SimpleNamespace

def get_env_config():
    return SimpleNamespace(data=SimpleNamespace(vibe_trading_data_cache=False, vibe_trading_data_cache_root=None, vibe_trading_sec_ua=os.environ.get("SEC_USER_AGENT")))

def get_env_value(name, default=None):
    return os.environ.get(name, default)
