import os
from pathlib import Path
from dotenv import load_dotenv

# Load backend/.env if the file exists (path relative to this file)
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.is_file():
    load_dotenv(dotenv_path=_env_path)

MODEL_FLASH = "gemini-3.8-flash"
MODEL_ARGON = "gemini-4-argon"

DRAWDOWN_LIMITS = {
    "CONSERVATIVE": -20.0,
    "MODERATE": -35.0,
    "AGGRESSIVE": -55.0,
}
MAX_HOLDING_PCT = 25.0
CROSSCHECK_TOL_PP = 0.5
NUMBER_TOL_PCT = 2.0
MAX_ROUNDS = 2
LESSONS = os.getenv("LESSONS", "off") == "on"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT")
