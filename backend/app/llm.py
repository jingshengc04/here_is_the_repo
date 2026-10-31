import base64
from dataclasses import dataclass
from typing import Any
import pandas as pd


@dataclass
class AskResult:
    data: Any
    text: str
    steps: list[dict]
    model_used: str
    duration_ms: int
    sources: list[dict]


def text_part(s: str) -> dict:
    return {"type": "text", "text": s}


def csv_part(df: pd.DataFrame) -> dict:
    csv_bytes = df.to_csv().encode("utf-8")
    data_b64 = base64.b64encode(csv_bytes).decode("utf-8")
    return {
        "type": "document",
        "mime_type": "text/csv",
        "data": data_b64,
    }


_client = None


def set_client(client: Any) -> None:
    global _client
    _client = client


def get_client() -> Any:
    global _client
    if _client is None:
        import os
        from google import genai
        from app.config import GEMINI_API_KEY
        api_key = GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
        _client = genai.Client(api_key=api_key)
    return _client
