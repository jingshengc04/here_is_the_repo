#!/usr/bin/env python3
"""Live smoke test for Gemini LLM helper (Task 3).

Run with:
  cd backend && uv run --env-file <env_file> python scripts/smoke_llm.py
"""

import asyncio
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from pydantic import BaseModel, Field

from app import config
from app.llm import ask, csv_part, text_part


class DiscoverySchema(BaseModel):
    symbol: str
    summary: str


class BacktestResultSchema(BaseModel):
    totalReturnPct: float


class RiskDecisionSchema(BaseModel):
    decision: str = Field(description="'APPROVED' or 'REJECTED'")
    confidence: float


async def test_call_1_flash_search_schema() -> bool:
    print("--- Call 1: Flash + google_search + schema ---")
    start = time.perf_counter()
    try:
        res = await ask(
            model=config.MODEL_FLASH,
            input="Search for latest NVIDIA (NVDA) stock news and return JSON with symbol NVDA and a 1-sentence summary.",
            tools=("google_search",),
            schema=DiscoverySchema,
            agent="DISCOVERY",
        )
        duration = int((time.perf_counter() - start) * 1000)
        assert isinstance(res.data, DiscoverySchema), f"Expected DiscoverySchema, got {type(res.data)}"
        assert res.data.symbol.upper() == "NVDA", f"Unexpected symbol: {res.data.symbol}"
        print(f"[OK] Call 1 passed: symbol={res.data.symbol}, sources={len(res.sources)}, duration={duration}ms")
        return True
    except Exception as e:
        duration = int((time.perf_counter() - start) * 1000)
        print(f"[ERR] Call 1 failed ({duration}ms): {e}")
        return False


async def test_call_2_flash_code_execution_csv() -> bool:
    print("--- Call 2: Flash + code_execution with CSV ---")
    start = time.perf_counter()
    try:
        # 4-row weekly CSV from PLAN.md
        df = pd.DataFrame(
            {"AAA": [100, 110, 99, 120], "BBB": [50, 45, 55, 60]},
            index=pd.to_datetime(["2024-01-05", "2024-01-12", "2024-01-19", "2024-01-26"]),
        )
        df.index.name = "Date"

        prompt = (
            "The attached CSV contains weekly prices for assets AAA and BBB over 4 dates.\n"
            "Write and execute Python code using pandas to calculate the total return percentage of a portfolio "
            "that is rebalanced weekly with 60% AAA and 40% BBB over the entire period.\n"
            "Specifically:\n"
            "1. Week 1 return = 0.6 * (110/100 - 1) + 0.4 * (45/50 - 1)\n"
            "2. Week 2 return = 0.6 * (99/110 - 1) + 0.4 * (55/45 - 1)\n"
            "3. Week 3 return = 0.6 * (120/99 - 1) + 0.4 * (60/55 - 1)\n"
            "Cumulative value = (1 + r1) * (1 + r2) * (1 + r3)\n"
            "Total return pct = (Cumulative value - 1) * 100\n"
            "Print the exact number and print JSON: {\"totalReturnPct\": <float>}."
        )

        parts = [
            text_part(prompt),
            csv_part(df),
        ]

        res = await ask(
            model=config.MODEL_FLASH,
            input=parts,
            tools=("code_execution",),
            agent="FINANCIALS",
        )
        duration = int((time.perf_counter() - start) * 1000)

        # Look for return percentage in output or steps
        val = None
        for step in res.steps:
            if step.get("type") == "code_execution_result":
                r_text = str(step.get("result", ""))
                # Search for totalReturnPct or number ~22.12
                match = re.search(r"\"totalReturnPct\":\s*([0-9.]+)", r_text)
                if match:
                    val = float(match.group(1))
                    break
        if val is None:
            match = re.search(r"22\.1[0-9]*", res.text)
            if match:
                val = float(match.group(0))

        assert val is not None, f"Could not find 22.12% return in output: {res.text[:300]}"
        assert abs(val - 22.12) < 0.5, f"Expected return ≈ 22.12%, got {val}%"

        print(f"[OK] Call 2 passed: totalReturnPct={val:.2f}%, steps={len(res.steps)}, duration={duration}ms")
        return True
    except Exception as e:
        duration = int((time.perf_counter() - start) * 1000)
        print(f"[ERR] Call 2 failed ({duration}ms): {e}")
        return False


async def test_call_3_argon_schema() -> bool:
    print("--- Call 3: Argon + schema ---")
    start = time.perf_counter()
    try:
        res = await ask(
            model=config.MODEL_ARGON,
            input="Evaluate the following risk status: max drawdown is -18% under a CONSERVATIVE limit of -20%. Return decision 'APPROVED' and confidence float 0.95.",
            schema=RiskDecisionSchema,
            agent="VERIFIER",
        )
        duration = int((time.perf_counter() - start) * 1000)
        assert isinstance(res.data, RiskDecisionSchema), f"Expected RiskDecisionSchema, got {type(res.data)}"
        assert res.data.decision in ("APPROVED", "REJECTED"), f"Unexpected decision: {res.data.decision}"
        print(f"[OK] Call 3 passed: decision={res.data.decision}, model_used={res.model_used}, duration={duration}ms")
        return True
    except Exception as e:
        duration = int((time.perf_counter() - start) * 1000)
        print(f"[ERR] Call 3 failed ({duration}ms): {e}")
        return False


async def main():
    print("Starting smoke_llm live test...")
    overall_start = time.perf_counter()

    ok1 = await test_call_1_flash_search_schema()
    ok2 = await test_call_2_flash_code_execution_csv()
    ok3 = await test_call_3_argon_schema()

    total_duration = int((time.perf_counter() - overall_start) * 1000)
    print(f"\nSummary: Call 1: {'OK' if ok1 else 'ERR'}, Call 2: {'OK' if ok2 else 'ERR'}, Call 3: {'OK' if ok3 else 'ERR'} (Total: {total_duration}ms)")

    if ok1 and ok2 and ok3:
        print("ALL 3 CALLS SUCCEEDED")
        sys.exit(0)
    else:
        print("ONE OR MORE CALLS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
