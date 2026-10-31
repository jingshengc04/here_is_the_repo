import asyncio
import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from app import main
from app.llm import AskResult
from app.models import ChatMessage, IntakeDraft, IntakeTurn, IntakeTurnRequest


@pytest.fixture
def client():
    return TestClient(main.app)


def test_intake_3_turn_conversation():
    """Scripted 3-turn conversation ends with isComplete=True and parsed draft."""
    async def _run():
        from app.intake import intake_turn

        # Turn 1: "I have 10k"
        t1_req = IntakeTurnRequest(messages=[ChatMessage(role="user", content="I have 10k")])
        mock_data1 = IntakeDraft(amountUsd=10000.0, horizonYears=None, riskTolerance=None, theme=None)
        mock_res1 = AskResult(
            data=mock_data1,
            text="",
            steps=[],
            model_used="gemini-3.8-flash",
            duration_ms=200,
            sources=[],
        )

        with patch("app.llm.ask", new_callable=AsyncMock, return_value=mock_res1):
            turn1 = await intake_turn(t1_req)
            assert turn1.intake.amountUsd == 10000.0
            assert turn1.intake.horizonYears is None
            assert turn1.isComplete is False
            assert len(turn1.reply) > 0

        # Turn 2: "2 years"
        t2_req = IntakeTurnRequest(messages=[
            ChatMessage(role="user", content="I have 10k"),
            ChatMessage(role="assistant", content=turn1.reply),
            ChatMessage(role="user", content="2 years"),
        ])
        mock_data2 = IntakeDraft(amountUsd=10000.0, horizonYears=2, riskTolerance=None, theme=None)
        mock_res2 = AskResult(
            data=mock_data2,
            text="",
            steps=[],
            model_used="gemini-3.8-flash",
            duration_ms=200,
            sources=[],
        )

        with patch("app.llm.ask", new_callable=AsyncMock, return_value=mock_res2):
            turn2 = await intake_turn(t2_req)
            assert turn2.intake.amountUsd == 10000.0
            assert turn2.intake.horizonYears == 2
            assert turn2.intake.riskTolerance is None
            assert turn2.isComplete is False

        # Turn 3: "moderate, AI stuff"
        t3_req = IntakeTurnRequest(messages=[
            ChatMessage(role="user", content="I have 10k"),
            ChatMessage(role="assistant", content=turn1.reply),
            ChatMessage(role="user", content="2 years"),
            ChatMessage(role="assistant", content=turn2.reply),
            ChatMessage(role="user", content="moderate, AI stuff"),
        ])
        mock_data3 = IntakeDraft(amountUsd=10000.0, horizonYears=2, riskTolerance="MODERATE", theme="AI stuff")
        mock_res3 = AskResult(
            data=mock_data3,
            text="",
            steps=[],
            model_used="gemini-3.8-flash",
            duration_ms=200,
            sources=[],
        )

        with patch("app.llm.ask", new_callable=AsyncMock, return_value=mock_res3):
            turn3 = await intake_turn(t3_req)
            assert turn3.intake.amountUsd == 10000.0
            assert turn3.intake.horizonYears == 2
            assert turn3.intake.riskTolerance == "MODERATE"
            assert turn3.intake.theme == "AI stuff"
            assert turn3.isComplete is True

    asyncio.run(_run())


def test_out_of_range_horizon_reset_to_null():
    """Horizon of 15 is out of range (1-10); set to null and ask again."""
    async def _run():
        from app.intake import intake_turn

        req = IntakeTurnRequest(messages=[ChatMessage(role="user", content="I have 10k for 15 years")])
        mock_data = IntakeDraft(amountUsd=10000.0, horizonYears=15, riskTolerance=None, theme=None)
        mock_res = AskResult(
            data=mock_data,
            text="",
            steps=[],
            model_used="gemini-3.8-flash",
            duration_ms=200,
            sources=[],
        )

        with patch("app.llm.ask", new_callable=AsyncMock, return_value=mock_res):
            turn = await intake_turn(req)
            assert turn.intake.horizonYears is None
            assert turn.intake.amountUsd == 10000.0
            assert turn.isComplete is False
            assert "horizon" in turn.reply.lower() or "year" in turn.reply.lower()

    asyncio.run(_run())


def test_execution_guard_buy_nvda_now():
    """If user asks to place a trade, reply mentions it does not place trades."""
    async def _run():
        from app.intake import intake_turn

        req = IntakeTurnRequest(messages=[ChatMessage(role="user", content="buy NVDA now")])
        mock_data = IntakeDraft(amountUsd=None, horizonYears=None, riskTolerance=None, theme=None)
        mock_res = AskResult(
            data=mock_data,
            text="I cannot place trades.",
            steps=[],
            model_used="gemini-3.8-flash",
            duration_ms=200,
            sources=[],
        )

        with patch("app.llm.ask", new_callable=AsyncMock, return_value=mock_res):
            turn = await intake_turn(req)
            reply_lower = turn.reply.lower()
            assert "trade" in reply_lower

    asyncio.run(_run())


def test_no_duplicate_trade_guard_when_model_already_mentions_trade():
    """If model reply already contains the trade guard, do not duplicate it."""
    async def _run():
        from app.intake import intake_turn, _IntakeExtraction

        req = IntakeTurnRequest(messages=[ChatMessage(
            role="user",
            content="AI infrastructure. Also can you just buy NVDA for me now?"
        )])
        model_reply = "This tool researches and backtests portfolios rather than executing trades or offering financial advice. How much would you like to invest?"
        mock_data = _IntakeExtraction(
            reply=model_reply,
            amountUsd=None,
            horizonYears=None,
            riskTolerance=None,
            theme="AI infrastructure",
        )
        mock_res = AskResult(
            data=mock_data,
            text="",
            steps=[],
            model_used="gemini-3.8-flash",
            duration_ms=200,
            sources=[],
        )

        with patch("app.llm.ask", new_callable=AsyncMock, return_value=mock_res):
            turn = await intake_turn(req)
            reply_lower = turn.reply.lower()
            assert reply_lower.count("trade") == 1
            assert "how much" in reply_lower

    asyncio.run(_run())


def test_intake_route_over_limit_21_messages_422(client):
    """21 messages returns 422 VALIDATION_ERROR via TestClient."""
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
        for i in range(21)
    ]
    r = client.post("/api/intake/messages", json={"messages": messages})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "VALIDATION_ERROR"


def test_intake_route_valid(client):
    """POST /api/intake/messages returns 200 with IntakeTurn."""
    from app.intake import intake_turn

    mock_turn = IntakeTurn(
        reply="How much are you looking to invest?",
        intake=IntakeDraft(),
        isComplete=False,
    )
    with patch("app.intake.intake_turn", new_callable=AsyncMock, return_value=mock_turn):
        r = client.post("/api/intake/messages", json={"messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200
        data = r.json()
        assert data["reply"] == "How much are you looking to invest?"
        assert data["isComplete"] is False


def test_parse_explicit_amounts():
    from app.intake import parse_explicit_amount

    assert parse_explicit_amount("I have 10k to invest") == 10000.0
    assert parse_explicit_amount("Investing $10,000") == 10000.0
    assert parse_explicit_amount("I have ten thousand dollars") == 10000.0
    assert parse_explicit_amount("Got 50K") == 50000.0
    assert parse_explicit_amount("About 1.5M") == 1500000.0
    assert parse_explicit_amount("Just $500") == 500.0
    assert parse_explicit_amount("Looking for 2 years") is None


def test_out_of_range_amount_reset_to_null():
    async def _run():
        from app.intake import intake_turn

        # Amount < 100 is out of range
        req = IntakeTurnRequest(messages=[ChatMessage(role="user", content="I have $50")])
        mock_data = IntakeDraft(amountUsd=50.0, horizonYears=2, riskTolerance="MODERATE", theme=None)
        mock_res = AskResult(
            data=mock_data,
            text="Please provide a valid amount.",
            steps=[],
            model_used="gemini-3.8-flash",
            duration_ms=200,
            sources=[],
        )

        with patch("app.llm.ask", new_callable=AsyncMock, return_value=mock_res):
            turn = await intake_turn(req)
            assert turn.intake.amountUsd is None
            assert turn.isComplete is False

    asyncio.run(_run())


@pytest.mark.skipif(os.getenv("LIVE") != "1", reason="Live intake test requires LIVE=1")
def test_live_intake_3_turns():
    """Live integration test: runs 3 turns for real with Gemini Flash."""
    async def _run():
        from app.intake import intake_turn

        # Turn 1
        t1_req = IntakeTurnRequest(messages=[ChatMessage(role="user", content="I have 10k")])
        turn1 = await intake_turn(t1_req)
        print("\n[Turn 1 Reply]:", turn1.reply)
        print("[Turn 1 Draft]:", turn1.intake.model_dump())
        assert turn1.intake.amountUsd == 10000.0
        assert turn1.isComplete is False

        # Turn 2
        t2_req = IntakeTurnRequest(messages=[
            ChatMessage(role="user", content="I have 10k"),
            ChatMessage(role="assistant", content=turn1.reply),
            ChatMessage(role="user", content="2 years"),
        ])
        turn2 = await intake_turn(t2_req)
        print("\n[Turn 2 Reply]:", turn2.reply)
        print("[Turn 2 Draft]:", turn2.intake.model_dump())
        assert turn2.intake.amountUsd == 10000.0
        assert turn2.intake.horizonYears == 2
        assert turn2.isComplete is False

        # Turn 3
        t3_req = IntakeTurnRequest(messages=[
            ChatMessage(role="user", content="I have 10k"),
            ChatMessage(role="assistant", content=turn1.reply),
            ChatMessage(role="user", content="2 years"),
            ChatMessage(role="assistant", content=turn2.reply),
            ChatMessage(role="user", content="moderate, AI stuff"),
        ])
        turn3 = await intake_turn(t3_req)
        print("\n[Turn 3 Reply]:", turn3.reply)
        print("[Turn 3 Draft]:", turn3.intake.model_dump())
        assert turn3.intake.amountUsd == 10000.0
        assert turn3.intake.horizonYears == 2
        assert turn3.intake.riskTolerance == "MODERATE"
        assert turn3.isComplete is True

    asyncio.run(_run())


