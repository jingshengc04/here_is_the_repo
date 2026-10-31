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

        # Turn 3: "moderate, AI stuff" -> prompts for optional preferences
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
            assert turn3.isComplete is False
            assert "want to avoid" in turn3.reply.lower() or "ethical exclusions" in turn3.reply.lower()

        # Turn 4: "no preferences" -> completes
        t4_req = IntakeTurnRequest(messages=[
            ChatMessage(role="user", content="I have 10k"),
            ChatMessage(role="assistant", content=turn1.reply),
            ChatMessage(role="user", content="2 years"),
            ChatMessage(role="assistant", content=turn2.reply),
            ChatMessage(role="user", content="moderate, AI stuff"),
            ChatMessage(role="assistant", content=turn3.reply),
            ChatMessage(role="user", content="no preferences"),
        ])
        mock_data4 = IntakeDraft(amountUsd=10000.0, horizonYears=2, riskTolerance="MODERATE", theme="AI stuff")
        mock_res4 = AskResult(
            data=mock_data4,
            text="Great, your intake is complete!",
            steps=[],
            model_used="gemini-3.8-flash",
            duration_ms=200,
            sources=[],
        )

        with patch("app.llm.ask", new_callable=AsyncMock, return_value=mock_res4):
            turn4 = await intake_turn(t4_req)
            assert turn4.intake.amountUsd == 10000.0
            assert turn4.intake.horizonYears == 2
            assert turn4.intake.riskTolerance == "MODERATE"
            assert turn4.intake.theme == "AI stuff"
            assert turn4.isComplete is True

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


def test_normalize_symbols():
    from app.intake import normalize_symbol_or_entity

    # Cryptos
    assert normalize_symbol_or_entity("doge") == "DOGE-USD"
    assert normalize_symbol_or_entity("dogecoin") == "DOGE-USD"
    assert normalize_symbol_or_entity("pepe") == "PEPE-USD"
    assert normalize_symbol_or_entity("bitcoin") == "BTC-USD"
    assert normalize_symbol_or_entity("btc") == "BTC-USD"
    # Companies
    assert normalize_symbol_or_entity("nvidia") == "NVDA"
    assert normalize_symbol_or_entity("Tesla") == "TSLA"
    assert normalize_symbol_or_entity("tsla") == "TSLA"
    # Sectors / plain words
    assert normalize_symbol_or_entity("oil", is_exclude=True) == "oil"
    assert normalize_symbol_or_entity("tobacco", is_exclude=True) == "tobacco"


def test_intake_extracts_preferences_from_user_turn():
    """Offline acceptance: 'I want some meme coins, especially doge, about 30% crypto, no Tesla'
    -> mustInclude has DOGE-USD, styles has MEME, CRYPTO is 30, exclude has TSLA.
    """
    async def _run():
        from app.intake import intake_turn, _IntakeExtraction, _PreferencesExtraction

        req = IntakeTurnRequest(messages=[ChatMessage(
            role="user",
            content="I want some meme coins, especially doge, about 30% crypto, no Tesla"
        )])

        mock_pref = _PreferencesExtraction(
            mustInclude=["DOGE-USD"],
            exclude=["TSLA"],
            assetClassTargets={"CRYPTO": 30.0},
            styles=["MEME"],
        )
        mock_data = _IntakeExtraction(
            reply="Got it, noted your preferences. How much would you like to invest?",
            amountUsd=None,
            horizonYears=None,
            riskTolerance=None,
            theme=None,
            preferences=mock_pref,
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
            assert turn.intake.preferences is not None
            assert "DOGE-USD" in turn.intake.preferences.mustInclude
            assert "MEME" in turn.intake.preferences.styles
            assert turn.intake.preferences.assetClassTargets == {"CRYPTO": 30.0}
            assert "TSLA" in turn.intake.preferences.exclude

    asyncio.run(_run())


def test_preferences_persist_across_later_turn():
    """Preferences gathered earlier must persist across subsequent turns."""
    async def _run():
        from app.intake import intake_turn, _IntakeExtraction, _PreferencesExtraction

        # Turn 1: user expresses preferences
        t1_req = IntakeTurnRequest(messages=[
            ChatMessage(role="user", content="I want some meme coins, especially doge, about 30% crypto, no Tesla")
        ])
        mock_pref1 = _PreferencesExtraction(
            mustInclude=["DOGE-USD"],
            exclude=["TSLA"],
            assetClassTargets={"CRYPTO": 30.0},
            styles=["MEME"],
        )
        mock_data1 = _IntakeExtraction(
            reply="Got it. How much do you want to invest?",
            amountUsd=None,
            horizonYears=None,
            riskTolerance=None,
            theme=None,
            preferences=mock_pref1,
        )
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
            assert turn1.intake.preferences is not None
            assert "DOGE-USD" in turn1.intake.preferences.mustInclude

        # Turn 2: user only provides amount and horizon, model does not re-extract preferences
        t2_req = IntakeTurnRequest(messages=[
            ChatMessage(role="user", content="I want some meme coins, especially doge, about 30% crypto, no Tesla"),
            ChatMessage(role="assistant", content=turn1.reply),
            ChatMessage(role="user", content="I have 10k for 3 years"),
        ])
        mock_data2 = _IntakeExtraction(
            reply="Got it, $10,000 for 3 years. What is your risk tolerance?",
            amountUsd=10000.0,
            horizonYears=3,
            riskTolerance=None,
            theme=None,
            preferences=None,  # model omitted preferences on turn 2
        )
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
            assert turn2.intake.horizonYears == 3
            # Preferences from Turn 1 must persist!
            assert turn2.intake.preferences is not None
            assert "DOGE-USD" in turn2.intake.preferences.mustInclude
            assert "MEME" in turn2.intake.preferences.styles
            assert turn2.intake.preferences.assetClassTargets == {"CRYPTO": 30.0}
            assert "TSLA" in turn2.intake.preferences.exclude

    asyncio.run(_run())


def test_conflict_warning_conservative_meme():
    """Conflict warning appears once when CONSERVATIVE + MEME / meme coins / crypto > 15%."""
    async def _run():
        from app.intake import intake_turn, _IntakeExtraction, _PreferencesExtraction

        req = IntakeTurnRequest(messages=[
            ChatMessage(role="user", content="I have 5k for 3 years, conservative"),
            ChatMessage(role="assistant", content="Anything you specifically want or want to avoid?"),
            ChatMessage(role="user", content="I want some meme coins like doge and pepe, and no oil companies"),
        ])
        mock_pref = _PreferencesExtraction(
            mustInclude=["DOGE-USD", "PEPE-USD"],
            exclude=["oil"],
            assetClassTargets=None,
            styles=["MEME"],
        )
        mock_data = _IntakeExtraction(
            reply="Understood.",
            amountUsd=5000.0,
            horizonYears=3,
            riskTolerance="CONSERVATIVE",
            theme=None,
            preferences=mock_pref,
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
            assert "Meme coins can drop 80%+; with a Conservative profile they'll be kept small (a few %). Want to switch to Aggressive?" in turn.reply
            assert turn.isComplete is False
            assert turn.intake.preferences is not None
            assert "DOGE-USD" in turn.intake.preferences.mustInclude
            assert "PEPE-USD" in turn.intake.preferences.mustInclude
            assert "oil" in turn.intake.preferences.exclude

    asyncio.run(_run())


def test_conflict_warning_switch_to_aggressive():
    """When user replies 'yes switch to aggressive', riskTolerance updates to AGGRESSIVE and completes."""
    async def _run():
        from app.intake import intake_turn, _IntakeExtraction, _PreferencesExtraction

        warning_msg = "Meme coins can drop 80%+; with a Conservative profile they'll be kept small (a few %). Want to switch to Aggressive?"
        req = IntakeTurnRequest(messages=[
            ChatMessage(role="user", content="I have 5k for 3 years, conservative"),
            ChatMessage(role="assistant", content="Anything you specifically want or want to avoid?"),
            ChatMessage(role="user", content="I want some meme coins like doge and pepe, and no oil companies"),
            ChatMessage(role="assistant", content=warning_msg),
            ChatMessage(role="user", content="yes, switch to aggressive"),
        ])
        mock_pref = _PreferencesExtraction(
            mustInclude=["DOGE-USD", "PEPE-USD"],
            exclude=["oil"],
            assetClassTargets=None,
            styles=["MEME"],
        )
        mock_data = _IntakeExtraction(
            reply="Great!",
            amountUsd=5000.0,
            horizonYears=3,
            riskTolerance="CONSERVATIVE",
            theme=None,
            preferences=mock_pref,
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
            assert turn.intake.riskTolerance == "AGGRESSIVE"
            assert turn.isComplete is True
            assert "DOGE-USD" in turn.intake.preferences.mustInclude

    asyncio.run(_run())


def test_conflict_warning_keep_conservative():
    """When user replies 'no, keep conservative', riskTolerance remains CONSERVATIVE and completes."""
    async def _run():
        from app.intake import intake_turn, _IntakeExtraction, _PreferencesExtraction

        warning_msg = "Meme coins can drop 80%+; with a Conservative profile they'll be kept small (a few %). Want to switch to Aggressive?"
        req = IntakeTurnRequest(messages=[
            ChatMessage(role="user", content="I have 5k for 3 years, conservative"),
            ChatMessage(role="assistant", content="Anything you specifically want or want to avoid?"),
            ChatMessage(role="user", content="I want some meme coins like doge and pepe, and no oil companies"),
            ChatMessage(role="assistant", content=warning_msg),
            ChatMessage(role="user", content="no, keep conservative"),
        ])
        mock_pref = _PreferencesExtraction(
            mustInclude=["DOGE-USD", "PEPE-USD"],
            exclude=["oil"],
            assetClassTargets=None,
            styles=["MEME"],
        )
        mock_data = _IntakeExtraction(
            reply="Understood.",
            amountUsd=5000.0,
            horizonYears=3,
            riskTolerance="CONSERVATIVE",
            theme=None,
            preferences=mock_pref,
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
            assert turn.intake.riskTolerance == "CONSERVATIVE"
            assert turn.isComplete is True
            assert "DOGE-USD" in turn.intake.preferences.mustInclude

    asyncio.run(_run())


def test_no_preferences_completes_intake():
    """A reply like 'no' or 'nothing' to the preferences prompt completes the intake."""
    async def _run():
        from app.intake import intake_turn, _IntakeExtraction

        req = IntakeTurnRequest(messages=[
            ChatMessage(role="user", content="10k, 2 years, moderate"),
            ChatMessage(role="assistant", content="Anything you specifically want or want to avoid, like certain coins, companies, dividends, or ethical exclusions?"),
            ChatMessage(role="user", content="no preferences"),
        ])
        mock_data = _IntakeExtraction(
            reply="All set!",
            amountUsd=10000.0,
            horizonYears=2,
            riskTolerance="MODERATE",
            theme=None,
            preferences=None,
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
            assert turn.isComplete is True
            assert turn.intake.amountUsd == 10000.0
            assert turn.intake.horizonYears == 2
            assert turn.intake.riskTolerance == "MODERATE"
            assert turn.intake.preferences is None

    asyncio.run(_run())


def test_invalid_targets_sum_over_100_dropped_and_asked_again():
    """Invalid parts (targets summing over 100) are dropped and asked about again. Never crash."""
    async def _run():
        from app.intake import intake_turn, _IntakeExtraction, _PreferencesExtraction

        req = IntakeTurnRequest(messages=[
            ChatMessage(role="user", content="I have 10k for 5 years, aggressive, but want 70% crypto and 50% stocks")
        ])
        # Model extracts targets summing to 120 > 100
        mock_pref = _PreferencesExtraction(
            mustInclude=[],
            exclude=[],
            assetClassTargets={"CRYPTO": 70.0, "STOCKS": 50.0},
            styles=[],
            notes=None,
        )
        mock_data = _IntakeExtraction(
            reply="Noted.",
            amountUsd=10000.0,
            horizonYears=5,
            riskTolerance="AGGRESSIVE",
            theme=None,
            preferences=mock_pref,
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
            # The invalid target allocations must be dropped (None)
            if turn.intake.preferences is not None:
                assert turn.intake.preferences.assetClassTargets is None
            # Must ask about it again
            assert "100%" in turn.reply
            # Must not be complete while targets are invalid
            assert turn.isComplete is False

    asyncio.run(_run())


@pytest.mark.skipif(os.getenv("LIVE") != "1", reason="Live intake test requires LIVE=1")
def test_live_intake_4_turns():
    """Live integration test: runs 4 turns for real with Gemini Flash."""
    async def _run():
        from app.intake import intake_turn

        # Turn 1: "I have 5k, want growth"
        t1_req = IntakeTurnRequest(messages=[ChatMessage(role="user", content="I have 5k, want growth")])
        turn1 = await intake_turn(t1_req)
        print("\n[Turn 1 Reply]:", turn1.reply)
        print("[Turn 1 Draft]:", turn1.intake.model_dump())
        assert turn1.intake.amountUsd == 5000.0
        assert turn1.isComplete is False

        # Turn 2: "3 years"
        t2_req = IntakeTurnRequest(messages=[
            ChatMessage(role="user", content="I have 5k, want growth"),
            ChatMessage(role="assistant", content=turn1.reply),
            ChatMessage(role="user", content="3 years"),
        ])
        turn2 = await intake_turn(t2_req)
        print("\n[Turn 2 Reply]:", turn2.reply)
        print("[Turn 2 Draft]:", turn2.intake.model_dump())
        assert turn2.intake.amountUsd == 5000.0
        assert turn2.intake.horizonYears == 3
        assert turn2.isComplete is False

        # Turn 3: "fairly cautious actually"
        t3_req = IntakeTurnRequest(messages=[
            ChatMessage(role="user", content="I have 5k, want growth"),
            ChatMessage(role="assistant", content=turn1.reply),
            ChatMessage(role="user", content="3 years"),
            ChatMessage(role="assistant", content=turn2.reply),
            ChatMessage(role="user", content="fairly cautious actually"),
        ])
        turn3 = await intake_turn(t3_req)
        print("\n[Turn 3 Reply]:", turn3.reply)
        print("[Turn 3 Draft]:", turn3.intake.model_dump())
        assert turn3.intake.amountUsd == 5000.0
        assert turn3.intake.horizonYears == 3
        assert turn3.intake.riskTolerance == "CONSERVATIVE"

        # Turn 4: "I want some meme coins like doge and pepe, and no oil companies"
        t4_req = IntakeTurnRequest(messages=[
            ChatMessage(role="user", content="I have 5k, want growth"),
            ChatMessage(role="assistant", content=turn1.reply),
            ChatMessage(role="user", content="3 years"),
            ChatMessage(role="assistant", content=turn2.reply),
            ChatMessage(role="user", content="fairly cautious actually"),
            ChatMessage(role="assistant", content=turn3.reply),
            ChatMessage(role="user", content="I want some meme coins like doge and pepe, and no oil companies"),
        ])
        turn4 = await intake_turn(t4_req)
        print("\n[Turn 4 Reply]:", turn4.reply)
        print("[Turn 4 Draft]:", turn4.intake.model_dump())
        assert turn4.intake.amountUsd == 5000.0
        assert turn4.intake.horizonYears == 3
        assert turn4.intake.riskTolerance == "CONSERVATIVE"
        assert turn4.intake.preferences is not None
        assert "DOGE-USD" in turn4.intake.preferences.mustInclude
        assert "PEPE-USD" in turn4.intake.preferences.mustInclude
        assert "oil" in turn4.intake.preferences.exclude
        assert "switch to Aggressive" in turn4.reply or "switch to aggressive" in turn4.reply.lower()

    asyncio.run(_run())


