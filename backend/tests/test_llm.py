import asyncio
import base64
import io
import json
from dataclasses import dataclass, field
from typing import Any, Optional
from pydantic import BaseModel
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

    tp = text_part("hello world")
    assert tp == {"type": "text", "text": "hello world"}

    df = pd.DataFrame({"A": [1, 2, 3], "B": [4.5, 5.5, 6.5]})
    part = csv_part(df)
    assert part["type"] == "document"
    assert part["mime_type"] == "text/csv"
    assert "data" in part

    raw_csv = base64.b64decode(part["data"]).decode("utf-8")
    recovered = pd.read_csv(io.StringIO(raw_csv), index_col=0)
    pd.testing.assert_frame_equal(df, recovered)


# --- Test doubles for offline LLM tests ---

@dataclass
class FakeStep:
    type: str
    arguments: Optional[Any] = None
    result: Optional[Any] = None
    content: Optional[list] = None
    sources: Optional[list] = None


@dataclass
class FakeInteraction:
    output_text: str
    steps: list = field(default_factory=list)


class FakeAPIError(Exception):
    def __init__(self, code: int, message: str = "API error"):
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class FakeInteractions:
    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return await self.handler(kwargs)


class FakeAio:
    def __init__(self, interactions):
        self.interactions = interactions


class FakeClient:
    def __init__(self, handler):
        self.aio = FakeAio(FakeInteractions(handler))


class SimpleSchema(BaseModel):
    name: str
    score: int


def test_schema_validation_success():
    from app.llm import ask, set_client
    from app import llm

    llm._BACKOFF_DELAYS = [0.001, 0.002, 0.004]

    async def _test():
        async def handler(kwargs):
            assert kwargs["model"] == "gemini-3.8-flash"
            assert "response_format" in kwargs
            return FakeInteraction(
                output_text=json.dumps({"name": "test_asset", "score": 95}),
                steps=[FakeStep(type="thought")],
            )

        client = FakeClient(handler)
        set_client(client)

        res = await ask(
            model="gemini-3.8-flash",
            input="Analyze asset",
            schema=SimpleSchema,
            agent="TEST_AGENT",
        )

        assert isinstance(res.data, SimpleSchema)
        assert res.data.name == "test_asset"
        assert res.data.score == 95
        assert res.model_used == "gemini-3.8-flash"
        assert len(client.aio.interactions.calls) == 1

    asyncio.run(_test())


def test_schema_repair_retry():
    from app.llm import ask, set_client
    from app import llm

    llm._BACKOFF_DELAYS = [0.001, 0.002, 0.004]
    call_count = 0

    async def _test():
        nonlocal call_count

        async def handler(kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return FakeInteraction(output_text="not valid json")
            return FakeInteraction(
                output_text=json.dumps({"name": "repaired_asset", "score": 100})
            )

        client = FakeClient(handler)
        set_client(client)

        res = await ask(
            model="gemini-3.8-flash",
            input="Analyze asset",
            schema=SimpleSchema,
            agent="TEST_AGENT",
        )

        assert call_count == 2
        assert isinstance(res.data, SimpleSchema)
        assert res.data.name == "repaired_asset"
        assert res.data.score == 100
        second_call_input = client.aio.interactions.calls[1]["input"]
        assert "Validation error" in str(second_call_input) or "validation" in str(second_call_input).lower()

    asyncio.run(_test())


def test_503_retry_then_success():
    from app.llm import ask, set_client
    from app import llm

    llm._BACKOFF_DELAYS = [0.001, 0.002, 0.004]
    call_count = 0

    async def _test():
        nonlocal call_count

        async def handler(kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise FakeAPIError(503, "Service Unavailable")
            return FakeInteraction(output_text="Success after retry")

        client = FakeClient(handler)
        set_client(client)

        res = await ask(
            model="gemini-3.8-flash",
            input="Hello",
            agent="TEST_AGENT",
        )

        assert call_count == 2
        assert res.text == "Success after retry"
        assert res.data == "Success after retry"

    asyncio.run(_test())


def test_argon_503_fallback_to_flash():
    from app.llm import ask, set_client
    from app import config, llm

    llm._BACKOFF_DELAYS = [0.001, 0.002, 0.004]

    async def _test():
        async def handler(kwargs):
            if kwargs["model"] == config.MODEL_ARGON:
                raise FakeAPIError(503, "Argon 503")
            elif kwargs["model"] == config.MODEL_FLASH:
                return FakeInteraction(output_text="Flash fallback succeeded")
            raise ValueError(f"Unexpected model: {kwargs['model']}")

        client = FakeClient(handler)
        set_client(client)

        res = await ask(
            model=config.MODEL_ARGON,
            input="Run allocation",
            agent="ORCHESTRATOR",
        )

        assert res.text == "Flash fallback succeeded"
        assert res.model_used == config.MODEL_FLASH
        assert len(client.aio.interactions.calls) == 4
        for c in client.aio.interactions.calls[:3]:
            assert c["model"] == config.MODEL_ARGON
        assert client.aio.interactions.calls[3]["model"] == config.MODEL_FLASH

    asyncio.run(_test())


def test_on_step_callbacks_and_sources():
    from app.llm import ask, set_client
    from app import llm

    llm._BACKOFF_DELAYS = [0.001, 0.002, 0.004]

    @dataclass
    class SearchArgs:
        queries: list

    @dataclass
    class CodeArgs:
        code: str

    @dataclass
    class URLCitation:
        title: str
        url: str

    @dataclass
    class TextContent:
        text: str
        annotations: list

    steps = [
        FakeStep(
            type="google_search_call",
            arguments=SearchArgs(queries=["top semiconductor stocks 2026"]),
        ),
        FakeStep(
            type="google_search_result",
            result=[{"title": "Search Result Title", "url": "https://example.com/search"}],
        ),
        FakeStep(
            type="code_execution_call",
            arguments=CodeArgs(code="print('running python')"),
        ),
        FakeStep(
            type="code_execution_result",
            result="running python\n",
        ),
        FakeStep(
            type="model_output",
            content=[
                TextContent(
                    text="Here is the output",
                    annotations=[
                        URLCitation(title="Semis Report", url="https://example.com/semis")
                    ],
                )
            ],
        ),
    ]

    async def _test():
        async def handler(kwargs):
            return FakeInteraction(output_text="Done", steps=steps)

        client = FakeClient(handler)
        set_client(client)

        fired_steps = []

        def on_step(s):
            fired_steps.append(s)

        res = await ask(
            model="gemini-3.8-flash",
            input="Find info",
            tools=("google_search", "code_execution"),
            agent="DISCOVERY",
            on_step=on_step,
        )

        assert len(fired_steps) == 5
        types = [s["type"] for s in fired_steps]
        assert "google_search_call" in types
        assert "google_search_result" in types
        assert "code_execution_call" in types
        assert "code_execution_result" in types

        search_step = next(s for s in fired_steps if s["type"] == "google_search_call")
        assert search_step.get("query") == "Searching: top semiconductor stocks 2026"

        code_step = next(s for s in fired_steps if s["type"] == "code_execution_call")
        assert "print('running python')" in code_step.get("code")

        urls = [s["url"] for s in res.sources]
        assert "https://example.com/search" in urls or "https://example.com/semis" in urls

    asyncio.run(_test())


def test_source_deduplication_and_citations():
    """An offline test with a fake model_output carrying 2 annotations (one duplicate URL) -> 1 source after dedup."""
    from app.llm import ask, set_client
    from app import llm

    llm._BACKOFF_DELAYS = [0.001, 0.002, 0.004]

    @dataclass
    class URLCitation:
        title: str
        url: str
        start_index: int
        end_index: int
        type: str = "url_citation"

    @dataclass
    class TextContent:
        text: str
        annotations: list

    steps = [
        FakeStep(
            type="model_output",
            content=[
                TextContent(
                    text="NVIDIA makes GPUs and systems.",
                    annotations=[
                        URLCitation(title="Nvidia Home", url="https://nvidia.com", start_index=0, end_index=15),
                        URLCitation(title="Nvidia Products", url="https://nvidia.com", start_index=16, end_index=30),
                    ],
                )
            ],
        ),
    ]

    async def _test():
        async def handler(kwargs):
            return FakeInteraction(output_text="NVIDIA makes GPUs and systems.", steps=steps)

        client = FakeClient(handler)
        set_client(client)

        res = await ask(
            model="gemini-3.8-flash",
            input="NVIDIA overview",
            tools=("google_search",),
            agent="DISCOVERY",
        )

        # Sources should be deduplicated to 1 item
        assert len(res.sources) == 1
        assert res.sources[0] == {"title": "Nvidia Home", "url": "https://nvidia.com"}

        # Citations should preserve both distinct spans
        assert hasattr(res, "citations")
        assert len(res.citations) == 2
        assert res.citations[0] == {
            "title": "Nvidia Home",
            "url": "https://nvidia.com",
            "start": 0,
            "end": 15,
        }
        assert res.citations[1] == {
            "title": "Nvidia Products",
            "url": "https://nvidia.com",
            "start": 16,
            "end": 30,
        }

    asyncio.run(_test())


def test_search_with_schema_two_step_fallback():
    """When google_search in tools and schema is set, make call without response_format first, then cheap schema conversion call."""
    from app.llm import ask, set_client
    from app import config, llm

    llm._BACKOFF_DELAYS = [0.001, 0.002, 0.004]

    @dataclass
    class URLCitation:
        title: str
        url: str
        start_index: int
        end_index: int
        type: str = "url_citation"

    @dataclass
    class TextContent:
        text: str
        annotations: list

    grounded_steps = [
        FakeStep(
            type="google_search_call",
            arguments={"queries": ["nvda news", "nvidia stock"]},
        ),
        FakeStep(
            type="model_output",
            content=[
                TextContent(
                    text="NVIDIA reports record revenue.",
                    annotations=[
                        URLCitation(title="Nvidia PR", url="https://nvidianews.nvidia.com", start_index=0, end_index=29)
                    ],
                )
            ],
        ),
    ]

    call_index = 0

    async def _test():
        nonlocal call_index

        async def handler(kwargs):
            nonlocal call_index
            call_index += 1
            if call_index == 1:
                # First call: must have google_search and NO response_format
                assert "tools" in kwargs
                assert "response_format" not in kwargs
                return FakeInteraction(output_text="NVIDIA reports record revenue.", steps=grounded_steps)
            elif call_index == 2:
                # Second call: cheap conversion using MODEL_FLASH, no tools, WITH response_format
                assert kwargs["model"] == config.MODEL_FLASH
                assert "tools" not in kwargs
                assert "response_format" in kwargs
                return FakeInteraction(output_text=json.dumps({"name": "NVDA", "score": 98}))
            raise ValueError(f"Unexpected call {call_index}")

        client = FakeClient(handler)
        set_client(client)

        res = await ask(
            model=config.MODEL_ARGON,
            input="Search for NVDA news and return schema",
            tools=("google_search",),
            schema=SimpleSchema,
            agent="DISCOVERY",
        )

        assert call_index == 2
        assert isinstance(res.data, SimpleSchema)
        assert res.data.name == "NVDA"
        assert res.data.score == 98
        # Sources and citations should come from the first (grounded) call!
        assert len(res.sources) == 1
        assert res.sources[0]["url"] == "https://nvidianews.nvidia.com"
        assert len(res.citations) == 1
        assert res.citations[0]["start"] == 0

    asyncio.run(_test())

