import asyncio
import base64
import inspect
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Union
import pandas as pd

from app import config

logger = logging.getLogger("app.llm")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

_BACKOFF_DELAYS = [1.0, 2.0, 4.0]


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
        api_key = config.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
        _client = genai.Client(api_key=api_key)
    return _client


def _is_retryable_error(err: Exception) -> bool:
    code = getattr(err, "code", None)
    if code is None:
        code = getattr(err, "status_code", None)
    if code in (429, 500, 503):
        return True
    err_str = str(err)
    if any(f"{c}" in err_str for c in ("429", "500", "503")):
        return True
    return False


def _is_503(err: Exception) -> bool:
    code = getattr(err, "code", None) or getattr(err, "status_code", None)
    if code == 503:
        return True
    return "503" in str(err)


def _truncate_and_strip(val: Any, max_len: int = 200) -> str:
    if isinstance(val, str):
        s = val
    elif isinstance(val, list):
        clean_list = []
        for item in val:
            if isinstance(item, dict):
                clean_item = dict(item)
                if "data" in clean_item and isinstance(clean_item["data"], str) and len(clean_item["data"]) > 50:
                    clean_item["data"] = f"<base64 {len(clean_item['data'])} chars>"
                clean_list.append(clean_item)
            else:
                clean_list.append(item)
        s = str(clean_list)
    elif isinstance(val, dict):
        clean_dict = dict(val)
        if "data" in clean_dict and isinstance(clean_dict["data"], str) and len(clean_dict["data"]) > 50:
            clean_dict["data"] = f"<base64 {len(clean_dict['data'])} chars>"
        s = str(clean_dict)
    else:
        s = str(val)

    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def _simplify_step(step: Any) -> dict:
    step_type = getattr(step, "type", None)
    if step_type is None and isinstance(step, dict):
        step_type = step.get("type")

    s_dict: dict[str, Any] = {"type": step_type or "unknown"}

    args = getattr(step, "arguments", None)
    if args is None and isinstance(step, dict):
        args = step.get("arguments")

    # Extract code if present
    code = getattr(args, "code", None) if args else None
    if code is None and isinstance(args, dict):
        code = args.get("code")
    if code is None:
        code = getattr(step, "code", None)
    if code is None and isinstance(step, dict):
        code = step.get("code")
    if code is not None:
        s_dict["code"] = code

    # Extract result if present
    res = getattr(step, "result", None)
    if res is None and isinstance(step, dict):
        res = step.get("result")
    if res is not None:
        s_dict["result"] = res

    # Extract query if present
    query = None
    queries = getattr(args, "queries", None) if args else None
    if queries is None and isinstance(args, dict):
        queries = args.get("queries")
    if queries:
        query = queries[0] if (isinstance(queries, list) and len(queries) == 1) else (", ".join(queries) if isinstance(queries, list) else str(queries))
    if query is None and args:
        query = getattr(args, "query", None) if not isinstance(args, dict) else args.get("query")
    if query is None:
        query = getattr(step, "query", None) if not isinstance(step, dict) else step.get("query")
    if query is not None:
        s_dict["query"] = query

    return s_dict


def _extract_sources(steps: list[Any]) -> list[dict]:
    extracted = []
    for step in steps:
        # Check model_output content annotations
        content = getattr(step, "content", None)
        if content is None and isinstance(step, dict):
            content = step.get("content")
        if content and isinstance(content, list):
            for c in content:
                annotations = getattr(c, "annotations", None)
                if annotations is None and isinstance(c, dict):
                    annotations = c.get("annotations")
                if annotations and isinstance(annotations, list):
                    for a in annotations:
                        url = getattr(a, "url", None) or (a.get("url") if isinstance(a, dict) else None)
                        title = getattr(a, "title", None) or (a.get("title") if isinstance(a, dict) else "")
                        if url:
                            extracted.append({"title": str(title or ""), "url": str(url)})

        # Check step.result (e.g. google_search_result)
        result = getattr(step, "result", None)
        if result is None and isinstance(step, dict):
            result = step.get("result")
        if result and isinstance(result, list):
            for item in result:
                url = getattr(item, "url", None) or (item.get("url") if isinstance(item, dict) else None)
                title = getattr(item, "title", None) or (item.get("title") if isinstance(item, dict) else "")
                if url:
                    extracted.append({"title": str(title or ""), "url": str(url)})

        # Check step.sources
        sources = getattr(step, "sources", None)
        if sources is None and isinstance(step, dict):
            sources = step.get("sources")
        if sources and isinstance(sources, list):
            for item in sources:
                url = getattr(item, "url", None) or (item.get("url") if isinstance(item, dict) else None)
                title = getattr(item, "title", None) or (item.get("title") if isinstance(item, dict) else "")
                if url:
                    extracted.append({"title": str(title or ""), "url": str(url)})

    # Deduplicate by URL
    deduped = []
    seen = set()
    for s in extracted:
        u = s.get("url")
        if u and u not in seen:
            seen.add(u)
            deduped.append(s)
    return deduped


async def _execute_call(client: Any, model: str, call_input: Any, tools_param: Any, response_format: Any) -> Any:
    kwargs: dict[str, Any] = {"model": model, "input": call_input}
    if tools_param is not None:
        kwargs["tools"] = tools_param
    if response_format is not None:
        kwargs["response_format"] = response_format

    if hasattr(client, "aio") and hasattr(client.aio, "interactions") and hasattr(client.aio.interactions, "create"):
        return await client.aio.interactions.create(**kwargs)
    elif hasattr(client, "interactions") and hasattr(client.interactions, "create"):
        return await asyncio.to_thread(client.interactions.create, **kwargs)
    raise AttributeError("Client does not have aio.interactions.create or interactions.create")


async def ask(
    *,
    model: str,
    input: Union[str, list[dict]],
    schema: Optional[Any] = None,
    tools: Sequence[str] = (),
    agent: str,
    on_step: Optional[Callable[[dict], Any]] = None,
) -> AskResult:
    start_time = time.perf_counter()
    client = get_client()

    tools_param = [{"type": t} for t in tools] if tools else None
    response_format = None
    if schema is not None:
        response_format = {
            "type": "text",
            "mime_type": "application/json",
            "schema": schema.model_json_schema(),
        }

    current_model = model
    model_used = current_model

    async def _call_with_retries(target_model: str, current_input: Any) -> tuple[Any, str]:
        active_model = target_model
        delays = _BACKOFF_DELAYS

        for attempt in range(len(delays)):
            try:
                interaction = await _execute_call(client, active_model, current_input, tools_param, response_format)
                return interaction, active_model
            except Exception as err:
                if not _is_retryable_error(err):
                    raise err

                if attempt < len(delays) - 1:
                    await asyncio.sleep(delays[attempt])
                else:
                    # Final attempt failed
                    if active_model == config.MODEL_ARGON and _is_503(err):
                        # Fall back once to gemini-3.8-flash
                        active_model = config.MODEL_FLASH
                        interaction = await _execute_call(client, active_model, current_input, tools_param, response_format)
                        return interaction, active_model
                    raise err
        raise RuntimeError("Retries exhausted without result")

    # First attempt (with retries and model fallback)
    interaction, model_used = await _call_with_retries(current_model, input)
    output_text = getattr(interaction, "output_text", "") or ""

    # Schema validation and repair if needed
    data = output_text
    if schema is not None:
        try:
            data = schema.model_validate_json(output_text)
        except Exception as val_err:
            repair_prompt = f"\n\nValidation error: {val_err}\nPlease return valid JSON matching the schema."
            if isinstance(input, str):
                repaired_input = input + repair_prompt
            elif isinstance(input, list):
                repaired_input = list(input) + [{"type": "text", "text": repair_prompt}]
            else:
                repaired_input = str(input) + repair_prompt

            interaction, model_used = await _call_with_retries(model_used, repaired_input)
            output_text = getattr(interaction, "output_text", "") or ""
            data = schema.model_validate_json(output_text)

    # Process steps
    raw_steps = getattr(interaction, "steps", []) or []
    simplified_steps = []
    for s in raw_steps:
        s_dict = _simplify_step(s)
        simplified_steps.append(s_dict)
        if on_step is not None:
            cb_res = on_step(s_dict)
            if inspect.isawaitable(cb_res):
                await cb_res

    sources = _extract_sources(raw_steps)
    duration_ms = int((time.perf_counter() - start_time) * 1000)

    # Logging
    logger.info(
        "agent=%s model=%s tools=%s duration_ms=%d model_used=%s prompt=%s output=%s",
        agent,
        model,
        list(tools) if tools else [],
        duration_ms,
        model_used,
        _truncate_and_strip(input),
        _truncate_and_strip(output_text),
    )

    return AskResult(
        data=data,
        text=output_text,
        steps=simplified_steps,
        model_used=model_used,
        duration_ms=duration_ms,
        sources=sources,
    )
