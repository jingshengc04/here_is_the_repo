import asyncio
import base64
import inspect
import logging
import time
from dataclasses import dataclass, field
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
    citations: list[dict] = field(default_factory=list)


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
        queries_str = "; ".join(queries) if isinstance(queries, list) else str(queries)
        query = f"Searching: {queries_str}"
        s_dict["queries"] = queries if isinstance(queries, list) else [queries]
    elif getattr(args, "query", None):
        raw_q = getattr(args, "query")
        query = f"Searching: {raw_q}" if not str(raw_q).startswith("Searching: ") else str(raw_q)
    elif isinstance(args, dict) and args.get("query"):
        raw_q = args.get("query")
        query = f"Searching: {raw_q}" if not str(raw_q).startswith("Searching: ") else str(raw_q)

    if query is None:
        raw_q = getattr(step, "query", None) if not isinstance(step, dict) else step.get("query")
        if raw_q:
            query = f"Searching: {raw_q}" if not str(raw_q).startswith("Searching: ") else str(raw_q)

    if query is not None:
        s_dict["query"] = query

    return s_dict


def _extract_citations(steps: list[Any]) -> list[dict]:
    citations = []
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
                        ann_type = getattr(a, "type", None) or (a.get("type") if isinstance(a, dict) else None)
                        url = getattr(a, "url", None) or (a.get("url") if isinstance(a, dict) else None)
                        if url and (ann_type in ("url_citation", None) or hasattr(a, "url")):
                            title = getattr(a, "title", None) or (a.get("title") if isinstance(a, dict) else "")
                            start = getattr(a, "start_index", None)
                            if start is None and isinstance(a, dict):
                                start = a.get("start_index", a.get("start"))
                            elif start is None:
                                start = getattr(a, "start", None)

                            end = getattr(a, "end_index", None)
                            if end is None and isinstance(a, dict):
                                end = a.get("end_index", a.get("end"))
                            elif end is None:
                                end = getattr(a, "end", None)

                            citations.append({
                                "title": str(title or ""),
                                "url": str(url),
                                "start": start,
                                "end": end,
                            })
    return citations


def _extract_sources(steps: list[Any]) -> list[dict]:
    citations = _extract_citations(steps)
    deduped = []
    seen = set()
    for c in citations:
        u = c.get("url")
        if u and u not in seen:
            seen.add(u)
            deduped.append({"title": c.get("title", ""), "url": u})

    # Fallback to step.result or step.sources if annotations weren't present
    if not deduped:
        for step in steps:
            result = getattr(step, "result", None) or (step.get("result") if isinstance(step, dict) else None)
            if result and isinstance(result, list):
                for item in result:
                    url = getattr(item, "url", None) or (item.get("url") if isinstance(item, dict) else None)
                    title = getattr(item, "title", None) or (item.get("title") if isinstance(item, dict) else "")
                    if url and url not in seen:
                        seen.add(url)
                        deduped.append({"title": str(title or ""), "url": str(url)})

            sources = getattr(step, "sources", None) or (step.get("sources") if isinstance(step, dict) else None)
            if sources and isinstance(sources, list):
                for item in sources:
                    url = getattr(item, "url", None) or (item.get("url") if isinstance(item, dict) else None)
                    title = getattr(item, "title", None) or (item.get("title") if isinstance(item, dict) else "")
                    if url and url not in seen:
                        seen.add(url)
                        deduped.append({"title": str(title or ""), "url": str(url)})

    return deduped


async def _execute_call(client: Any, model: str, call_input: Any, tools_param: Any, response_format: Any, generation_config: Optional[dict] = None) -> Any:
    kwargs: dict[str, Any] = {"model": model, "input": call_input}
    if generation_config:
        kwargs["generation_config"] = generation_config
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
    generation_config: Optional[dict] = None,
) -> AskResult:
    start_time = time.perf_counter()
    client = get_client()

    tools_list = list(tools) if tools else []
    tools_param = [{"type": t} for t in tools_list] if tools_list else None
    has_google_search = any(
        t == "google_search" or (isinstance(t, dict) and t.get("type") == "google_search")
        for t in tools_list
    )

    current_model = model

    async def _call_with_retries(target_model: str, current_input: Any, current_tools: Any, current_resp_fmt: Any) -> tuple[Any, str]:
        active_model = target_model
        delays = _BACKOFF_DELAYS

        for attempt in range(len(delays)):
            try:
                interaction = await _execute_call(client, active_model, current_input, current_tools, current_resp_fmt, generation_config)
                return interaction, active_model
            except Exception as err:
                if not _is_retryable_error(err):
                    raise err

                if attempt < len(delays) - 1:
                    await asyncio.sleep(delays[attempt])
                else:
                    if active_model == config.MODEL_ARGON and _is_503(err):
                        # Fall back once to gemini-3.8-flash
                        active_model = config.MODEL_FLASH
                        interaction = await _execute_call(client, active_model, current_input, current_tools, current_resp_fmt, generation_config)
                        return interaction, active_model
                    raise err
        raise RuntimeError("Retries exhausted without result")

    is_search_with_schema = has_google_search and (schema is not None)

    if is_search_with_schema:
        # Step 1: Make grounded search call WITHOUT response_format to preserve url_citations
        path_taken = "two_step_search_and_schema_conversion"
        interaction, model_used = await _call_with_retries(
            current_model, input, current_tools=tools_param, current_resp_fmt=None
        )
        grounded_text = getattr(interaction, "output_text", "") or ""

        # Process steps & progress from the search call
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
        citations = _extract_citations(raw_steps)

        # Step 2: Cheap conversion call with gemini-3.8-flash, no tools, structured response_format
        schema_format = {
            "type": "text",
            "mime_type": "application/json",
            "schema": schema.model_json_schema(),
        }
        convert_prompt = (
            f"Original user request: {_truncate_and_strip(input, 1000)}\n\n"
            f"Grounded research text:\n{grounded_text}\n\n"
            f"Extract and return valid JSON matching the schema."
        )

        schema_interaction, _ = await _call_with_retries(
            config.MODEL_FLASH, convert_prompt, current_tools=None, current_resp_fmt=schema_format
        )
        output_text = getattr(schema_interaction, "output_text", "") or ""

        try:
            data = schema.model_validate_json(output_text)
        except Exception as val_err:
            repair_prompt = f"{convert_prompt}\n\nValidation error: {val_err}\nPlease return valid JSON matching the schema."
            schema_interaction, _ = await _call_with_retries(
                config.MODEL_FLASH, repair_prompt, current_tools=None, current_resp_fmt=schema_format
            )
            output_text = getattr(schema_interaction, "output_text", "") or ""
            data = schema.model_validate_json(output_text)

    else:
        path_taken = "single_step_direct"
        response_format = None
        if schema is not None:
            response_format = {
                "type": "text",
                "mime_type": "application/json",
                "schema": schema.model_json_schema(),
            }

        interaction, model_used = await _call_with_retries(
            current_model, input, current_tools=tools_param, current_resp_fmt=response_format
        )
        output_text = getattr(interaction, "output_text", "") or ""

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

                interaction, model_used = await _call_with_retries(
                    model_used, repaired_input, current_tools=tools_param, current_resp_fmt=response_format
                )
                output_text = getattr(interaction, "output_text", "") or ""
                data = schema.model_validate_json(output_text)

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
        citations = _extract_citations(raw_steps)

    duration_ms = int((time.perf_counter() - start_time) * 1000)

    # Logging at INFO with path documentation
    logger.info(
        "agent=%s model=%s tools=%s path=%s duration_ms=%d model_used=%s sources_count=%d prompt=%s output=%s",
        agent,
        model,
        tools_list,
        path_taken,
        duration_ms,
        model_used,
        len(sources),
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
        citations=citations,
    )
