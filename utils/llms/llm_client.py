import csv
import io
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import openai
import requests
from typing_extensions import Literal

from . import OPEN_AI_PRICING, OPEN_AI_TOOL_PRICING

logger = logging.getLogger(__name__)


@dataclass
class LLMUsage:
    """Container for a single LLM call's token usage and cost."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    token_cost_usd: Optional[float] = None
    tool_calls: Dict[str, int] = None
    tool_cost_usd: Optional[float] = None
    cost_usd: Optional[float] = None  # token + tool

    def as_dict(self) -> Dict[str, Any]:
        """Convenient dict view for logging/serialization."""
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "token_cost_usd": self.token_cost_usd,
            "tool_calls": self.tool_calls or {},
            "tool_cost_usd": self.tool_cost_usd,
            "cost_usd": self.cost_usd,
        }


class LLMClient:
    def __init__(self,
                 api: str = "openai",
                 base: str = "https://api.openai.com/v1",
                 model: str = "gpt-5",
                 temperature: float = 0.0,
                 api_key: str = "",
                 pricing: Optional[Dict[str, Dict[str, float]]] = None,
                 tool_pricing: Optional[Dict[str, float]] = None,
                 usage_log_path: Optional[str | Path] = None,
                 batch_usage_log_path: Optional[str | Path] = None):
        """
        Initialise LLM Client for OpenAI and Ollama
        Parameters
        ----------
        api - either "openai" or "ollama"
        base - URL
        model - llm model
        temperature - llm temperature
        api_key - only applicable to openai
        pricing - optional dict of per-1M token costs by model prefix, e.g.
            {"gpt-4.1": {"input": 5.0, "output": 15.0}}
        tool_pricing - optional dict of per-call tool costs, e.g. {"web_search": 0.01}
        usage_log_path - optional CSV file path to append usage rows per call
        batch_usage_log_path - optional CSV file path to append usage rows per batch estimate
        """
        self.api = api
        self.base = base
        self.model = model
        self.temperature = temperature
        if self.api == "openai":
            self.api_key = api_key
        self.pricing = pricing or {}
        if self.pricing == {} and self.api == "openai":
            self.pricing = OPEN_AI_PRICING
        self.tool_pricing = tool_pricing or (OPEN_AI_TOOL_PRICING if self.api == "openai" else {})
        self.last_usage: Optional[LLMUsage] = None
        self.usage_history: List[LLMUsage] = []
        self.usage_log_path = Path(usage_log_path) if usage_log_path else None
        self.batch_usage_log_path = (
            Path(batch_usage_log_path)
            if batch_usage_log_path
            else self._derive_batch_usage_log_path()
        )

    def _derive_batch_usage_log_path(self) -> Optional[Path]:
        if not self.usage_log_path:
            return None
        suffix = self.usage_log_path.suffix or ".csv"
        return self.usage_log_path.with_name(f"{self.usage_log_path.stem}_batch{suffix}")

    def _ensure_openai_api_key(self) -> None:
        if self.api != "openai":
            return
        if self.api_key:
            openai.api_key = self.api_key

    def call(self, prompt: str, log_usage: bool = False, **kwargs_additional) -> str | None:
        """
        Call the LLM
        Parameters
        ----------
        prompt
        log_usage: when True, logs token usage (and cost if pricing is provided)
        kwargs_additional: mainly for openai. Example
        >>> LLMClient().call("Hello",
        >>>     tools=[{"type": "web_search", "search_context_size": "high"}],
        >>>     reasoning={"effort": "medium"},
        >>>     stream=True,
        >>>     tool_choice='auto',
        >>>     input=[
        >>>         {
        >>>             "role": "system",
        >>>             "content": "You are a helpful assistant."
        >>>         },
        >>>         {
        >>>             "role": "user",
        >>>             "content": prompt
        >>>         }
        >>>     ]
        >>> )

        Returns
        -------

        """
        if self.api == "openai":
            self._ensure_openai_api_key()
            # If the model starts with gpt-5, temperature is not supported
            kwargs: Dict[str, Any] = {
                "model": self.model,
            }
            if not kwargs_additional.get("input"):
                # Use a simple user input with prompt
                kwargs["input"] = prompt
            if not self.model.startswith("gpt-5"):
                kwargs["temperature"] = self.temperature
            # Add further arguments
            kwargs.update(kwargs_additional)

            stream = kwargs_additional.get("stream", False)
            # If not streaming, we can extract usage from the response
            if not stream:
                response = openai.responses.create(**kwargs)
                usage = self._extract_usage(response)
                self._record_usage(usage, log_usage)
                return response.output_text

            stream_obj = openai.responses.create(**kwargs)
            text_chunks: list[str] = []
            final_response = None
            for event in stream_obj:
                # Text deltas from the model
                if event.type == "response.output_text.delta":
                    text_chunks.append(event.delta)
                # Final event may contain the aggregated response with usage
                elif event.type == "response.completed":
                    final_response = getattr(event, "response", None)
                # Handle errors or completion markers
                elif event.type == "response.error":
                    logger.error(event.error)
            # Pull usage from the streaming wrapper when possible
            usage = self._extract_usage(getattr(stream_obj, "response", None) or final_response)
            self._record_usage(usage, log_usage)
            return "".join(text_chunks) or None
        elif self.api == "ollama":
            response = requests.post(
                f"{self.base}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
            )
            response.raise_for_status()
            return response.json()['message']['content']
        else:
            logger.error("Unsupported API: %s", self.api)
            return None

    def translate(self, text: str, source_lang: str, dest_lang: str = "English") -> str:
        """
        Translate the text from source language to destination language
        Parameters
        ----------
        text
        source_lang: source language
        dest_lang: default is English

        Returns
        -------

        """
        prompt = f"Translate the following text from {source_lang} to {dest_lang} (no commentary, only translation):\n\n{text}"
        out = self.call(prompt).strip()
        if out and out != text:
            return out
        return text

    # ----------------------- Usage helpers ----------------------- #
    def _extract_usage(self, response: Any) -> Optional[LLMUsage]:
        """
        Normalize the usage payload from OpenAI Responses API into an LLMUsage.
        """
        usage_raw = getattr(response, "usage", None)
        if usage_raw is None and isinstance(response, dict):
            usage_raw = response.get("usage")
        if not usage_raw:
            return None

        def _usage_get(usage_obj: Any, *keys: str) -> Any:
            if isinstance(usage_obj, dict):
                for key in keys:
                    if key in usage_obj:
                        return usage_obj.get(key)
                return None
            for key in keys:
                if hasattr(usage_obj, key):
                    return getattr(usage_obj, key)
            return None

        input_tokens = _usage_get(usage_raw, "input_tokens", "prompt_tokens") or 0
        output_tokens = _usage_get(usage_raw, "output_tokens", "completion_tokens") or 0
        total_tokens = _usage_get(usage_raw, "total_tokens") or (input_tokens + output_tokens)
        token_cost = self._estimate_cost(input_tokens=input_tokens, output_tokens=output_tokens, model=self.model)
        tool_calls = self._extract_tool_calls(response)
        tool_cost = self._estimate_tool_cost(tool_calls)
        total_cost = None
        if token_cost is not None or tool_cost is not None:
            total_cost = (token_cost or 0.0) + (tool_cost or 0.0)
        return LLMUsage(
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            token_cost_usd=token_cost,
            tool_calls=tool_calls,
            tool_cost_usd=tool_cost,
            cost_usd=total_cost,
        )

    def _estimate_cost(self, input_tokens: int, output_tokens: int, model: Optional[str] = None) -> Optional[float]:
        """
        Estimate USD cost based on provided pricing table.
        pricing structure: {model_prefix: {"input": <per-1m>, "output": <per-1m>}}
        """
        model = model or self.model
        pricing = self._match_pricing(model)
        if not pricing:
            return None
        input_price = pricing.get("input", 0.0)
        output_price = pricing.get("output", 0.0)
        cost = (input_tokens / 1000000) * input_price + (output_tokens / 1000000) * output_price
        return round(cost, 6)

    def _match_pricing(self, model: str) -> Optional[Dict[str, float]]:
        """
        Fetch pricing by exact model or prefix match from self.pricing.
        """
        if model in self.pricing:
            return self.pricing[model]
        for prefix, price in self.pricing.items():
            if model.startswith(prefix):
                return price
        return None

    @staticmethod
    def _extract_tool_calls(response: Any) -> Dict[str, int]:
        """
        Best-effort extraction of tool call counts from the response object.
        Returns a dict: {tool_name: count}
        """
        def _get_attr(obj: Any, key: str) -> Any:
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)

        counts: Dict[str, int] = {}
        # Direct attribute
        tool_calls = getattr(response, "tool_calls", None)
        if tool_calls is None and isinstance(response, dict):
            tool_calls = response.get("tool_calls")

        # If tool_calls is iterable, tally names
        if tool_calls:
            for call in tool_calls:
                name = _get_attr(call, "name")
                if name is None:
                    t = _get_attr(call, "type")
                    if t == "web_search_call":
                        name = "web_search"
                    elif isinstance(t, str) and t.endswith("_call"):
                        name = t[:-5]
                if not name:
                    continue
                counts[name] = counts.get(name, 0) + 1
            return counts

        # 2) Responses API: tool calls commonly appear in response.output as "<tool>_call"
        output = getattr(response, "output", None)
        if output is None and isinstance(response, dict):
            output = response.get("output")

        if output and isinstance(output, list):
            for item in output:
                t = _get_attr(item, "type")
                if t == "web_search_call":
                    counts["web_search"] = counts.get("web_search", 0) + 1
                    continue
                if isinstance(t, str) and t.endswith("_call"):
                    tool_name = t[:-5]  # strip "_call"
                    counts[tool_name] = counts.get(tool_name, 0) + 1
                    continue
                if t in {"tool_call", "function_call"}:
                    name = _get_attr(item, "name") or _get_attr(item, "tool_name")
                    if name:
                        counts[name] = counts.get(name, 0) + 1
                        continue
                content = _get_attr(item, "content")
                if content and isinstance(content, list):
                    for entry in content:
                        t2 = _get_attr(entry, "type")
                        if t2 == "web_search_call":
                            counts["web_search"] = counts.get("web_search", 0) + 1
                            continue
                        if isinstance(t2, str) and t2.endswith("_call"):
                            tool_name = t2[:-5]
                            counts[tool_name] = counts.get(tool_name, 0) + 1
        return counts

    def _estimate_tool_cost(self, tool_calls: Dict[str, int]) -> Optional[float]:
        """
        Estimate tool cost based on per-call pricing.
        """
        if not tool_calls:
            return 0.0
        total = 0.0
        for name, count in tool_calls.items():
            price = self.tool_pricing.get(name)
            if price is None:
                continue
            total += price * count
        return round(total, 6)

    @staticmethod
    def _log_usage(usage: LLMUsage) -> None:
        """
        Log usage in a consistent format.
        """
        logger.info(
            "LLM usage | model=%s input_tokens=%s output_tokens=%s total_tokens=%s token_cost=%s tool_calls=%s tool_cost=%s total_cost=%s",
            usage.model,
            usage.input_tokens,
            usage.output_tokens,
            usage.total_tokens,
            usage.token_cost_usd,
            usage.tool_calls or {},
            usage.tool_cost_usd,
            usage.cost_usd,
        )

    def _record_usage(self, usage: Optional[LLMUsage], log_usage: bool) -> None:
        """
        Store usage, optionally log, and append to CSV if configured.
        """
        if not usage:
            return
        self.last_usage = usage
        self.usage_history.append(usage)
        if log_usage:
            self._log_usage(usage)
        if self.usage_log_path:
            self._append_usage_csv(self.usage_log_path, usage)

    @staticmethod
    def _append_usage_csv(path: Path, usage: LLMUsage) -> None:
        """
        Append a single usage row to CSV. Creates file with header if missing.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open(mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "model",
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "token_cost_usd",
                    "tool_calls",
                    "tool_cost_usd",
                    "cost_usd",
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(usage.as_dict())

    def export_usage_csv(self, path: str | Path, usages: Optional[Sequence[LLMUsage]] = None) -> Path:
        """
        Write all recorded usages to CSV (overwrite). Useful after many calls.
        Parameters:
            path: path to the output CSV file
            usages: optional list of usages to write - default to the entire history
        Returns:
            path to the output CSV file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = usages if usages is not None else self.usage_history
        with path.open(mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "model",
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "token_cost_usd",
                    "tool_calls",
                    "tool_cost_usd",
                    "cost_usd",
                ],
            )
            writer.writeheader()
            for u in rows:
                writer.writerow(u.as_dict())
        return path

    def get_last_usage(self) -> Optional[LLMUsage]:
        """
        Retrieve the last recorded usage (if any).
        """
        return self.last_usage

    def get_usage_history(self) -> List[LLMUsage]:
        """
        Retrieve the in-memory list of all usage records for this client.
        """
        return list(self.usage_history)

    # ----------------------- Batch helpers ----------------------- #
    def create_batch(self, reqs: Sequence[Dict[str, Any]], completion_window: Literal["24h"] = "24h"):
        """
        Create a batch job for the /v1/responses endpoint.
        Each request item must have keys: custom_id (str), body (dict).
        """
        if self.api != "openai":
            raise ValueError("Batch API is only available for OpenAI backend.")
        self._ensure_openai_api_key()
        # Build JSONL payload expected by the Batch API
        lines = []
        for req in reqs:
            lines.append(json.dumps({
                "custom_id": req["custom_id"],
                "method": "POST",
                "url": "/v1/responses",
                "body": req["body"],
            }))
        payload = "\n".join(lines).encode("utf-8")
        input_file = openai.files.create(
            file=("batch_requests.jsonl", io.BytesIO(payload)),
            purpose="batch",
        )
        batch = openai.batches.create(
            input_file_id=input_file.id,
            endpoint="/v1/responses",
            completion_window=completion_window,
        )
        return batch

    def poll_batch(self, batch_id: str, *, interval: int = 30, timeout: int | None = None):
        """
        Poll a batch until it reaches a terminal state.
        If timeout is provided, raise TimeoutError after that many seconds.
        """
        if self.api != "openai":
            raise ValueError("Batch API is only available for OpenAI backend.")
        self._ensure_openai_api_key()
        start = time.time()
        while True:
            batch = openai.batches.retrieve(batch_id)
            if batch.status == "completed":
                return batch
            if batch.status in {"failed", "expired", "cancelled"}:
                logger.error("Batch %s ended with status=%s error=%s", batch_id, batch.status, getattr(batch, "error", None))
                return batch
            if timeout is not None and time.time() - start > timeout:
                raise TimeoutError(f"Batch {batch_id} did not finish in {timeout} seconds")
            time.sleep(interval)

    def fetch_batch_output(self, batch) -> List[Dict[str, Any]]:
        """
        Download and parse the JSONL output of a completed batch.
        Returns a list of response objects keyed by custom_id.
        """
        if self.api != "openai":
            raise ValueError("Batch API is only available for OpenAI backend.")
        self._ensure_openai_api_key()
        if not getattr(batch, "output_file_id", None):
            return []
        raw = openai.files.content(batch.output_file_id).read().decode("utf-8")
        return [json.loads(line) for line in raw.splitlines() if line.strip()]

    def estimate_batch_cost(
            self,
            batch_results: list[dict[str, Any]],
            *,
            model: str | None = None,
            discount: float | None = None,  # e.g., 0.5 if you want to apply OpenAI's batch discount
    ) -> LLMUsage:
        """
        Aggregate token/tool usage from Batch API output lines and estimate cost.
        `batch_results` is the parsed JSONL list from fetch_batch_output.
        Examples:
            >>> batch_usage = self.estimate_batch_cost(batch_results, discount=0.5)  # set discount=None to skip
            >>> logger.info("Batch cost: %s USD (input=%s, output=%s)", batch_usage.cost_usd, batch_usage.input_tokens, batch_usage.output_tokens)
        """
        model = model or self.model
        total_input = total_output = 0
        tool_counts: dict[str, int] = {}

        for line in batch_results:
            custom_id = line.get("custom_id")
            resp = (line.get("response") or {})
            if resp.get("status_code") != 200:
                continue
            body = resp.get("body") or {}
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except json.JSONDecodeError:
                    body = {}
            usage = body.get("usage") or (body.get("response") or {}).get("usage") or {}
            if not usage:
                logger.debug("Batch response missing usage for %s", custom_id)
            total_input += usage.get("input_tokens") or usage.get("prompt_tokens") or 0
            total_output += usage.get("output_tokens") or usage.get("completion_tokens") or 0

            # Optional: tally tool calls if present (including output-based tool calls)
            tool_response = body.get("response") or body
            calls = self._extract_tool_calls(tool_response)
            for name, count in calls.items():
                tool_counts[name] = tool_counts.get(name, 0) + count

        token_cost = self._estimate_cost(total_input, total_output, model=model)
        if discount is not None and token_cost is not None:
            token_cost = round(token_cost * discount, 6)

        tool_cost = self._estimate_tool_cost(tool_counts)
        total_cost = None
        if token_cost is not None or tool_cost is not None:
            total_cost = (token_cost or 0.0) + (tool_cost or 0.0)

        batch_usage = LLMUsage(
            model=model,
            input_tokens=total_input,
            output_tokens=total_output,
            total_tokens=total_input + total_output,
            token_cost_usd=token_cost,
            tool_calls=tool_counts,
            tool_cost_usd=tool_cost,
            cost_usd=total_cost,
        )
        if self.batch_usage_log_path:
            self._append_usage_csv(self.batch_usage_log_path, batch_usage)
        return batch_usage
