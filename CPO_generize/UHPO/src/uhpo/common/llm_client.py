from __future__ import annotations

import http.client
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple


DEFAULT_ENDPOINT = "https://chat.ecnu.edu.cn/open/api/v1/chat/completions"
DEFAULT_MODEL = "ecnu-max"


def dotenv_values(path: str | Path) -> dict[str, str]:
    source = Path(path)
    if not source.exists():
        return {}
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"invalid dotenv line {line_number} in {source}")
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"invalid dotenv variable on line {line_number} in {source}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[name] = value
    return values


def api_keys_from_environment(
    env_file: str | Path = ".env",
    base_variable: str = "ECNU_API_KEY",
) -> list[tuple[str, str]]:
    file_values = dotenv_values(env_file)
    names = [base_variable]
    names.extend(f"{base_variable}_SHARD_{index}" for index in range(1, 33))
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name in names:
        value = os.environ.get(name, "").strip() or file_values.get(name, "").strip()
        if value and value not in seen:
            seen.add(value)
            result.append((name, value))
    return result


class ECNUClient:
    def __init__(
        self,
        api_key: str,
        endpoint: str = DEFAULT_ENDPOINT,
        model: str = DEFAULT_MODEL,
        timeout: int = 600,
        retries: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        thinking: bool = False,
        insecure_tls: bool = False,
    ) -> None:
        if not api_key:
            raise ValueError("ECNU API key is required")
        self.api_key = api_key
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.insecure_tls = insecure_tls

    def complete(
        self, messages: Sequence[Dict[str, str]]
    ) -> Tuple[str, Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.thinking:
            payload["thinking"] = {"type": "enabled"}
        request_data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                self.endpoint,
                data=request_data,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "cpo-go-cl-adjudicator/1.0",
                },
            )
            try:
                context = (
                    ssl._create_unverified_context() if self.insecure_tls else None
                )
                with urllib.request.urlopen(
                    request, timeout=self.timeout, context=context
                ) as response:
                    response_data = json.loads(response.read().decode("utf-8"))
                choices = response_data.get("choices") or []
                if not choices:
                    raise RuntimeError("ECNU response contains no choices")
                content = choices[0].get("message", {}).get("content", "")
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("ECNU response contains no message content")
                metadata = {
                    "id": response_data.get("id", ""),
                    "model": response_data.get("model", self.model),
                    "usage": response_data.get("usage", {}),
                    "finish_reason": choices[0].get("finish_reason", ""),
                }
                return content, metadata
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")[:2000]
                last_error = RuntimeError(f"ECNU HTTP {exc.code}: {body}")
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    raise last_error
                retry_after = exc.headers.get("Retry-After", "")
                delay = float(retry_after) if retry_after.isdigit() else min(30, 2**attempt)
            except (
                urllib.error.URLError,
                http.client.RemoteDisconnected,
                ConnectionResetError,
                ConnectionAbortedError,
                BrokenPipeError,
                TimeoutError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc
                delay = min(30, 2**attempt)
            if attempt < self.retries:
                time.sleep(delay)
        raise RuntimeError(f"ECNU request failed after retries: {last_error}")
