from __future__ import annotations

import http.client
import json
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple


DEFAULT_ENDPOINT = "https://chat.ecnu.edu.cn/open/api/v1/chat/completions"
DEFAULT_MODEL = "ecnu-max"
QUOTA_LIMIT_MARKERS = (
    "credits 限制",
    "credit limit",
    "credits limit",
    "quota exceeded",
    "resource_exhausted",
    "额度已用完",
    "超出额度",
)
_IPV4_CONFIGURATION_LOCK = threading.Lock()
_ORIGINAL_GETADDRINFO = socket.getaddrinfo
_IPV4_ONLY_ENABLED = False


def configure_ipv4_only_from_environment(
    variable: str = "ECNU_FORCE_IPV4",
) -> bool:
    """Force AF_UNSPEC ECNU lookups to IPv4 when explicitly requested."""

    value = os.environ.get(variable, "").strip().casefold()
    if value not in {"1", "true", "yes", "on"}:
        return False
    global _IPV4_ONLY_ENABLED
    with _IPV4_CONFIGURATION_LOCK:
        if _IPV4_ONLY_ENABLED:
            return False

        def ipv4_getaddrinfo(
            host: str | bytes | None,
            port: str | int | None,
            family: int = 0,
            type: int = 0,
            proto: int = 0,
            flags: int = 0,
        ) -> list[tuple[Any, ...]]:
            effective_family = socket.AF_INET if family in {0, socket.AF_UNSPEC} else family
            return _ORIGINAL_GETADDRINFO(
                host, port, effective_family, type, proto, flags
            )

        socket.getaddrinfo = ipv4_getaddrinfo
        _IPV4_ONLY_ENABLED = True
        return True


class EcnuQuotaExceededError(RuntimeError):
    """Raised when one API key has exhausted its fixed-window credits."""


class EcnuAuthenticationError(RuntimeError):
    """Raised when an ECNU API key is rejected by the gateway."""


def is_quota_limit_response(status_code: int, body: str) -> bool:
    normalized = body.casefold()
    return status_code == 429 and any(
        marker.casefold() in normalized for marker in QUOTA_LIMIT_MARKERS
    )


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
    disabled_variable = f"{base_variable}_DISABLED_VARIABLES"
    disabled_value = (
        os.environ.get(disabled_variable, "").strip()
        or file_values.get(disabled_variable, "").strip()
    )
    disabled = {
        value
        for value in re.split(r"[\s,;]+", disabled_value)
        if value
    }
    names = [base_variable]
    names.extend(f"{base_variable}_SHARD_{index}" for index in range(1, 33))
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name in names:
        if name in disabled:
            continue
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
        response_schema: Mapping[str, Any] | None = None,
        response_schema_name: str = "structured_response",
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
        self.response_schema = response_schema
        self.response_schema_name = response_schema_name

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
        if self.response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": self.response_schema_name,
                    "schema": self.response_schema,
                },
            }
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
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
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
                message = f"ECNU HTTP {exc.code}: {body}"
                if exc.code == 401:
                    raise EcnuAuthenticationError(message)
                if is_quota_limit_response(exc.code, body):
                    raise EcnuQuotaExceededError(message)
                last_error = RuntimeError(message)
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
