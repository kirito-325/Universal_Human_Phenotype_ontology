from __future__ import annotations

import json
import http.client
import hashlib
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple


DEFAULT_USER_AGENT = (
    "marker-literature-pipeline/0.1 "
    "(contact: set --email for NCBI and publisher courtesy)"
)


class HttpClient:
    def __init__(
        self,
        email: str = "",
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: int = 45,
        sleep: float = 0.34,
        retries: int = 2,
    ) -> None:
        self.email = email
        self.user_agent = user_agent
        self.timeout = timeout
        self.sleep = sleep
        self.retries = retries

    def _headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        if self.email:
            headers["From"] = self.email
        if extra:
            headers.update(extra)
        return headers

    def request_bytes(
        self,
        url: str,
        data: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        method: Optional[str] = None,
    ) -> Tuple[bytes, str, str]:
        encoded = None
        if data is not None:
            encoded = urllib.parse.urlencode(data).encode("utf-8")
        request = urllib.request.Request(
            url, data=encoded, headers=self._headers(headers), method=method
        )
        last_error = None
        for attempt in range(self.retries + 1):
            if self.sleep:
                time.sleep(self.sleep)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = response.read()
                    final_url = response.geturl()
                    content_type = response.headers.get("Content-Type", "")
                    return body, final_url, content_type
            except (
                urllib.error.URLError,
                TimeoutError,
                http.client.RemoteDisconnected,
                http.client.IncompleteRead,
                ConnectionResetError,
            ) as exc:
                last_error = exc
                if attempt >= self.retries:
                    raise
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"request failed: {url}: {last_error}")

    def head(self, url: str, headers: Optional[Dict[str, str]] = None) -> Tuple[int, str, str, int]:
        request = urllib.request.Request(
            url, headers=self._headers(headers), method="HEAD"
        )
        last_error = None
        for attempt in range(self.retries + 1):
            if self.sleep:
                time.sleep(self.sleep)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    content_length = response.headers.get("Content-Length", "0")
                    return (
                        response.status,
                        response.geturl(),
                        response.headers.get("Content-Type", ""),
                        int(content_length) if content_length.isdigit() else 0,
                    )
            except urllib.error.HTTPError as exc:
                content_length = exc.headers.get("Content-Length", "0")
                return (
                    exc.code,
                    exc.geturl(),
                    exc.headers.get("Content-Type", ""),
                    int(content_length) if content_length.isdigit() else 0,
                )
            except (
                urllib.error.URLError,
                TimeoutError,
                http.client.RemoteDisconnected,
                http.client.IncompleteRead,
                ConnectionResetError,
            ) as exc:
                last_error = exc
                if attempt >= self.retries:
                    raise
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"HEAD failed: {url}: {last_error}")

    def request_text(
        self,
        url: str,
        data: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, str, str]:
        body, final_url, content_type = self.request_bytes(url, data=data, headers=headers)
        charset = "utf-8"
        if "charset=" in content_type:
            charset = content_type.split("charset=", 1)[1].split(";", 1)[0].strip()
        return body.decode(charset or "utf-8", errors="replace"), final_url, content_type

    def request_json(
        self,
        url: str,
        data: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        text, final_url, content_type = self.request_text(url, data=data, headers=headers)
        return json.loads(text), final_url, content_type

    def download(
        self,
        url: str,
        path: Path,
        headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, str]:
        path.parent.mkdir(parents=True, exist_ok=True)
        body, final_url, content_type = self.request_bytes(url, headers=headers)
        if self._looks_like_pmc_pow(body, content_type):
            body, final_url, content_type = self._download_with_pmc_pow(
                url, body, content_type
            )
        path.write_bytes(body)
        return final_url, content_type

    def _looks_like_pmc_pow(self, body: bytes, content_type: str) -> bool:
        if "text/html" not in content_type.lower():
            return False
        head = body[:5000].decode("utf-8", errors="ignore")
        return "POW_CHALLENGE" in head and "cloudpmc-viewer-pow" in head

    def _download_with_pmc_pow(
        self, url: str, body: bytes, content_type: str
    ) -> Tuple[bytes, str, str]:
        text = body.decode("utf-8", errors="ignore")
        challenge_match = re.search(r'POW_CHALLENGE\s*=\s*"([^"]+)"', text)
        difficulty_match = re.search(r'POW_DIFFICULTY\s*=\s*"([^"]+)"', text)
        cookie_match = re.search(r'POW_COOKIE_NAME\s*=\s*"([^"]+)"', text)
        if not challenge_match or not cookie_match:
            return body, url, content_type
        challenge = challenge_match.group(1)
        difficulty = int(difficulty_match.group(1)) if difficulty_match else 4
        cookie_name = cookie_match.group(1)
        nonce = self._solve_pow(challenge, difficulty)
        cookie = f"{cookie_name}={challenge},{nonce}"
        return self.request_bytes(url, headers={"Cookie": cookie})

    def _solve_pow(self, challenge: str, difficulty: int) -> int:
        prefix = "0" * difficulty
        nonce = 0
        while True:
            digest = hashlib.sha256(f"{challenge}{nonce}".encode("utf-8")).hexdigest()
            if digest.startswith(prefix):
                return nonce
            nonce += 1
