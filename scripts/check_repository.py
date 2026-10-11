#!/usr/bin/env python3
"""Check publishable files without displaying credential values."""

from __future__ import annotations

import ipaddress
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD_WORKSPACES = {"CPO_generize", "CPO_mapping", "CPO_construction", "marker_literuture", "mission3", "legacy"}
BLOCKED_DIRS = {
    "data", "database", "logs", "runs", "archives", ".trash", "work", "results",
    "outputs", "output", "expanded", "name_mapping", "__pycache__", ".venv",
    ".aws", ".codex", ".claude", "node_modules",
}
BLOCKED_SUFFIXES = {
    ".sqlite", ".sqlite3", ".db", ".pt", ".pth", ".npz", ".npy", ".h5",
    ".hdf5", ".parquet", ".pkl", ".pdf", ".xlsx", ".xls", ".docx", ".pptx",
    ".zip", ".gz", ".bz2", ".tar", ".pem", ".key",
}
PATTERNS = {
    "access token": re.compile(r"(?:sk-|ghp_|github_pat_|AIza)[A-Za-z0-9_-]{16,}"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "authenticated URL": re.compile(r"https?://[^\s/]+:[^\s/@]+@"),
    "original machine path": re.compile(r"(?<!to)/(?:Users|home)/[^/\s]+/|/Volumes/[^/\s]+/"),
}
ASSIGNMENT = re.compile(
    r"(?i)[\"']?(?:[A-Z0-9_]*(?:API_KEY|APIKEY|PASSWORD|SECRET|ACCESS_TOKEN))"
    r"[\"']?\s*[:=]\s*[\"']([^\"'\n]+)[\"']"
)
PLACEHOLDERS = {"your-api-key", "test-key", "fake-key", "dummy-key", "test", "secret"}
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def candidate_files() -> list[Path]:
    raw = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
    )
    return sorted({ROOT / name.decode() for name in raw.split(b"\0") if name})


def main() -> int:
    failures = []
    counts = Counter()
    for path in candidate_files():
        relative = path.relative_to(ROOT)
        if not path.exists():
            continue
        if relative.parts[0] in OLD_WORKSPACES:
            failures.append(f"{relative}: task code is still in an old mixed workspace")
        if path.is_symlink():
            failures.append(f"{relative}: symlink outside a self-contained snapshot")
            continue
        if BLOCKED_DIRS.intersection(relative.parts[:-1]) or path.suffix in BLOCKED_SUFFIXES:
            failures.append(f"{relative}: data, generated artifact or private configuration")
        if path.name.startswith('.env') and path.name != '.env.example':
            failures.append(f"{relative}: non-example environment file")
        if path.stat().st_size > 1_000_000:
            failures.append(f"{relative}: exceeds the code snapshot size limit")
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            failures.append(f"{relative}: unexpected binary file")
            continue
        counts['files'] += 1
        if path.suffix == '.py':
            try:
                compile(content, str(relative), 'exec')
                counts['python'] += 1
            except SyntaxError as exc:
                failures.append(f"{relative}:{exc.lineno}: Python syntax error")
        if path.suffix == '.json':
            try:
                json.loads(content)
                counts['json'] += 1
            except json.JSONDecodeError as exc:
                failures.append(f"{relative}:{exc.lineno}: invalid JSON")
        # This scanner contains token and path signatures, never real credentials.
        if path == Path(__file__).resolve():
            continue
        for number, line in enumerate(content.splitlines(), 1):
            for label, pattern in PATTERNS.items():
                if pattern.search(line):
                    failures.append(f"{relative}:{number}: possible {label}")
            for match in ASSIGNMENT.finditer(line):
                if len(match[1]) >= 8 and match[1].lower() not in PLACEHOLDERS:
                    failures.append(f"{relative}:{number}: possible literal credential")
            for match in IPV4.finditer(line):
                try:
                    address = ipaddress.ip_address(match[0])
                except ValueError:
                    continue
                if not address.is_loopback:
                    failures.append(f"{relative}:{number}: non-loopback IP address")
    for failure in failures:
        print(failure)
    print(f"Checked {counts['files']} files, {counts['python']} Python sources, "
          f"{counts['json']} JSON files; {len(failures)} issue(s).")
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
