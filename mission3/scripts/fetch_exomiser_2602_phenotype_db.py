from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
import zlib
from datetime import datetime, timezone
from pathlib import Path


SOURCE_URL = "https://g-879a9f.f5dc97.75bc.dn.glob.us/data/2602_phenotype.zip"
ARCHIVE_ENTRY = "2602_phenotype/2602_phenotype.mv.db"
ARCHIVE_BYTES = 13_478_530_504
PAYLOAD_START = 186
COMPRESSED_SIZE = 11_918_718_855
PAYLOAD_END = PAYLOAD_START + COMPRESSED_SIZE - 1
EXPECTED_SIZE = 22_215_954_432
EXPECTED_CRC32 = 0x24F6767A
CHUNK_SIZE = 8 * 1024 * 1024


def _file_record(path: Path, checksum: int, digest: str) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "crc32": f"{checksum & 0xFFFFFFFF:08x}",
        "sha256": digest,
    }


def _download_range(compressed: Path, timeout: int, retries: int) -> None:
    compressed.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    while (
        compressed.stat().st_size if compressed.exists() else 0
    ) < COMPRESSED_SIZE:
        existing = compressed.stat().st_size if compressed.exists() else 0
        if existing > COMPRESSED_SIZE:
            raise ValueError(f"Compressed partial is too large: {existing}")
        start = PAYLOAD_START + existing
        request = urllib.request.Request(
            SOURCE_URL,
            headers={
                "Range": f"bytes={start}-{PAYLOAD_END}",
                "User-Agent": "CellPhenoRank/0.1 Exomiser-2602 retrieval",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", None)
                content_range = response.headers.get("Content-Range", "")
                if status != 206 or not content_range.startswith(
                    f"bytes {start}-{PAYLOAD_END}/"
                ):
                    raise RuntimeError(
                        f"Server did not honor requested range: HTTP {status}, "
                        f"Content-Range={content_range}"
                    )
                with compressed.open("ab") as handle:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        before = existing
                        handle.write(chunk)
                        existing += len(chunk)
                        if existing // (512 * 1024 * 1024) != before // (
                            512 * 1024 * 1024
                        ):
                            print(
                                f"downloaded {existing / 1024**3:.2f} / "
                                f"{COMPRESSED_SIZE / 1024**3:.2f} GiB",
                                flush=True,
                            )
            failures = 0
        except Exception as error:
            failures += 1
            if failures >= retries:
                raise RuntimeError(
                    f"Range download failed after {failures} attempts; rerun to resume: {error}"
                ) from error
            print(f"download interrupted ({error}); retrying from saved bytes", flush=True)
            time.sleep(min(10, failures * 2))
    actual = compressed.stat().st_size
    if actual != COMPRESSED_SIZE:
        raise ValueError(f"Incomplete compressed payload: {actual} != {COMPRESSED_SIZE}")


def _inflate(compressed: Path, output: Path) -> tuple[int, str]:
    partial = output.with_suffix(output.suffix + ".partial")
    partial.unlink(missing_ok=True)
    decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    checksum = 0
    digest = hashlib.sha256()
    output_bytes = 0
    try:
        with compressed.open("rb") as source, partial.open("wb") as target:
            while True:
                chunk = source.read(CHUNK_SIZE)
                if not chunk:
                    break
                data = decompressor.decompress(chunk)
                if data:
                    before = output_bytes
                    target.write(data)
                    output_bytes += len(data)
                    checksum = zlib.crc32(data, checksum)
                    digest.update(data)
                    if output_bytes // (2 * 1024**3) != before // (2 * 1024**3):
                        print(
                            f"inflated {output_bytes / 1024**3:.2f} / "
                            f"{EXPECTED_SIZE / 1024**3:.2f} GiB",
                            flush=True,
                        )
            tail = decompressor.flush()
            if tail:
                target.write(tail)
                output_bytes += len(tail)
                checksum = zlib.crc32(tail, checksum)
                digest.update(tail)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    checksum &= 0xFFFFFFFF
    if not decompressor.eof:
        partial.unlink(missing_ok=True)
        raise ValueError("Raw DEFLATE stream ended before its final block")
    if output_bytes != EXPECTED_SIZE or checksum != EXPECTED_CRC32:
        partial.unlink(missing_ok=True)
        raise ValueError(
            f"Database validation failed: bytes={output_bytes}, crc32={checksum:08x}"
        )
    partial.replace(output)
    return checksum, digest.hexdigest()


def fetch(
    output: Path,
    timeout: int,
    retries: int,
    keep_compressed: bool,
) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.parent / "source_manifest.json"
    if output.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record = manifest.get("file", {})
        if output.stat().st_size == EXPECTED_SIZE and int(record.get("bytes", 0)) == EXPECTED_SIZE:
            return {**manifest, "status": "existing_size_and_manifest_valid"}
    compressed = output.parent / f"{output.name}.deflate.partial"
    _download_range(compressed, timeout, retries)
    checksum, digest = _inflate(compressed, output)
    if not keep_compressed:
        compressed.unlink()
    result = {
        "status": "downloaded_inflated_and_validated",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source_url": SOURCE_URL,
        "archive_bytes": ARCHIVE_BYTES,
        "archive_entry": ARCHIVE_ENTRY,
        "source_archive_range": f"bytes={PAYLOAD_START}-{PAYLOAD_END}",
        "source_archive_compressed_bytes": COMPRESSED_SIZE,
        "file": _file_record(output, checksum, digest),
        "release": {
            "exomiser_phenotype": "2602",
            "hpo": "2026-01-08",
            "hpo_annotations": "2026-01-08",
            "mp": "2026-01-27",
            "impc": "23.0",
            "h2": "2.2.224",
        },
        "provenance": {
            "release_notes": "https://github.com/exomiser/Exomiser/discussions/640",
            "archive_last_modified": "2026-02-12T17:45:25Z",
        },
    }
    manifest_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch only the H2 database entry from Exomiser 2602 phenotype data"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/exomiser_2602/2602_phenotype.mv.db"),
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--keep-compressed", action="store_true")
    args = parser.parse_args()
    result = fetch(args.output, args.timeout, args.retries, args.keep_compressed)
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
