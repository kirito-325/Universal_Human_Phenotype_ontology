from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zlib
from datetime import datetime, timezone
from pathlib import Path


SOURCE_URL = (
    "https://storage.googleapis.com/google-code-archive-source/v2/"
    "code.google.com/phenodigm/source-archive.zip"
)
ARCHIVE_ENTRY = "phenodigm/trunk/files/mappings/hp_mp_mappings.bz2"

# Values are taken from the Google Code archive central directory. The byte range is
# the raw DEFLATE payload for ARCHIVE_ENTRY, excluding its 107-byte local header.
RANGE_START = 18_276
COMPRESSED_SIZE = 701_043_113
RANGE_END = RANGE_START + COMPRESSED_SIZE - 1
EXPECTED_SIZE = 701_426_412
EXPECTED_CRC32 = 0x02E9598A
CHUNK_SIZE = 4 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    checksum = 0
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            size += len(chunk)
            checksum = zlib.crc32(chunk, checksum)
            digest.update(chunk)
    checksum &= 0xFFFFFFFF
    if size != EXPECTED_SIZE:
        raise ValueError(f"Unexpected matrix size: {size} != {EXPECTED_SIZE}")
    if checksum != EXPECTED_CRC32:
        raise ValueError(
            f"Unexpected matrix CRC32: {checksum:08x} != {EXPECTED_CRC32:08x}"
        )
    return {"bytes": size, "crc32": f"{checksum:08x}", "sha256": digest.hexdigest()}


def fetch(output: Path, timeout: int) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.parent / "source_manifest.json"
    if output.is_file():
        validation = validate(output)
        result = {
            "status": "existing_valid_file",
            "source_url": SOURCE_URL,
            "archive_entry": ARCHIVE_ENTRY,
            "source_archive_range": f"bytes={RANGE_START}-{RANGE_END}",
            "file": {"path": str(output.resolve()), **validation},
        }
        if not manifest_path.is_file():
            manifest_path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        return result

    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    request = urllib.request.Request(
        SOURCE_URL,
        headers={
            "Range": f"bytes={RANGE_START}-{RANGE_END}",
            "User-Agent": "CellPhenoRank/0.1 PhenoDigm legacy matrix retrieval",
        },
    )
    decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    input_bytes = 0
    output_bytes = 0
    checksum = 0
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", None)
            if status != 206:
                raise RuntimeError(f"Server did not honor Range request: HTTP {status}")
            content_range = response.headers.get("Content-Range", "")
            expected_range = f"bytes {RANGE_START}-{RANGE_END}/"
            if not content_range.startswith(expected_range):
                raise RuntimeError(f"Unexpected Content-Range: {content_range}")
            with partial.open("wb") as handle:
                while True:
                    compressed = response.read(CHUNK_SIZE)
                    if not compressed:
                        break
                    input_bytes += len(compressed)
                    data = decompressor.decompress(compressed)
                    if data:
                        handle.write(data)
                        output_bytes += len(data)
                        checksum = zlib.crc32(data, checksum)
                        digest.update(data)
                    if input_bytes // (64 * 1024 * 1024) != (
                        input_bytes - len(compressed)
                    ) // (64 * 1024 * 1024):
                        print(
                            f"downloaded {input_bytes / 1024**2:.0f} MiB; "
                            f"inflated {output_bytes / 1024**2:.0f} MiB",
                            flush=True,
                        )
                tail = decompressor.flush()
                if tail:
                    handle.write(tail)
                    output_bytes += len(tail)
                    checksum = zlib.crc32(tail, checksum)
                    digest.update(tail)
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    checksum &= 0xFFFFFFFF
    if input_bytes != COMPRESSED_SIZE:
        partial.unlink(missing_ok=True)
        raise ValueError(
            f"Incomplete range download: {input_bytes} != {COMPRESSED_SIZE}"
        )
    if output_bytes != EXPECTED_SIZE or checksum != EXPECTED_CRC32:
        partial.unlink(missing_ok=True)
        raise ValueError(
            f"Inflated entry failed validation: bytes={output_bytes}, crc32={checksum:08x}"
        )
    output_digest = digest.hexdigest()
    partial.replace(output)
    result = {
        "status": "downloaded_and_validated",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source_url": SOURCE_URL,
        "archive_entry": ARCHIVE_ENTRY,
        "source_archive_range": f"bytes={RANGE_START}-{RANGE_END}",
        "source_archive_compressed_bytes": COMPRESSED_SIZE,
        "file": {
            "path": str(output.resolve()),
            "bytes": output_bytes,
            "crc32": f"{checksum:08x}",
            "sha256": output_digest,
        },
        "provenance": {
            "repository": "https://code.google.com/archive/p/phenodigm/",
            "archive_last_modified": "2016-02-18T00:32:50Z",
            "method_paper": "https://doi.org/10.1093/database/bat025",
            "matrix_columns": [
                "hp_id",
                "hp_name",
                "mp_id",
                "mp_name",
                "simJ",
                "IC",
            ],
            "pair_score": "sqrt(simJ * IC)",
        },
    }
    manifest_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch the archived original PhenoDigm HP-MP OWLSim matrix"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/phenodigm_legacy/hp_mp_mappings.bz2"),
    )
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    result = fetch(args.output, args.timeout)
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
