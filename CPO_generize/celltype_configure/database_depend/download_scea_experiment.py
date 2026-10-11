from __future__ import annotations

import argparse
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen


BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/microarray/data/atlas/sc_experiments/"


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href", "")
        if href:
            self.links.append(href)


def fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "go-cl-constraint-pipeline/1.0"})
    with urlopen(request, timeout=120) as response:
        return response.read()


def download(url: str, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    digest = hashlib.sha256()
    request = Request(url, headers={"User-Agent": "go-cl-constraint-pipeline/1.0"})
    with urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            handle.write(block)
            digest.update(block)
    temporary.replace(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def run(args: argparse.Namespace) -> None:
    experiment_url = urljoin(args.base_url.rstrip("/") + "/", args.experiment_id + "/")
    parser = LinkParser()
    parser.feed(fetch_bytes(experiment_url).decode("utf-8", errors="replace"))
    filenames = {Path(link).name for link in parser.links}
    required_suffixes = [
        ".cell_metadata.tsv",
        ".marker_genes_inferred_cell_type_-_ontology_labels.tsv",
    ]
    if args.include_background:
        required_suffixes.append(".aggregated_filtered_counts.mtx_rows.gz")
    selected: list[str] = []
    for suffix in required_suffixes:
        matches = sorted(name for name in filenames if name.endswith(suffix))
        if not matches:
            available = sorted(name for name in filenames if "marker_genes" in name)
            raise ValueError(
                f"Experiment {args.experiment_id} has no file ending in {suffix}. "
                f"Available marker files: {available}"
            )
        selected.append(matches[0])
    output_dir = Path(args.output_dir) / args.experiment_id
    records = [
        download(urljoin(experiment_url, filename), output_dir / filename)
        for filename in selected
    ]
    manifest = {
        "experiment_id": args.experiment_id,
        "source": experiment_url,
        "files": records,
    }
    (output_dir / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the small SCEA files needed for GO marker enrichment."
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--include-background", action="store_true")
    parser.add_argument("--base-url", default=BASE_URL)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

