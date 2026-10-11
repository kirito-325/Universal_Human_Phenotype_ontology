from __future__ import annotations

import csv
import re
import tempfile
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

from .utils import read_json, utc_now, write_json


STATISTICS_FIELDS = (
    "cell_cluster",
    "gene",
    "p_val",
    "avg_logFC",
    "pct.1",
    "pct.2",
    "p_val_adj",
)


def normalize_header(value: Any) -> str:
    text = str(value or "").strip().casefold()
    text = text.replace("%", " percent ")
    text = re.sub(r"[._/\\:()\[\]-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


RAW_ALIASES = {
    "cell_cluster": {
        "annotation",
        "cell annotation",
        "cell class",
        "cell cluster",
        "cell clusters",
        "cell population",
        "cell subtype",
        "cell type",
        "cell types",
        "celltype",
        "celltypes",
        "cluster",
        "cluster id",
        "cluster identity",
        "cluster name",
        "clusters",
        "ident",
        "identity",
        "population",
        "subcluster",
        "subtype",
    },
    "gene": {
        "feature",
        "features",
        "gene",
        "gene id",
        "gene name",
        "gene names",
        "gene symbol",
        "gene symbols",
        "geneid",
        "genename",
        "genes",
        "genesymbol",
        "marker gene",
        "marker genes",
        "symbol",
    },
    "p_val": {
        "nominal p value",
        "p val",
        "p value",
        "p values",
        "pval",
        "pvalue",
        "raw p val",
        "raw p value",
        "unadjusted p value",
    },
    "avg_logFC": {
        "average log fc",
        "average log fold change",
        "average log2 fc",
        "avg diff",
        "avg log fc",
        "avg log fold change",
        "avg log2 fc",
        "avg difference",
        "avg logfc",
        "avg log2fc",
        "log fc",
        "log fold change",
        "log2 fc",
        "log2 fold change",
        "logfc",
        "log2fc",
    },
    "pct.1": {
        "pct 1",
        "pct1",
        "percent 1",
        "percentage 1",
    },
    "pct.2": {
        "pct 2",
        "pct2",
        "percent 2",
        "percentage 2",
    },
    "p_val_adj": {
        "adj p val",
        "adj p value",
        "adjusted p val",
        "adjusted p value",
        "adjusted p values",
        "fdr",
        "p adj",
        "p adjusted",
        "p val adj",
        "p value adj",
        "padj",
        "pval adj",
        "pvalue adj",
        "q value",
        "qvalue",
    },
}

HEADER_ALIASES = {
    field: {normalize_header(value) for value in values}
    for field, values in RAW_ALIASES.items()
}
ALIAS_TO_FIELD = {
    alias: field
    for field, aliases in HEADER_ALIASES.items()
    for alias in aliases
}

NON_SIGNATURE_HEADERS = {
    "age",
    "antibody",
    "antigen",
    "barcode",
    "candidate gene",
    "cell",
    "cell id",
    "cell type",
    "celltype",
    "channel",
    "clone",
    "company",
    "condition",
    "description",
    "dilution",
    "donor",
    "feature",
    "fluorochrome",
    "forward primer",
    "forward sequence",
    "gender",
    "gene",
    "gene id",
    "gene name",
    "gene symbol",
    "gene target",
    "group",
    "guide",
    "id",
    "ligand",
    "marker",
    "oligo",
    "patient",
    "patient id",
    "peak channel",
    "primer",
    "probe",
    "rank",
    "reverse primer",
    "reverse sequence",
    "sample",
    "sample id",
    "score",
    "sequence",
    "sex",
    "sgrna",
    "sgrna sequence",
    "subject",
    "target",
    "term",
    "treatment",
}
NON_GENE_VALUES = {
    "false",
    "gene",
    "genes",
    "msigdb",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "reference",
    "references",
    "source",
    "true",
}
GENE_VALUE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*(?:[.-][A-Za-z0-9]+)*$")


def csv_rows(path: Path) -> Iterator[List[str]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(line.replace("\x00", "") for line in handle)
        for row in reader:
            yield [str(value).strip() for value in row]


def read_sample(path: Path, max_rows: int = 250) -> List[List[str]]:
    return list(islice(csv_rows(path), max_rows))


def statistics_header_mappings(row: Sequence[str]) -> List[Dict[str, int]]:
    occurrences: Dict[str, List[int]] = {field: [] for field in STATISTICS_FIELDS}
    for index, value in enumerate(row):
        field = ALIAS_TO_FIELD.get(normalize_header(value))
        if field:
            occurrences[field].append(index)
    if any(not occurrences[field] for field in STATISTICS_FIELDS):
        return []
    group_count = min(len(occurrences[field]) for field in STATISTICS_FIELDS)
    return [
        {field: occurrences[field][group_index] for field in STATISTICS_FIELDS}
        for group_index in range(group_count)
    ]


def find_statistics_header(
    rows: Sequence[Sequence[str]],
) -> Tuple[int, List[Dict[str, int]]]:
    for row_index, row in enumerate(rows[:80]):
        mappings = statistics_header_mappings(row)
        if mappings:
            return row_index, mappings
    return -1, []


def row_value(row: Sequence[str], index: int) -> str:
    return row[index].strip() if index < len(row) else ""


def looks_numeric(value: str) -> bool:
    text = value.strip().replace(",", "")
    text = re.sub(r"^[<>=~]+", "", text).strip()
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        float(text)
    except ValueError:
        return False
    return True


def statistics_tuple(
    row: Sequence[str], mapping: Dict[str, int]
) -> Tuple[str, ...] | None:
    values = tuple(row_value(row, mapping[field]) for field in STATISTICS_FIELDS)
    if not values[0] or not values[1]:
        return None
    numeric_count = sum(looks_numeric(value) for value in values[2:])
    if numeric_count < 4:
        return None
    return values


def extract_statistics_rows(path: Path) -> List[Tuple[str, ...]]:
    output: List[Tuple[str, ...]] = []
    active_mappings: List[Dict[str, int]] = []
    for row in csv_rows(path):
        mappings = statistics_header_mappings(row)
        if mappings:
            active_mappings = mappings
            continue
        for mapping in active_mappings:
            values = statistics_tuple(row, mapping)
            if values is not None:
                output.append(values)
    return output


def looks_like_gene(value: str) -> bool:
    text = value.strip()
    if not text or len(text) > 48 or text.casefold() in NON_GENE_VALUES:
        return False
    if not GENE_VALUE_RE.fullmatch(text):
        return False
    return any(character.isalpha() for character in text)


def gene_values(value: str) -> List[str]:
    text = value.strip()
    if looks_like_gene(text):
        return [text]
    if not any(delimiter in text for delimiter in (",", ";", "|")):
        return []
    parts = [part.strip() for part in re.split(r"[,;|]", text) if part.strip()]
    if len(parts) < 2 or not all(looks_like_gene(part) for part in parts):
        return []
    return parts


def is_non_signature_header(value: str) -> bool:
    normalized = normalize_header(value)
    if normalized in ALIAS_TO_FIELD or normalized in NON_SIGNATURE_HEADERS:
        return True
    return bool(
        re.fullmatch(
            r"(?:receptor|primer|guide|channel|clone|sequence|replicate) ?\d+",
            normalized,
        )
    )


def signature_columns_for_header(
    rows: Sequence[Sequence[str]], header_index: int
) -> Tuple[List[Tuple[int, str]], float, int]:
    header = rows[header_index]
    candidates: List[Tuple[int, str]] = []
    for column_index, value in enumerate(header):
        name = value.strip()
        normalized = normalize_header(name)
        if not name or normalized.startswith("unnamed"):
            continue
        if looks_numeric(name) or is_non_signature_header(normalized):
            continue
        candidates.append((column_index, name))
    if len(candidates) < 2:
        return [], 0.0, 0

    selected: List[Tuple[int, str]] = []
    total_values = 0
    total_genes = 0
    data_rows = rows[header_index + 1 :]
    for column_index, name in candidates:
        values = [
            row[column_index].strip()
            for row in data_rows
            if column_index < len(row) and row[column_index].strip()
        ]
        if not values:
            continue
        gene_cells = sum(bool(gene_values(value)) for value in values)
        if gene_cells < 2 or gene_cells / len(values) < 0.8:
            continue
        selected.append((column_index, name))
        total_values += len(values)
        total_genes += gene_cells
    data_width = max((len(row) for row in rows[header_index:]), default=0)
    active_columns = 0
    for column_index in range(data_width):
        if any(
            column_index < len(row) and row[column_index].strip()
            for row in rows[header_index:]
        ):
            active_columns += 1
    if (
        len(selected) < 2
        or total_genes < 6
        or total_values == 0
        or active_columns == 0
        or len(selected) / active_columns < 0.75
        or total_genes / total_values < 0.85
    ):
        return [], 0.0, 0
    return selected, total_genes / total_values, total_genes


def find_signature_header(
    rows: Sequence[Sequence[str]],
) -> Tuple[int, List[Tuple[int, str]]]:
    for header_index in range(min(20, len(rows))):
        row = rows[header_index]
        normalized = [normalize_header(value) for value in row if value.strip()]
        if len(normalized) >= 2 and any(
            is_non_signature_header(value)
            for value in normalized
        ):
            return -1, []
        columns, _, _ = signature_columns_for_header(rows, header_index)
        if columns:
            return header_index, columns
    return -1, []


def extract_signature_pairs(
    path: Path, header_index: int, columns: Sequence[Tuple[int, str]]
) -> List[Tuple[str, str]]:
    output: List[Tuple[str, str]] = []
    for row_index, row in enumerate(csv_rows(path)):
        if row_index <= header_index:
            continue
        for column_index, cell_name in columns:
            value = row_value(row, column_index)
            for gene in gene_values(value):
                output.append((cell_name.strip(), gene))
    return output


def source_table_paths(record_dir: Path) -> List[Path]:
    manifest_path = record_dir / "extracted" / "tables" / "tables_manifest.json"
    manifest = read_json(manifest_path, default=[]) or []
    paths: Dict[Path, None] = {}
    for entry in manifest:
        output_path = str(entry.get("output_path", ""))
        if not output_path:
            continue
        path = record_dir / output_path
        if path.is_file() and path.suffix.casefold() == ".csv":
            paths[path] = None
    tables_dir = record_dir / "extracted" / "tables"
    if tables_dir.exists():
        for path in tables_dir.glob("*.csv"):
            paths.setdefault(path, None)
    return sorted(paths, key=lambda path: str(path))


def write_csv_atomic(path: Path, rows: Iterable[Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def write_statistics(path: Path, rows: Sequence[Tuple[str, ...]]) -> None:
    write_csv_atomic(path, [STATISTICS_FIELDS, *rows])


def write_signatures(
    path: Path,
    cell_order: Sequence[str],
    genes_by_cell: Dict[str, List[str]],
) -> None:
    if not cell_order:
        write_csv_atomic(path, [])
        return
    row_count = max(len(genes_by_cell[cell_name]) for cell_name in cell_order)
    rows: List[List[str]] = [list(cell_order)]
    for row_index in range(row_count):
        rows.append(
            [
                genes_by_cell[cell_name][row_index]
                if row_index < len(genes_by_cell[cell_name])
                else ""
                for cell_name in cell_order
            ]
        )
    write_csv_atomic(path, rows)


def standardize_record(record_dir: Path, output_subdir: str = "standardized") -> Dict[str, Any]:
    output_dir = record_dir / output_subdir
    statistics_path = output_dir / "marker_statistics.csv"
    signatures_path = output_dir / "marker_signatures.csv"
    statistics_rows: List[Tuple[str, ...]] = []
    statistics_seen = set()
    cell_order: List[str] = []
    genes_by_cell: Dict[str, List[str]] = {}
    signature_seen = set()
    statistics_sources: List[str] = []
    signature_sources: List[str] = []
    statistics_duplicates = 0
    signature_duplicates = 0
    errors: List[str] = []
    input_paths = source_table_paths(record_dir)

    for path in input_paths:
        relative_path = str(path.relative_to(record_dir))
        try:
            sample = read_sample(path)
            _, statistics_mappings = find_statistics_header(sample)
            if statistics_mappings:
                extracted_statistics = extract_statistics_rows(path)
                added = 0
                for values in extracted_statistics:
                    if values in statistics_seen:
                        statistics_duplicates += 1
                        continue
                    statistics_seen.add(values)
                    statistics_rows.append(values)
                    added += 1
                if added:
                    statistics_sources.append(relative_path)
                continue

            header_index, signature_columns = find_signature_header(sample)
            if header_index < 0:
                continue
            extracted_pairs = extract_signature_pairs(path, header_index, signature_columns)
            added = 0
            for cell_name, gene in extracted_pairs:
                key = (cell_name, gene)
                if key in signature_seen:
                    signature_duplicates += 1
                    continue
                signature_seen.add(key)
                if cell_name not in genes_by_cell:
                    genes_by_cell[cell_name] = []
                    cell_order.append(cell_name)
                genes_by_cell[cell_name].append(gene)
                added += 1
            if added:
                signature_sources.append(relative_path)
        except (OSError, csv.Error, UnicodeError) as exc:
            errors.append(f"{relative_path}: {type(exc).__name__}: {exc}")

    write_statistics(statistics_path, statistics_rows)
    write_signatures(signatures_path, cell_order, genes_by_cell)
    summary = {
        "generated_at": utc_now(),
        "input_tables": len(input_paths),
        "statistics_source_tables": statistics_sources,
        "signature_source_tables": signature_sources,
        "statistics_rows": len(statistics_rows),
        "signature_cell_names": len(cell_order),
        "signature_cell_gene_pairs": len(signature_seen),
        "duplicates_removed": {
            "statistics": statistics_duplicates,
            "signatures": signature_duplicates,
        },
        "errors": errors,
        "outputs": {
            "statistics": str(statistics_path.relative_to(record_dir)),
            "signatures": str(signatures_path.relative_to(record_dir)),
        },
    }
    write_json(output_dir / "standardization_summary.json", summary)
    return summary


def standardize_records(
    database: Path,
    records: Iterable[Tuple[Path, Dict[str, Any]]],
    output_subdir: str = "standardized",
    summary_name: str = "standardization_summary.json",
) -> Dict[str, Any]:
    totals = {
        "papers": 0,
        "input_tables": 0,
        "statistics_source_tables": 0,
        "signature_source_tables": 0,
        "statistics_rows": 0,
        "signature_cell_gene_pairs": 0,
        "errors": 0,
    }
    for record_dir, _ in records:
        summary = standardize_record(record_dir, output_subdir=output_subdir)
        totals["papers"] += 1
        totals["input_tables"] += summary["input_tables"]
        totals["statistics_source_tables"] += len(summary["statistics_source_tables"])
        totals["signature_source_tables"] += len(summary["signature_source_tables"])
        totals["statistics_rows"] += summary["statistics_rows"]
        totals["signature_cell_gene_pairs"] += summary["signature_cell_gene_pairs"]
        totals["errors"] += len(summary["errors"])
    result = {
        "generated_at": utc_now(),
        "database": str(database),
        "output_subdir": output_subdir,
        **totals,
    }
    write_json(database / summary_name, result)
    return result
