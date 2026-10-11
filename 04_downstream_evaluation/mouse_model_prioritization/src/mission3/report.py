from __future__ import annotations

import html
import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

from .models import ModelScore, PhenotypeEvidence


def write_jsonl_reports(
    path: str | Path,
    rankings: Mapping[str, Sequence[ModelScore]],
    evidence_by_pair: Mapping[tuple[str, str], Sequence[PhenotypeEvidence]],
) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for disease_id, rows in sorted(rankings.items()):
            for score in rows:
                evidence = evidence_by_pair.get((disease_id, score.model_id), ())
                gaps = [row.hpo_id for row in evidence if row.assay_state != "reproduced"]
                payload = {
                    "disease_id": disease_id,
                    "model": asdict(score),
                    "phenotype_evidence": [asdict(row) for row in evidence],
                    "gap_hpo_ids": gaps,
                    "recommendation": recommendation(score, evidence),
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def recommendation(score: ModelScore, evidence: Sequence[PhenotypeEvidence]) -> str:
    if score.genetic_class == "conflicting":
        return "Phenotype evidence may be useful, but the allele mechanism conflicts with the human disease."
    if score.cpo_score is not None and score.cpo_score >= 0.6 and score.surface_score >= 0.6:
        return "Prioritize for mechanism-focused studies; surface and cellular-process evidence are concordant."
    if score.surface_score >= 0.6 and score.cpo_coverage == 0:
        return "Phenotype-similar model; cellular mechanism remains untested because CPO evidence is missing."
    if any(row.assay_state == "tested_normal" for row in evidence):
        return "Use with caution: at least one comparable disease phenotype was tested and found normal."
    missing_cells = sorted(
        {
            row.best_hpo_cl_id
            for row in evidence
            if row.assay_state != "reproduced" and row.best_hpo_cl_id
        }
    )
    drivers = sorted(
        {
            driver
            for row in evidence
            if row.assay_state != "reproduced"
            for driver in row.suggested_cre_drivers
        }
    )
    if drivers:
        return "Mechanism gap: review cell-matched Cre drivers " + ", ".join(drivers) + "."
    if missing_cells:
        return "Mechanism gap: prioritize a conditional model targeting " + ", ".join(missing_cells) + "."
    return "Insufficient evidence for a strong suitability recommendation."


def _fmt(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.3f}"
    return html.escape(str(value))


def write_html_report(
    path: str | Path,
    rankings: Mapping[str, Sequence[ModelScore]],
    evidence_by_pair: Mapping[tuple[str, str], Sequence[PhenotypeEvidence]],
    evaluation: Mapping[str, object],
    warnings: Sequence[str],
) -> None:
    disease_sections: list[str] = []
    for disease_id, rows in sorted(rankings.items()):
        table_rows: list[str] = []
        details: list[str] = []
        for score in rows:
            evidence = evidence_by_pair.get((disease_id, score.model_id), ())
            table_rows.append(
                "<tr>"
                f"<td>{score.rank}</td><td><a href='#{html.escape(disease_id)}-{html.escape(score.model_id)}'>{html.escape(score.model_name)}</a></td>"
                f"<td>{_fmt(score.surface_score)}</td><td>{_fmt(score.cpo_score)}</td>"
                f"<td>{_fmt(score.cpo_coverage)}</td><td>{html.escape(score.genetic_class)}</td>"
                f"<td>{_fmt(score.total_score)}</td><td>{'yes' if score.known_model else 'no'}</td>"
                "</tr>"
            )
            evidence_rows = "".join(
                "<tr>"
                f"<td>{html.escape(row.hpo_id)}</td><td>{html.escape(row.hpo_name)}</td>"
                f"<td>{html.escape(row.best_mp_id or 'NA')}</td><td>{_fmt(row.surface_similarity)}</td>"
                f"<td><span class='state {html.escape(row.assay_state)}'>{html.escape(row.assay_state)}</span></td>"
                f"<td>{_fmt(row.cpo_score)}</td><td>{html.escape(row.best_hpo_cl_id or 'NA')}</td>"
                f"<td>{html.escape(row.best_hpo_go_id or 'NA')}</td>"
                "</tr>"
                for row in evidence
            )
            details.append(
                f"<details id='{html.escape(disease_id)}-{html.escape(score.model_id)}'>"
                f"<summary>#{score.rank} {html.escape(score.model_name)}: {html.escape(recommendation(score, evidence))}</summary>"
                f"<p><strong>Genetic mechanism:</strong> {html.escape(score.genetic_reason)}</p>"
                "<table><thead><tr><th>HPO</th><th>Phenotype</th><th>Best MP</th><th>Surface</th>"
                "<th>Assay state</th><th>CPO</th><th>Human CL</th><th>Human GO</th></tr></thead>"
                f"<tbody>{evidence_rows}</tbody></table></details>"
            )
        title = rows[0].disease_name if rows else disease_id
        disease_sections.append(
            f"<section><h2>{html.escape(title)} <small>{html.escape(disease_id)}</small></h2>"
            "<table><thead><tr><th>Rank</th><th>Model</th><th>Surface</th><th>CPO</th>"
            "<th>CPO coverage</th><th>Genetic</th><th>Total</th><th>Known</th></tr></thead>"
            f"<tbody>{''.join(table_rows)}</tbody></table>{''.join(details)}</section>"
        )
    warning_html = "".join(f"<li>{html.escape(item)}</li>" for item in warnings) or "<li>None</li>"
    evaluation_text = html.escape(json.dumps(evaluation, ensure_ascii=False, indent=2))
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mission 3 mouse-model suitability</title>
<style>
body{{font-family:Arial,sans-serif;margin:0;color:#202124;background:#f7f8fa}}main{{max-width:1280px;margin:auto;padding:24px}}
h1{{font-size:28px}}h2{{margin-top:36px}}small{{color:#687078;font-weight:normal}}section{{border-top:1px solid #ccd1d6}}
table{{width:100%;border-collapse:collapse;background:white;margin:12px 0 18px}}th,td{{padding:8px;border:1px solid #dfe3e7;text-align:left;font-size:13px}}
th{{background:#eef1f3}}details{{background:white;border:1px solid #dfe3e7;margin:8px 0;padding:10px}}summary{{cursor:pointer;font-weight:600}}
.state{{font-weight:600}}.reproduced{{color:#18733c}}.tested_normal,.opposite_direction{{color:#a33622}}.untested,.not_mappable{{color:#6b7075}}
pre{{white-space:pre-wrap;background:#202124;color:#f5f5f5;padding:14px;overflow:auto}}a{{color:#075aa8}}
</style></head><body><main>
<h1>Mechanism-aware mouse-model suitability</h1>
<p>Missing mechanism evidence is treated as unknown. Availability is not included in biological suitability scores.</p>
<h2>Run warnings</h2><ul>{warning_html}</ul>
{''.join(disease_sections)}
<h2>Evaluation</h2><pre>{evaluation_text}</pre>
</main></body></html>"""
    Path(path).write_text(document, encoding="utf-8")
