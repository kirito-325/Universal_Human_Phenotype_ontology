"""Output requirements shared by model instructions and deterministic validation."""

import copy
import json

from pydantic import ValidationError
from pydantic_core import PydanticCustomError

DIRECTION_DESCRIPTION = (
    "Direction of the named cellular process in process_text, or the cellular quality in quality_text "
    "when process_text is empty; not the polarity of the input radiographic/clinical phenotype."
)
SOURCE_DOCUMENT_DESCRIPTION = (
    "Exact document_id returned by read_document for the window containing this quote. "
    "Abstract PMID and full-text PMC documents are distinct sources even for the same paper."
)
QUOTE_DESCRIPTION = "One exact contiguous substring copied from a read window of the named document; never paraphrase or join fragments."

SCIENCE_POLICY = """
SCIENTIFIC ACCEPTANCE STANDARD: SUPPORT-FIRST MECHANISM EXPLORATION
Accept a scientifically defensible inference with explicit assumptions and inferred
provenance. Missing exact-site experiments, missing citations for model knowledge,
zero search hits and alternative etiologies alone are not rejection reasons.
Assess logical coherence and biological applicability separately from experimental
validation. Seek support for the key reasoning steps and state the assumptions bridging
remaining gaps. Dedicated challenge searches and exhaustive exclusion of alternatives
are not required. Related papers or shared genes alone do not justify a complete mechanism.
Evidence effect/scope labels are not automatic scientific vetoes. Other-model negative
findings and absence of a reported phenotype usually limit transfer, not refute the target
prediction. Judge their actual applicability. If an already-read source explicitly
invalidates a necessary premise under applicable conditions, refine the prediction or
identify that concrete defect. Source misinterpretation and unjustifiable biology remain
blocking issues. Do not upgrade an untested prediction to demonstrated causality.
Judge direction against the named cellular process/quality, not against the input
phenotype's polarity. Decreased clearance can plausibly produce increased deposits;
different signs at different steps are not by themselves a direction_mismatch.
A direction defect requires an inversion of the same process/comparison or an
incoherent proposed mechanism. Explain the concrete conflicting propositions and
their source or biological basis. Lack of a study is not contradictory evidence.
Alternatives and concurrent abnormalities do not by themselves disprove a proposed
contribution. Refine unsupported optional etiologic detail before rejecting a
defensible cellular-abnormality conclusion.
"""

FIELD_PLACEMENT_GUIDE = """
FIELD PLACEMENT BEFORE SUBMISSION
Paths below use claim for SubmitAssessment; Extraction uses claims[i] instead.
- claim.supporting_quote and claim.supporting_document_id are one optional main-quote
  pair. Copy BOTH from the same read_document result. Do not replace a PMID document
  ID with the paper's PMC ID, even if the title/PMID is identical. Empty quote => empty ID.
- claim.reasoning_steps[i].citations[j] holds the exact {document_id,quote} for that
  premise. Different premises may cite different read documents. Omit span_start
  unless the host supplies the exact absolute offset; no source metadata belongs here.
- claim.reasoning_steps[i].assumptions holds applicability assumptions. A factual
  literature step has assumptions=[]; use inference with citations+assumptions, or
  model_knowledge with citations=[] for learned premises. For inferred cell identity,
  include scope=cell_identity and basis=inference/model_knowledge; a literature-only
  cell_identity step does not explain the extrapolation.
- Only SubmitAssessment has supporting_document_ids / contradicting_document_ids, lists of
  actually read sources, not fields inside claim and not lists of unexamined search hits.
- claim.experimental_evidence holds actual study findings, species, context and scope;
  claim.reasoning_steps holds your interpretation. Record used primary experiments as
  component evidence when appropriate; [] means no experimental entry was recorded,
  not that no relevant experiment exists. Review speculation is not an experiment.
- No basis_notes, assumptions_note or other invented fields.
"""

COMMON_SUPPORT_FIELDS = ("cell_supported", "process_supported", "abnormality_supported",
                         "phenotype_bridge_supported", "species_supported")
INFERRED_SUPPORT_FIELDS = ("inference_supported", "scope_transfer_supported")

RULES = {
    "step.definition": "definition step uses supplied phenotype metadata, not literature or assumptions: citations=[] and assumptions=[].",
    "step.model_knowledge": "model_knowledge has no claimed literature citations: citations=[]; assumptions are allowed. Split cited premises into a literature/inference step.",
    "step.citations": "literature and inference steps require source quotations in citations; uncited learned premises use basis=model_knowledge.",
    "step.inference_assumptions": "inference step requires explicit applicability assumptions and cited premises.",
    "step.assumptions": "steps with assumptions must be labeled inference or model_knowledge; factual literature steps use assumptions=[].",
    "claim.quote_source": "supporting_document_id requires a supporting_quote; when quote is empty, source ID must be empty.",
    "claim.source_unstaged": "Main quotation source is not staged; use an existing document_id and read it before citing.",
    "claim.source_unread": "Main quotation source has not been read in this task; read the named document before citing.",
    "claim.source_mismatch": "Main quotation matches another actually read document, not the selected source. Copy the document_id from that quote's read window; abstract and full text are not interchangeable.",
    "claim.source_ambiguous": "Main quotation matches multiple read documents; choose its exact supporting_document_id.",
    "claim.quote_unread": "Main quotation exists in the selected document but not in a supplied read window; read that exact window first.",
    "claim.quote_verbatim": "Main quotation is not verbatim in the selected read source; copy an exact contiguous substring, keeping punctuation and case.",
    "claim.unsourced_modes": "claims without a source quote must mark cell and overall evidence inferred.",
    "claim.unsourced_link": "claims without a source quote cannot declare a directly reported phenotype link: link_evidence_mode=inferred and a non-direct bridge_scope.",
    "claim.unsourced_provenance": "claims without a source quote require model_inference / inference provenance: assertion=model_inference, evidence_type=inference.",
    "claim.mode_summary": "evidence_mode must summarize cell_evidence_mode and link_evidence_mode: inferred if either is inferred, otherwise direct.",
    "claim.link_scope": "link_evidence_mode must agree with bridge_scope: direct iff bridge_scope=direct_phenotype.",
    "claim.direct": "direct evidence requires direct_phenotype and no inferred bridge: reasoning_steps=[].",
    "claim.inference": "inferred evidence requires a chain with an explicit inference step: at least one basis=inference or model_knowledge.",
    "claim.cell_inference": "inferred cell identity requires a cell_identity inference or model_knowledge step; a literature-only identity step is insufficient to declare an inferred identity.",
    "claim.inference_relation": "inferred_link_requires_noncausal_relation_type: unproven causal explanations use mechanistically_compatible_with or associated_with, not the demonstrated-causality label causally_upstream_of.",
    "claim.chain": "reasoning steps must form a connected ordered chain: each step.source exactly equals the previous step.target.",
    "claim.endpoint": "reasoning chain must end at cell_text + ': ' + quality_text, exactly.",
    "claim.start": "reasoning_chain_must_start_at_phenotype_label: accepted forms are INPUT_PHENOTYPE, input ID, label, label (input ID), input ID: label, input ID (label). Whitespace/case is normalized; unrelated labels or IDs are forbidden. The host stores the canonical label.",
    "claim.taxon": "species/taxon mismatch: human=NCBITaxon:9606, mouse=NCBITaxon:10090, rat=NCBITaxon:10116, unknown=NCBITaxon:0; other_model requires its actual different taxon ID.",
    "citation.offset": "quote_not_in_supplied_window: quote and supplied span_start must match exactly within a read window. Omit optional span_start for host positioning; never guess offsets.",
    "citation.quote": "quote_not_in_supplied_window: copy an exact contiguous quote from an actually read window.",
    "citation.source_mismatch": "Quotation matches another actually read document, not its declared citation source. Use that read window's exact document_id; abstract and full text are separate sources.",
    "reasoning.quote": "reasoning_quote_not_in_read_windows: cite only an exact quote in a read window of the named document.",
    "reasoning.source": "reasoning_citation_document_not_staged_or_read: read the named staged source before citing it.",
    "hypothesis.seed_fields": "New hypothesis seed output contains document_id and quote only; span_start is not an output field. The host computes offsets. Legacy stored seeds are migrated separately.",
    "mapping.label": "selected ID requires its exact canonical label; for ID=NONE or empty, label must be empty and confidence=unmapped.",
    "mapping.id": "invalid_or_obsolete_mapping: choose an existing, non-obsolete ID in the requested CL/GO namespace, or NONE with unmapped confidence.",
    "mapping.canonical": "id_label_mismatch: the supplied label must exactly match the selected ID's canonical index label.",
    "mapping.confidence": "selected term cannot have unmapped confidence; confidence describes term fit, not experimental proof.",
    "assessment.sources": "Assessment source lists contain unread documents; read these sources or remove their IDs from the named list.",
    "assessment.reassessment_note": "Explicit scientific reevaluation must explain in revision_note whether and why the previous revised assessment changed; a format failure is not scientific counterevidence.",
    "assessment.repair_path": "Format repair may change evidence/provenance fields only; scientific decision, rationale and cellular claim are fixed. Use explicit reevaluate with a reassessment_reason for scientific changes.",
    "evidence.ref": "Unknown or unread evidence_ref; select a passage returned to this task. The host supplies quotes, document IDs, hashes and offsets; do not invent or substitute publication IDs.",
    "review.experiments": "Recorded experimental entries require a scientific judgment of their actual interpretation, species and scope.",
    "submission.conflict": "Two supplied values conflict. Preserve the existing scientific content; do not guess which value to overwrite.",
    "assessment.claim_required": "retain requires cell/abnormality/phenotype_link in claim; supply the scientific proposal.",
    "assessment.decision_claim": "reject/inconclusive require claim=null; the decision and retained scientific proposal conflict.",
}

REPAIR_CLAIM_FIELDS = {
    "supporting_document_id", "supporting_quote", "reasoning_steps", "experimental_evidence",
    "evidence_mode", "cell_evidence_mode", "link_evidence_mode", "assertion", "evidence_type",
    "taxon_id", "hypothesis_id",
}


def apply_assessment_repair(previous, updates):
    """Apply field patches without regenerating the scientific assessment."""
    result = copy.deepcopy(previous)
    for index, update in enumerate(updates):
        path = update["path"]
        allowed = (path[0] in {"supporting_document_ids", "contradicting_document_ids"}
                   or len(path) >= 2 and path[0] == "claim" and path[1] in REPAIR_CLAIM_FIELDS)
        if not allowed:
            raise contract_error("assessment.repair_path", field=("updates", index, "path"), actual=path)
        target = result
        try:
            for key in path[:-1]:
                target = target[key]
            key = path[-1]
            if isinstance(target, list) and (type(key) is not int or key < 0):
                raise IndexError
            if not isinstance(target, (dict, list)):
                raise TypeError
            if update.get("operation", "set") == "remove":
                del target[key]
            else:
                target[key] = copy.deepcopy(update.get("value"))
        except (KeyError, IndexError, TypeError) as error:
            raise contract_error("assessment.repair_path", field=("updates", index, "path"),
                                 expected="an existing object/list path in previous_submission", actual=path) from error
    return result


def contract_error(code, *, field=(), expected=None, actual=None, related_paths=None):
    context = {"field": list(field)}
    message = RULES[code]
    if expected is not None:
        context["expected"] = expected
        message += " Expected {expected}."
    if actual is not None:
        context["actual"] = actual
        message += " Received {actual}."
    if related_paths:
        context["related_paths"] = related_paths
    return PydanticCustomError(code, message, context)


def output_contract(role, *, compact=False):
    if compact:
        return "\nCOMPACT OUTPUT CONTRACT\n" + (
            "Use only the final tool's Schema fields and JSON types. Host-owned metadata, "
            "chain strings, evidence flags, canonical labels and derived review booleans are not model fields. "
            "Choose supplied evidence_ref IDs instead of copying source text or computing offsets. "
            "Scientific explanations, direction, assumptions, experiment interpretation and ontology choice remain your responsibility. "
            "An unknown ref or missing scientific explanation cannot be repaired by inventing a value. "
            "Reasonable uncited model knowledge and absent experimental validation remain allowed.\n"
        )
    prefixes = {
        "extract": ("step.", "claim.", "citation.", "reasoning."),
        "reason": ("step.", "claim.", "citation.", "reasoning."),
        "verify_hypothesis": ("step.", "claim.", "citation.", "reasoning.", "assessment."),
        "normalize": ("mapping.",),
        "hypothesize": ("hypothesis.", "citation.quote"),
    }.get(role, ())
    lines = [f"- {code}: {text}" for code, text in RULES.items() if code.startswith(prefixes)] if prefixes else []
    if role == "review":
        lines.extend([
            "- For inferred claims the following fields mean scientifically defensible, NOT experimentally proved: "
            + ", ".join((*COMMON_SUPPORT_FIELDS, *INFERRED_SUPPORT_FIELDS, "cell_inference_supported")) + ".",
            "- Missing experimental validation is allowed. Use blocking_issues only for concrete scientific defects.",
            "- causal_supported is applicable only to a demonstrated causal relation. Inapplicable checks must not veto an inferred prediction.",
            "- partial/insufficient_evidence with defensible inference checks can be retained conditionally; they do not automatically reject an inferred prediction.",
        ])
    return "\nFORMAL OUTPUT CONTRACT\n" + (
        "Use only the submission tool's JSON Schema field names, types and enum values; unknown fields are forbidden. "
        "Arrays must be arrays, not JSON-encoded strings; booleans/numbers must use their JSON types, not strings. "
        "Scientific plausibility and format validity are separate.\n"
        + "\n".join(lines)
        + (FIELD_PLACEMENT_GUIDE if role in {"extract", "reason", "verify_hypothesis"} else "")
        + "\nOn invalid_format, correct the named fields using expected/actual feedback and resubmit. "
          "Do not change unrelated prose, retrieve unrelated papers, or label a format error as a rejected scientific hypothesis.\n"
    )


def submission_error_feedback(error, *, compact=False):
    """Use exception metadata, not parsing the default human-facing traceback."""
    tool = getattr(error, "tool_name", None)
    current = getattr(error, "source", error)
    last_cause = current
    seen = set()
    while not isinstance(current, ValidationError) and current is not None and id(current) not in seen:
        seen.add(id(current))
        last_cause = current
        current = current.__cause__
    issues = []
    if isinstance(current, ValidationError):
        for entry in current.errors(include_url=False):
            context = entry.get("ctx", {})
            nested = context.get("error")
            if isinstance(nested, ResearchActionRequired):
                raise nested
            if hasattr(nested, "diagnostics"):
                for detail in nested.diagnostics:
                    field = "document_id" if detail["error_code"] == "source_not_read" else "quote"
                    issues.append({"code": detail["error_code"], "path": ["hypotheses", detail["hypothesis_index"] - 1,
                        "seed_evidence", detail["citation_index"] - 1, field], "message": detail["next_action"]})
                continue
            path = [*entry["loc"], *context.get("field", [])]
            if not compact and tool in {"SubmitAssessment", "SubmitAssessmentRepair"} and path and path[0] in {"reasoning_steps", "cell_evidence_mode",
                    "evidence_mode", "link_evidence_mode", "bridge_scope", "supporting_document_id",
                    "assertion", "taxon_id", "supporting_quote", "experimental_evidence"}:
                path.insert(0, "claim")
            issue = {"code": entry["type"], "path": path, "message": entry["msg"]}
            for key in ("expected", "actual", "related_paths"):
                if key in context:
                    issue[key] = context[key]
            issues.append(issue)
    else:
        if isinstance(last_cause, ResearchActionRequired):
            raise last_cause
        if isinstance(last_cause, (TypeError, KeyError, AttributeError)):
            raise last_cause
        issues.append({"code": "submission_validation_failed", "path": [], "message": str(getattr(error, "source", error))})
    status = "invalid_format"
    if compact:
        if any(issue["code"] == "submission.conflict" for issue in issues):
            status = "field_conflict"
        elif any(issue["code"] == "evidence.ref" for issue in issues):
            status = "invalid_reference"
        elif any(issue["code"] == "assessment.decision_claim" for issue in issues):
            status = "scientific_conflict"
        elif any(issue["code"] in {"missing", "string_too_short", "too_short", "assessment.claim_required"} for issue in issues):
            status = "needs_reasoning"
    actions = {"invalid_format": "correct_fields_and_resubmit", "field_conflict": "resolve_conflicting_values",
               "invalid_reference": "select_a_read_reference", "needs_reasoning": "complete_missing_scientific_content",
               "scientific_conflict": "explicit_scientific_reevaluation"}
    return json.dumps({"status": status, "tool": tool, "issues": issues, "next_action": actions[status]}, ensure_ascii=False)


class SubmissionRepairError(ValueError):
    def __init__(self, tool, issues, arguments, *, status="invalid_format"):
        self.tool = tool
        self.issues = issues
        self.arguments = arguments
        self.status = status
        super().__init__(f"no_progress: unresolved submission validation errors in {tool}; address the named fields")


class ResearchActionRequired(ValueError):
    def __init__(self, status, message, **details):
        self.status = status
        self.details = details
        super().__init__(message)
