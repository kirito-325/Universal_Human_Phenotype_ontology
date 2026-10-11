"""Literature-first prerequisites. The root still chooses every research action."""

import re

from .schemas import DiscoveryPlan, HypothesisBatch, QuoteReference
from .storage import fingerprint
from .validation import anchor_quote


def normalized(value):
    return re.sub(r"\W+", " ", value.casefold()).strip()


class HypothesisCitationError(ValueError):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__("hypothesis_citation_errors: " + str(diagnostics))


class ResearchWorkflow:
    def __init__(self, session):
        self.session = session
        self.enabled = session.settings.workflow_mode == "literature_first"
        self.background = None
        self.plan = None
        self.hypotheses = {}
        self.generation_notes = []

    def snapshot(self):
        return {"enabled": self.enabled, "background": self.background, "discovery_plan": self.plan,
                "hypotheses": self.hypotheses, "generation_notes": self.generation_notes,
                "requirements": {"hypotheses": self.session.settings.min_hypotheses,
                                 "verification_intents": ["support"], "challenge_required": False,
                                 "discovery_directions": self.session.settings.min_discovery_directions,
                                 "discovery_papers": self.session.settings.min_discovery_papers}}

    def register_plan(self, plan: DiscoveryPlan):
        if self.background is None:
            raise ValueError("retrieve_phenotype_background_before_planning")
        if self.hypotheses:
            raise ValueError("discovery_plan_frozen_after_hypothesis_generation")
        if len(plan.directions) < self.session.settings.min_discovery_directions:
            raise ValueError("insufficient_discovery_directions")
        if len({d.direction_id for d in plan.directions}) != len(plan.directions):
            raise ValueError("duplicate_discovery_direction_id")
        if len({d.kind for d in plan.directions}) < 3:
            raise ValueError("discovery_requires_at_least_three_distinct_search_facets")
        queries = [normalized(q) for d in plan.directions for q in d.queries]
        if len(set(queries)) != len(queries):
            raise ValueError("duplicate_queries_do_not_establish_breadth")
        self.plan = plan.model_dump()

    def check_search(self, *, direction_id="", hypothesis_id="", intent="discovery"):
        if not self.enabled:
            return
        if self.background is None:
            raise ValueError("retrieve_phenotype_background_before_literature_search")
        if intent == "discovery":
            if not self.plan or direction_id not in {d["direction_id"] for d in self.plan["directions"]}:
                raise ValueError("discovery_search_requires_registered_direction_id")
            if hypothesis_id:
                raise ValueError("hypothesis_search_requires_support_or_challenge_intent")
        elif intent in {"support", "challenge"}:
            if hypothesis_id not in self.hypotheses:
                raise ValueError("validation_search_requires_known_hypothesis_id")
        else:
            raise ValueError("invalid_search_intent")

    def discovery_gaps(self):
        if not self.enabled:
            return []
        gaps = []
        if self.background is None:
            gaps.append("phenotype_background")
        if not self.plan:
            return [*gaps, "discovery_plan"]
        searched = {q.get("direction_id") for q in self.session.search_log
                    if q.get("intent") == "discovery" and "error" not in q and q.get("retmax", 0) > 0}
        gaps.extend("discovery_search:" + d["direction_id"] for d in self.plan["directions"]
                    if d["direction_id"] not in searched)
        papers = {d.pmid or d.document_id for d in self.session.documents.values() if not d.retracted}
        if len(papers) < self.session.settings.min_discovery_papers:
            gaps.append("retrieve_more_discovery_papers")
        return gaps

    def require_discovery(self):
        if gaps := self.discovery_gaps():
            raise ValueError("complete_broad_literature_recall_before_hypotheses: " + ", ".join(gaps))

    def register_hypotheses(self, batch: HypothesisBatch, provided, *, commit=True, extend=False):
        self.require_discovery()
        if self.hypotheses and not extend:
            raise ValueError("hypothesis_batch_already_registered; refine individual hypotheses during verification")
        minimum = 1 if self.hypotheses else self.session.settings.min_hypotheses
        if len(batch.hypotheses) < minimum:
            raise ValueError(f"at_least_{minimum}_distinct_hypotheses_required; report a limitation instead of inventing candidates")
        pairs = {(normalized(h.cell_text), normalized(h.process_text), h.direction, normalized(h.mechanism))
                 for h in batch.hypotheses}
        if len(pairs) != len(batch.hypotheses):
            raise ValueError("hypotheses_must_differ_in_cell_or_process_or_direction_not_just_wording")
        existing = {(normalized(h["draft"]["cell_text"]), normalized(h["draft"]["process_text"]),
                     h["draft"]["direction"], normalized(h["draft"]["mechanism"])) for h in self.hypotheses.values()}
        if pairs & existing:
            raise ValueError("new_hypotheses_duplicate_existing_candidates")
        direction_ids = {d["direction_id"] for d in self.plan["directions"]}
        used = {h.direction_id for h in batch.hypotheses} | {h["draft"]["direction_id"] for h in self.hypotheses.values()}
        if not used <= direction_ids or len(used) < 3:
            raise ValueError("hypotheses_must_cover_at_least_three_registered_directions")
        read_ids = {w["document_id"] for w in provided}
        required_reads = 1 if self.hypotheses else self.session.settings.min_discovery_papers
        if len({self.session.documents[d].pmid or d for d in read_ids}) < required_reads:
            raise ValueError("read_the_discovery_corpus_before_proposing_hypotheses")
        pending = {}
        diagnostics = []
        for hypothesis_index, draft in enumerate(batch.hypotheses, 1):
            spans = []
            for citation_index, ref in enumerate(draft.seed_evidence, 1):
                issue = {"hypothesis_index": hypothesis_index, "citation_index": citation_index,
                         "document_id": ref.document_id}
                if ref.document_id not in read_ids:
                    diagnostics.append({**issue, "error_code": "source_not_read",
                                        "next_action": "read the staged document before citing it"})
                    continue
                doc = self.session.documents[ref.document_id]
                matches = []
                for window in provided:
                    if window["document_id"] == ref.document_id:
                        try:
                            matches.append(anchor_quote(QuoteReference(**ref.model_dump()), doc,
                                                        window["start"], window["end"]))
                        except ValueError:
                            continue
                if not matches:
                    code = "retracted_source" if doc.retracted else (
                        "quote_not_verbatim" if ref.quote not in doc.text else "quote_not_in_read_windows")
                    diagnostics.append({**issue, "error_code": code,
                                        "next_action": "copy an exact quote from an actually read window"})
                    continue
                spans.append(matches[0].model_dump())
            hid = "hyp_" + fingerprint({"phenotype": self.session.phenotype.phenotype_id,
                                       "draft": draft.model_dump(), "seed_evidence": spans})[:24]
            pending[hid] = {"hypothesis_id": hid, "draft": draft.model_dump(), "seed_evidence": spans,
                            "status": "proposed", "assessment": None, "assessment_history": [], "claim_id": None,
                            "created_after_search_count": self.session.searches}
        if diagnostics:
            raise HypothesisCitationError(diagnostics)
        if commit:
            self.hypotheses.update(pending)
            self.generation_notes.append({"diversity_rationale": batch.diversity_rationale, "limitation": batch.limitation})
        return list(pending.values())

    def validation_searches(self, hypothesis_id):
        item = self.hypotheses[hypothesis_id]
        return [q for q in self.session.search_log if q.get("hypothesis_id") == hypothesis_id
                and q.get("search_number", 0) > item["created_after_search_count"] and "error" not in q]

    def require_verification(self, hypothesis_id):
        if hypothesis_id not in self.hypotheses:
            raise ValueError("unknown_hypothesis_id")
        searches = self.validation_searches(hypothesis_id)
        intents = {q.get("intent") for q in searches if q.get("retmax", 0) > 0}
        if "support" not in intents:
            raise ValueError("run_support_literature_search_after_generating_this_hypothesis")

    def require_normalization(self, claim_id):
        if not self.enabled:
            return
        if len(self.hypotheses) < self.session.settings.min_hypotheses:
            raise ValueError("generate_and_verify_the_hypothesis_batch_before_normalization")
        if any(h["status"] == "proposed" for h in self.hypotheses.values()):
            raise ValueError("finish_verifying_all_hypotheses_before_normalization")
        if not any(h["status"] == "retained" and h["claim_id"] == claim_id for h in self.hypotheses.values()):
            raise ValueError("only_retained_hypotheses_can_be_normalized")

    def pending(self):
        if not self.enabled:
            return []
        gaps = self.discovery_gaps()
        if len(self.hypotheses) < self.session.settings.min_hypotheses:
            gaps.append("generate_minimum_hypothesis_batch")
        gaps.extend("verify:" + hid for hid, h in self.hypotheses.items() if h["status"] == "proposed")
        gaps.extend("resolve_retained:" + hid for hid, h in self.hypotheses.items()
                    if h["status"] == "retained" and (not h["claim_id"] or h["claim_id"] in self.session.retired))
        return gaps
