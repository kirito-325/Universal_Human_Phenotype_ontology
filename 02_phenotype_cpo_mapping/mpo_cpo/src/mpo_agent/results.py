"""Strict model provenance and MP-level union of reviewed cellular mechanisms."""

from typing import Literal

from pydantic import Field, model_validator

from .schemas import (
    SCHEMA_VERSION,
    ClaimDraft,
    Direction,
    MouseModel,
    Observation,
    Phenotype,
    Record,
    ResearchCompletion,
)


class ModelResearchResult(Record):
    model: MouseModel
    status: Literal["pending", "researching", "mechanisms_found", "inconclusive", "needs_sources", "deferred", "processing_failed"]
    reason: str
    observations: list[Observation]
    coverage: dict
    errors: list[dict]


class ScopedObservation(Record):
    model_id: str
    observation: Observation


class MechanismMember(Record):
    model_id: str
    claim_id: str
    claim: ClaimDraft


class MechanismGroup(Record):
    mechanism_group_id: str
    cl_id: str
    go_id: str
    direction: Direction
    model_ids: list[str] = Field(min_length=1)
    members: list[MechanismMember] = Field(min_length=1)
    scope_note: str


class MPResult(Record):
    schema_version: str = SCHEMA_VERSION
    run_fingerprint: str
    phenotype: Phenotype
    status: Literal["completed", "partial", "processing_failed"]
    models: list[ModelResearchResult]
    relations: list[ScopedObservation]
    mechanisms: list[MechanismGroup]
    completion: ResearchCompletion | None
    summary_counts: dict
    context_sha256: str
    errors: list[dict] = Field(default_factory=list)

    @model_validator(mode="after")
    def provenance(self):
        models = {row.model.model_id: row for row in self.models}
        if len(models) != len(self.models):
            raise ValueError("duplicate_model_results")
        for relation in self.relations:
            model = models.get(relation.model_id)
            observation = relation.observation
            if model is None or model.status != "mechanisms_found":
                raise ValueError("relation_requires_a_completed_supporting_model")
            if observation.phenotype_id != self.phenotype.phenotype_id or observation.claim.species != "mouse":
                raise ValueError("MP_relation_requires_the_target_MP_and_mouse_cells")
            if observation.release_scope not in {"primary", "conditional", "cross_species"}:
                raise ValueError("only_reviewed_accepted_relations_are_aggregated")
            if not any(item.claim_id == observation.claim_id and item == observation for item in model.observations):
                raise ValueError("relation_not_in_its_model_result")
        relations = {(row.model_id, row.observation.claim_id): row.observation for row in self.relations}
        seen = set()
        for group in self.mechanisms:
            if set(group.model_ids) != {member.model_id for member in group.members}:
                raise ValueError("mechanism_group_model_ids_must_match_its_members")
            for member in group.members:
                key = (member.model_id, member.claim_id)
                observation = relations.get(key)
                if observation is None or member.claim != observation.claim:
                    raise ValueError("mechanism_member_must_match_an_accepted_model_relation")
                if (group.cl_id, group.go_id, group.direction) != (
                        observation.mapping.cl.term_id, observation.mapping.go.term_id, observation.claim.direction):
                    raise ValueError("mechanism_group_must_match_member_ontology_and_direction")
                if key in seen:
                    raise ValueError("duplicate_mechanism_member")
                seen.add(key)
        if seen != set(relations):
            raise ValueError("mechanism_groups_must_preserve_all_accepted_model_relations")
        if self.status == "completed" and any(row.status not in {"mechanisms_found", "inconclusive"} for row in self.models):
            raise ValueError("completed_MP_run_cannot_hide_model_coverage_gaps")
        return self
