import itertools

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from pydantic import Field


class ScriptedModel(FakeMessagesListChatModel):
    model_name: str = "mpo-agent-test"
    observed_tools: list = Field(default_factory=list)

    def _get_ls_params(self, **kwargs):
        return {"ls_provider": "openai", "ls_model_name": self.model_name, "ls_model_type": "chat"}

    def bind_tools(self, tools, **kwargs):
        self.observed_tools.extend(t.name if hasattr(t, "name") else t.get("function", t).get("name") for t in tools)
        return self


calls = itertools.count()


def response(name, args):
    return AIMessage("", tool_calls=[{"id": "test_call_" + str(next(calls)), "name": name, "args": args}])


def assessment(ref="E1"):
    return {"decision": "retain", "rationale": "Synthetic model-scoped scientific reasoning.",
        "claim": {"cell": {"cell_text": "neurons", "explanation": "Synthetic mouse cell reasoning.", "evidence_refs": [ref]},
            "abnormality": {"process_text": "apoptosis", "quality_text": "increased apoptosis", "direction": "increased",
                "explanation": "Synthetic model process reasoning.", "evidence_refs": [ref]},
            "phenotype_link": {"explanation": "Synthetic mouse phenotype applicability.", "evidence_refs": []},
            "assumptions": ["Synthetic model-specific applicability assumption."]}}


def judgments():
    return {key: {"status": "uncertain", "reason": "Plausible synthetic prediction without experimental confirmation."}
            for key in ("cell_identity", "cell_abnormality", "phenotype_link", "ontology_mapping")}
