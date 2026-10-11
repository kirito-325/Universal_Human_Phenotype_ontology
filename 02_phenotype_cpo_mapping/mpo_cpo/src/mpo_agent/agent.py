"""Task-specific adaptation of deep_life_sci/agent.py: root + QuickJS + subagents.

Retains create_deep_agent, CodeInterpreterMiddleware, a PTC allowlist, source-error
capture and restricted leaves. Research ordering is chosen by the root through eval.
Sandbox execution, UI/artifact middleware, trials and figure tools are removed.
"""

from deepagents import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_quickjs import CodeInterpreterMiddleware
from pydantic import create_model

from . import legacy_prompts, prompts
from .contracts import submission_error_feedback
from .schemas import ResearchCompletion, ScientificCompletion
from .upstream.middleware.tool_errors import with_error_capture


def build_agent(session, engine):
    """Assemble the adapted upstream agent against one isolated research session."""
    tools = with_error_capture(session.root_tools())
    compact = engine.compact_submissions and session.workflow.enabled
    return create_deep_agent(
        model=engine.models["research"],
        tools=tools,
        system_prompt=(prompts if compact else legacy_prompts).build_root_prompt(engine.settings),
        subagents=session.subagents(),
        response_format=ToolStrategy(create_model("ResearchCompletion", __base__=ScientificCompletion) if compact
                                    else ResearchCompletion, handle_errors=submission_error_feedback),
        middleware=[
            FilesystemMiddleware(tools=["read_file"]),
            engine.limit,
            CodeInterpreterMiddleware(
                ptc=[t.name for t in tools],
                mode="thread",
                timeout=engine.settings.eval_timeout_seconds,
                max_result_chars=40_000,
                max_ptc_calls=engine.settings.max_ptc_calls,
            ),
        ],
    ).with_config(recursion_limit=engine.settings.recursion_limit)
