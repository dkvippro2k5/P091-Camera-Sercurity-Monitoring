"""LangGraph orchestration for the asynchronous incident-review workflow.

The graph runs after the rule engine has already emitted an alert. It deliberately does
not sit in the camera frame path: detection, tracking, and temporal rules remain direct
and deterministic. Graph state is in-memory because the alert and evidence are already
persisted by MonitoringService; durable checkpointing can be added when retry/resume
semantics are required in production.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.models.camera_schemas import Alert, AlertEvidenceFrame
from src.services.vlm import TriageVerdict


class IncidentWorkflowState(TypedDict, total=False):
    alert: Alert
    frames: list[bytes]
    pre_count: int
    evidence_frames: list[AlertEvidenceFrame]
    examples: list[str]
    verdict: TriageVerdict
    error: str


WaitForPostRoll = Callable[[Alert], None]
CollectEvidence = Callable[[Alert], tuple[list[bytes], int, list[AlertEvidenceFrame], list[str]]]
AssessEvidence = Callable[[Alert, list[bytes], int, list[str]], TriageVerdict]
PersistVerdict = Callable[[Alert, TriageVerdict, list[AlertEvidenceFrame]], None]
PersistFailure = Callable[[Alert, str], None]


def build_incident_workflow(
    *,
    wait_for_post_roll: WaitForPostRoll,
    collect_evidence: CollectEvidence,
    assess_evidence: AssessEvidence,
    persist_verdict: PersistVerdict,
    persist_failure: PersistFailure,
):
    """Compile the review workflow with service-owned side-effect callbacks."""

    def wait_node(state: IncidentWorkflowState) -> dict[str, Any]:
        wait_for_post_roll(state["alert"])
        return {}

    def collect_node(state: IncidentWorkflowState) -> dict[str, Any]:
        frames, pre_count, evidence_frames, examples = collect_evidence(state["alert"])
        return {
            "frames": frames,
            "pre_count": pre_count,
            "evidence_frames": evidence_frames,
            "examples": examples,
        }

    def assess_node(state: IncidentWorkflowState) -> dict[str, Any]:
        verdict = assess_evidence(
            state["alert"],
            state.get("frames", []),
            state.get("pre_count", 0),
            state.get("examples", []),
        )
        return {"verdict": verdict}

    def persist_node(state: IncidentWorkflowState) -> dict[str, Any]:
        persist_verdict(
            state["alert"],
            state["verdict"],
            state.get("evidence_frames", []),
        )
        return {}

    def fail_node(state: IncidentWorkflowState) -> dict[str, Any]:
        error = state.get("error", "incident workflow failed")
        persist_failure(state["alert"], error)
        return {}

    def route_after_wait(state: IncidentWorkflowState) -> str:
        return "failed" if state.get("error") else "collect"

    def route_after_collect(state: IncidentWorkflowState) -> str:
        return "failed" if state.get("error") else "assess"

    def route_after_assess(state: IncidentWorkflowState) -> str:
        return "failed" if state.get("error") else "persist"

    def route_after_persist(state: IncidentWorkflowState) -> str:
        return "failed" if state.get("error") else "done"

    # LangGraph does not automatically convert node exceptions into state. The wrapper
    # nodes make failures explicit so an upstream VLM/evidence error always reaches the
    # alert's uncertain/failure persistence path.
    def guarded(node: Callable[[IncidentWorkflowState], dict[str, Any]]):
        def run(state: IncidentWorkflowState) -> dict[str, Any]:
            try:
                return node(state)
            except Exception as exc:  # noqa: BLE001 - workflow must preserve the alert
                return {"error": f"{type(exc).__name__}: {exc}"}

        return run

    graph = StateGraph(IncidentWorkflowState)
    graph.add_node("wait_for_post_roll", guarded(wait_node))
    graph.add_node("collect_evidence", guarded(collect_node))
    graph.add_node("assess_evidence", guarded(assess_node))
    graph.add_node("persist_verdict", guarded(persist_node))
    graph.add_node("persist_failure", fail_node)
    graph.set_entry_point("wait_for_post_roll")
    graph.add_conditional_edges(
        "wait_for_post_roll",
        route_after_wait,
        {"collect": "collect_evidence", "failed": "persist_failure"},
    )
    graph.add_conditional_edges(
        "collect_evidence",
        route_after_collect,
        {"assess": "assess_evidence", "failed": "persist_failure"},
    )
    graph.add_conditional_edges(
        "assess_evidence",
        route_after_assess,
        {"persist": "persist_verdict", "failed": "persist_failure"},
    )
    graph.add_conditional_edges(
        "persist_verdict",
        route_after_persist,
        {"done": END, "failed": "persist_failure"},
    )
    graph.add_edge("persist_failure", END)
    return graph.compile()
