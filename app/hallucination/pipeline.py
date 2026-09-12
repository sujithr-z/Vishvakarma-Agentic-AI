"""Tiered Hallucination Manager integrating detection and mitigation.

Based on the CIBC three-tier framework (arXiv:2601.09929v1):
  - Tier 1 (Model-level)  : entropy / self-declared confidence scoring
  - Tier 2 (Context-level): RACE multi-sample consistency check
  - Tier 3 (Data-level)   : fact-grounding against state observations & tools

Usage within the agent loop::

    manager = TieredHallucinationManager(llm_client, list(TOOLS.keys()))
    decision_dict, mitigated = manager.assess_and_mitigate(state, decision_dict)
"""
from typing import Any, Dict, List, Optional, Tuple

from app.hallucination.detector import audit_hallucination_risk


class TieredHallucinationManager:
    """Orchestrates the three-tier hallucination audit and targeted mitigation.

    Parameters
    ----------
    llm_client : Any
        The LLM client used for re-generation when mitigation requires it.
    allowed_tools : List[str]
        Registry of valid deterministic tool names exposed to the agent.
    """

    def __init__(self, llm_client: Any, allowed_tools: List[str]):
        self.llm = llm_client
        self.allowed_tools = allowed_tools

    def assess_and_mitigate(
        self,
        state: Any,
        decision_dict: Dict[str, Any],
        candidate_samples: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, Any], bool]:
        """Run hallucination audit and targeted mitigation on the decision dict.

        Returns a tuple ``(updated_decision_dict, mitigation_performed)``.
        Tool decisions proceed because their results become new evidence. A
        final answer that fails data grounding is redirected to the curated KB.
        """
        observations = []
        for obs in getattr(state, "observations", []) or []:
            observations.append(obs.model_dump() if hasattr(obs, "model_dump") else obs)
        audit = audit_hallucination_risk(
            decision_type=decision_dict.get("type", ""),
            decision_dict=decision_dict,
            content_or_reason=decision_dict.get("content") or decision_dict.get("reason", ""),
            state_observations=observations,
            allowed_tools=self.allowed_tools,
            current_design=getattr(state, "current_design", None),
            evaluation=getattr(state, "evaluation", None),
            cost_impact=getattr(state, "cost_impact", None),
            candidate_samples=candidate_samples,
        )
        decision_dict["grounding_audit"] = audit
        data_failed = not audit["tier_breakdown"]["data_tier"]["passed"]
        rag_required = (
            decision_dict.get("type") == "final"
            and getattr(state, "intent", "") == "knowledge_query"
            and not any(o.get("source") == "knowledge_base" for o in observations if isinstance(o, dict))
        )
        uncertainty_disclosed = any(
            marker in decision_dict.get("content", "").lower()
            for marker in ("unavailable", "not available", "unknown", "cannot verify", "insufficient data")
        )
        if decision_dict.get("type") == "final" and (rag_required or (data_failed and not uncertainty_disclosed and getattr(state, "intent", "") not in {"general_query", "greeting"})):
            already_searched = any(o.get("source") == "knowledge_base" for o in observations if isinstance(o, dict))
            if not already_searched and "search_knowledge" in self.allowed_tools:
                return {
                    "type": "tool",
                    "tool_name": "search_knowledge",
                    "arguments": {"query": getattr(state, "user_query", "")},
                    "reason": "Draft contains unsupported claims; retrieve curated evidence before answering.",
                }, True
            decision_dict["content"] = (
                "Evidence limitation: the available tools and knowledge base do not fully verify this answer.\n"
                + decision_dict.get("content", "")
            )
            return decision_dict, True
        return decision_dict, bool(audit.get("needs_mitigation"))
