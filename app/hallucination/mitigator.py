"""Mitigation utilities for the multi-tier hallucination framework.

Implements Tier 1 model‑level mitigation (temperature scaling, multi‑pass self‑critique),
Tier 2 context‑level mitigation (instruction layering, map‑reduce summarisation), and
Tier 3 data‑level grounding (knowledge injection, deterministic re‑evaluation).
"""
import math
from typing import Any, Dict, List, Optional, Tuple

# ------------------------------------------------------------------------------
# Tier 1: Model‑Level Mitigation
# ------------------------------------------------------------------------------

def apply_temperature_scaling(base_temperature: float, uncertainty: float) -> float:
    """Adjust temperature based on uncertainty (higher uncertainty => higher temperature).

    Args:
        base_temperature: Original LLM sampling temperature.
        uncertainty: Float in [0, 1] where 1 = maximum uncertainty.
    Returns:
        Adjusted temperature bounded between 0.0 and 2.0.
    """
    # Linear interpolation: at 0 uncertainty keep base, at 1 uncertainty add 0.5
    adjusted = base_temperature + 0.5 * uncertainty
    return max(0.0, min(2.0, adjusted))


def multi_pass_self_critique(llm_client, prompt: str, num_passes: int = 3) -> Tuple[str, float]:
    """Run the same prompt multiple times, collect the responses and a consensus confidence.

    Returns a tuple of (selected_response, consensus_confidence).
    """
    responses = []
    for _ in range(num_passes):
        resp = llm_client.generate(prompt=prompt, temperature=0.1)
        responses.append(resp.strip())
    # Simple majority vote; if tie, pick first.
    from collections import Counter
    count = Counter(responses)
    best, freq = count.most_common(1)[0]
    confidence = freq / num_passes
    return best, confidence

# ------------------------------------------------------------------------------
# Tier 2: Context‑Level Mitigation
# ------------------------------------------------------------------------------

def instruction_layering(base_prompt: str, extra_guidance: str) -> str:
    """Inject explicit verification guidance into the LLM prompt.

    The extra guidance asks the model to declare uncertainties and verify numeric claims.
    """
    return f"{base_prompt}\n\n{extra_guidance}"


def map_reduce_context_summarisation(observations: List[Dict[str, Any]], chunk_size: int = 8) -> str:
    """Compress a long observation list using a simple map‑reduce approach.

    The map phase creates overlapping chunks (30% overlap) and keeps the first sentence of each
    observation as a lightweight summary. The reduce phase concatenates these intermediate summaries.
    """
    if not observations:
        return ""
    overlap = max(1, int(chunk_size * 0.3))
    chunks = []
    i = 0
    while i < len(observations):
        chunk = observations[i : i + chunk_size]
        summary = ". ".join(o.get("content", "").split(".")[0] for o in chunk if o.get("content"))
        chunks.append(summary)
        i += chunk_size - overlap
    return " \n".join(chunks)

# ------------------------------------------------------------------------------
# Tier 3: Data‑Level Mitigation
# ------------------------------------------------------------------------------

def inject_knowledge(base_prompt: str, knowledge_snippets: List[str]) -> str:
    """Append retrieved knowledge snippets to the prompt, separated by a delimiter.
    """
    if not knowledge_snippets:
        return base_prompt
    knowledge_block = "\n---\n".join(knowledge_snippets)
    return f"{base_prompt}\n\nRelevant engineering principles:\n{knowledge_block}"


def re_execute_deterministic_tool(tool_name: str, args: Dict[str, Any], tools_registry: Dict[str, Any]) -> Any:
    """Execute a deterministic tool again to obtain a verified result.
    """
    tool_fn = tools_registry.get(tool_name)
    if not tool_fn:
        raise ValueError(f"Tool '{tool_name}' not registered for re‑execution.")
    return tool_fn(**args)

"""Public API of the mitigator module.
The ``apply_mitigation`` helper decides which tier to act upon based on the
``detected_tier`` reported by the auditor and returns any modified prompt or
additional actions the agent should take.
"""

def apply_mitigation(
    detected_tier: Optional[str],
    llm_client: Any,
    base_prompt: str,
    decision_dict: Dict[str, Any],
    state_observations: List[Any],
    allowed_tools: List[str],
    tools_registry: Dict[str, Any],
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Apply targeted mitigation and optionally replace the decision.

    Returns a tuple of (possibly updated_prompt, optional new_decision_dict).
    """
    if detected_tier == "model_tier":
        # Adjust temperature based on reported model uncertainty.
        uncertainty = decision_dict.get("model_uncertainty", 0.5)
        new_temp = apply_temperature_scaling(base_temperature=0.1, uncertainty=uncertainty)
        revised_resp, _ = multi_pass_self_critique(llm_client, base_prompt, num_passes=2)
        return revised_resp, None

    if detected_tier == "context_tier":
        extra = (
            "Please ensure any numeric claim is backed by a tool result. "
            "If uncertain, explicitly state the confidence level."
        )
        new_prompt = instruction_layering(base_prompt, extra)
        return new_prompt, None

    if detected_tier == "data_tier":
        snippets = [obs.get("content", "") for obs in state_observations if obs.get("source") == "knowledge_base"]
        new_prompt = inject_knowledge(base_prompt, snippets)
        return new_prompt, None

    return base_prompt, None
