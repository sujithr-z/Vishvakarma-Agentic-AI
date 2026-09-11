"""Multi-Tier Hallucination Detection Module based on CIBC Research Framework.

Implements:
- Tier 1: Model Intrinsic Detection (Token/Semantic Entropy, Sampling Variance, Self-Declared Confidence)
- Tier 2: Context Alignment Detection (RACE - Reasoning & Answer Consistency Evaluation, Instruction Compliance)
- Tier 3: Data Grounding Detection (Deterministic Cross-Verification against NBC 2016, TERI-2021, TARU-2015, Tool Observations)
"""
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import Counter


def calculate_probabilistic_entropy(probabilities: List[float]) -> float:
    """
    Compute probabilistic / discrete distribution entropy:
    H = - sum(p * log2(p)) for p in probabilities (where p > 0).
    """
    if not probabilities:
        return 0.0
    
    total = sum(probabilities)
    if total <= 0:
        return 0.0
    
    normalized = [p / total for p in probabilities if p > 0]
    entropy = -sum(p * math.log2(p) for p in normalized)
    return round(entropy, 4)


def cluster_semantic_responses(responses: List[str]) -> List[List[str]]:
    """
    Lightweight rule- and keyword-based semantic clustering for responses.
    Groups semantically equivalent candidate answers.
    """
    if not responses:
        return []
    
    def normalize_text(t: str) -> str:
        t = t.lower().strip()
        t = re.sub(r'[^\w\s\d]', '', t)
        # normalize whitespace
        t = " ".join(t.split())
        return t

    clusters: List[List[str]] = []
    cluster_reprs: List[Set[str]] = []

    for resp in responses:
        norm = normalize_text(resp)
        words = set(norm.split())
        matched = False
        for idx, rep in enumerate(cluster_reprs):
            if not words and not rep:
                clusters[idx].append(resp)
                matched = True
                break
            # Jaccard overlap on content words
            intersection = len(words & rep)
            union = len(words | rep)
            similarity = intersection / union if union > 0 else 0.0
            if similarity >= 0.60:
                clusters[idx].append(resp)
                matched = True
                break
        
        if not matched:
            clusters.append([resp])
            cluster_reprs.append(words)

    return clusters


def estimate_semantic_entropy(responses: List[str]) -> Dict[str, Any]:
    """
    Calculate semantic entropy Hs = - sum(p(c_k) * log2(p(c_k))) over semantic clusters.
    Lower entropy indicates high semantic consistency; high entropy indicates divergence/hallucination risk.
    """
    if not responses:
        return {"semantic_entropy": 0.0, "clusters_count": 0, "certainty": 1.0}
    
    clusters = cluster_semantic_responses(responses)
    n = len(responses)
    proportions = [len(c) / n for c in clusters]
    
    entropy = calculate_probabilistic_entropy(proportions)
    # Normalized certainty between 0.0 (high entropy/ambiguity) and 1.0 (consensus)
    max_possible_entropy = math.log2(n) if n > 1 else 1.0
    certainty = max(0.0, min(1.0, 1.0 - (entropy / max_possible_entropy if max_possible_entropy > 0 else 0.0)))
    
    return {
        "semantic_entropy": entropy,
        "clusters_count": len(clusters),
        "total_samples": n,
        "cluster_sizes": [len(c) for c in clusters],
        "certainty": round(certainty, 3)
    }


def extract_self_declared_confidence(text: str) -> Optional[float]:
    """
    Extract self-declared confidence score from model response text or metadata.
    e.g. 'Confidence: 0.85', 'Confidence: 85%', 'Confidence Score: 0.9'.
    """
    if not text:
        return None
    
    match = re.search(r'(?:confidence|certainty)(?:\s+score)?\s*[:=]\s*(\d+(?:\.\d+)?|\.\d+)(%)?', text, re.IGNORECASE)
    if match:
        val_str, is_pct = match.group(1), bool(match.group(2))
        try:
            val = float(val_str)
            if is_pct or val > 1.0:
                val = val / 100.0
            return max(0.0, min(1.0, val))
        except ValueError:
            pass
    return None


# ------------------------------------------------------------------------------
# Tier 2: Context Alignment & RACE (Reasoning and Answer Consistency Evaluation)
# ------------------------------------------------------------------------------

def evaluate_race(
    reasoning: str,
    answer: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    RACE Framework (Reasoning and Answer Consistency Evaluation - Section 2.5).
    Jointly evaluates answer plausibility, reasoning consistency, and mutual information I(R, A | Q).
    Detects 'right answer for the wrong reasons' or logical disconnects.
    """
    context = context or {}
    issues: List[str] = []
    
    reasoning_clean = (reasoning or "").strip()
    answer_clean = (answer or "").strip()
    
    # 1. Uncertainty in reasoning trace H(R|Q)
    h_r = 0.1
    if not reasoning_clean:
        h_r = 0.8
        issues.append("Missing or unarticulated reasoning trace (low transparency).")
    elif len(reasoning_clean.split()) < 4:
        h_r = 0.5
        issues.append("Underspecified / trivial reasoning trace.")

    # 2. Check logical entailment and mutual information I(R, A | Q)
    i_ra = 0.9
    
    # Check for contradictions between reasoning and answer (e.g. reasoning says 'exceeds budget' but answer says 'approved')
    pos_words = {"pass", "passed", "compliant", "approved", "within", "satisfies", "feasible", "comfortable"}
    neg_words = {"fail", "failed", "exceeds", "violation", "unfeasible", "uncomfortable", "over budget", "inadequate"}
    
    r_lower = reasoning_clean.lower()
    a_lower = answer_clean.lower()
    
    r_has_neg = any(w in r_lower for w in neg_words)
    a_has_pos = any(w in a_lower for w in pos_words)
    r_has_pos = any(w in r_lower for w in pos_words)
    a_has_neg = any(w in a_lower for w in neg_words)
    
    if (r_has_neg and not r_has_pos) and (a_has_pos and not a_has_neg):
        i_ra = 0.1
        issues.append("RACE Disconnect: Reasoning indicates failure/violations but final conclusion claims compliance.")
    elif (r_has_pos and not r_has_neg) and (a_has_neg and not a_has_pos):
        i_ra = 0.2
        issues.append("RACE Disconnect: Reasoning indicates compliance but final conclusion claims failure.")
        
    # Check for false regulatory standard citations
    fake_standards = ["nbc 2025", "teri-2099", "iso 99999", "taru-1980", "nbc 1950"]
    for fs in fake_standards:
        if fs in r_lower or fs in a_lower:
            i_ra = min(i_ra, 0.2)
            issues.append(f"RACE Regulatory Hallucination: Cited non-existent or fabricated standard '{fs}'.")

    # Joint uncertainty score: H(R, A | Q) = H(R|Q) + H(A|Q) - I(R, A|Q)
    h_a = 0.1 if answer_clean else 0.9
    joint_uncertainty = max(0.0, min(1.0, h_r + h_a - (i_ra * 0.5)))
    consistency_score = round(max(0.0, min(1.0, 1.0 - joint_uncertainty)), 3)
    
    return {
        "race_consistency_score": consistency_score,
        "reasoning_uncertainty": h_r,
        "answer_uncertainty": h_a,
        "mutual_support": round(i_ra, 3),
        "is_consistent": consistency_score >= 0.65,
        "issues": issues
    }


def check_instruction_compliance(decision_dict: Dict[str, Any], allowed_tools: List[str]) -> Dict[str, Any]:
    """
    Tier 2 Context Check: Verifies strict adherence to instruction constraints and tool naming.
    """
    violations: List[str] = []
    d_type = decision_dict.get("type")
    
    if d_type not in ["tool", "final"]:
        violations.append(f"Invalid decision type '{d_type}'. Must be 'tool' or 'final'.")
        
    if d_type == "tool":
        tool_name = decision_dict.get("tool_name")
        if not tool_name:
            violations.append("Tool decision missing 'tool_name'.")
        elif tool_name not in allowed_tools:
            violations.append(f"Hallucinated tool name '{tool_name}'. Allowed: {allowed_tools}")
            
    return {
        "compliant": len(violations) == 0,
        "violations": violations
    }


# ------------------------------------------------------------------------------
# Tier 3: Data Tier Grounding & Deterministic Consistency
# ------------------------------------------------------------------------------

def extract_numbers_with_context(text: str) -> List[Tuple[float, str]]:
    """Extract numbers and their trailing units or keywords."""
    pattern = r'(?:₹|rs\.?|inr)?\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*(?:₹|inr|rs|deg|°c|c|m²|sqm|sq\.m|m|people|persons|%)?'
    results: List[Tuple[float, str]] = []
    for match in re.finditer(pattern, text, re.IGNORECASE):
        num_str = match.group(1).replace(",", "")
        try:
            val = float(num_str)
            full_match = match.group(0).strip()
            results.append((val, full_match))
        except ValueError:
            pass
    return results


def verify_factual_consistency(
    claim_text: str,
    state_observations: List[Any],
    current_design: Optional[Dict[str, Any]] = None,
    evaluation: Optional[Dict[str, Any]] = None,
    cost_impact: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Tier 3 Data-Tier Grounding:
    Cross-checks claims in model output against deterministic tool observations and verified calculations.
    Flags fabricated numbers, ungrounded costs, or false thermal metrics.
    """
    issues: List[str] = []
    text_lower = claim_text.lower()
    
    # Collect all verified ground-truth values from state
    verified_numbers: Set[float] = set()
    
    # 1. From observations
    for obs in state_observations:
        obs_str = obs.content if hasattr(obs, "content") else str(obs)
        for num, _ in extract_numbers_with_context(obs_str):
            verified_numbers.add(round(num, 2))
            verified_numbers.add(round(num, 0))
            
    # 2. From design
    if current_design:
        dim = current_design.get("dimensions", {})
        for k in ["length_m", "width_m", "height_m", "floor_area_sqm"]:
            if k in dim:
                val = float(dim[k])
                verified_numbers.add(round(val, 2))
                verified_numbers.add(round(val, 0))
                
    # 3. From evaluations
    if evaluation:
        cost = evaluation.get("cost", {})
        if "total_cost" in cost:
            tc = float(cost["total_cost"])
            verified_numbers.add(round(tc, 2))
            verified_numbers.add(round(tc, 0))
        thermal = evaluation.get("thermal", {})
        if "estimated_indoor_temp" in thermal:
            ti = float(thermal["estimated_indoor_temp"])
            verified_numbers.add(round(ti, 2))
            verified_numbers.add(round(ti, 0))
        if "upper_90_limit" in thermal:
            u90 = float(thermal["upper_90_limit"])
            verified_numbers.add(round(u90, 2))
            verified_numbers.add(round(u90, 0))

    if cost_impact:
        tot = cost_impact.get("total_inr")
        if tot is not None:
            verified_numbers.add(round(float(tot), 2))
            verified_numbers.add(round(float(tot), 0))

    # Common valid constants (e.g. 100%, 0, 1, 2, 3, 4, 5, 2014, 2015, 2016, 2021)
    whitelisted = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 100.0, 2014.0, 2015.0, 2016.0, 2021.0}
    verified_numbers.update(whitelisted)

    # Check for specific ungrounded cost fabrications
    cost_matches = re.findall(r'(?:₹|inr|rs\.?)\s*([\d,]+(?:\.\d+)?)', claim_text, re.IGNORECASE)
    for cm in cost_matches:
        val = float(cm.replace(",", ""))
        if val > 100 and round(val, 0) not in verified_numbers and round(val, 2) not in verified_numbers:
            # Check if within 5% of any verified number
            is_close = any(abs(val - v) / v < 0.05 for v in verified_numbers if v > 0)
            if not is_close:
                issues.append(f"Ungrounded financial claim: ₹{val:,.2f} has no corresponding evidence in state observations.")

    # Check for ungrounded temperature claims
    temp_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:°c|deg c|degrees celsius)', claim_text, re.IGNORECASE)
    for tm in temp_matches:
        val = float(tm)
        if round(val, 1) not in verified_numbers and round(val, 0) not in verified_numbers:
            is_close = any(abs(val - v) < 0.5 for v in verified_numbers if 10 <= v <= 60)
            if not is_close and val > 10:
                issues.append(f"Ungrounded temperature claim: {val}°C does not match verified thermal calculation.")

    grounding_score = max(0.0, 1.0 - (0.35 * len(issues)))
    return {
        "grounded": len(issues) == 0,
        "grounding_score": round(grounding_score, 3),
        "issues": issues,
        "verified_numbers_checked": len(verified_numbers)
    }


# ------------------------------------------------------------------------------
# Unified Multi-Tier Hallucination Auditor
# ------------------------------------------------------------------------------

def audit_hallucination_risk(
    decision_type: str,
    decision_dict: Dict[str, Any],
    content_or_reason: str,
    state_observations: List[Any],
    allowed_tools: List[str],
    current_design: Optional[Dict[str, Any]] = None,
    evaluation: Optional[Dict[str, Any]] = None,
    cost_impact: Optional[Dict[str, Any]] = None,
    candidate_samples: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Comprehensive multi-tier hallucination audit aggregating Model, Context, and Data tiers.
    """
    tier_reports: Dict[str, Any] = {}
    all_issues: List[str] = []
    
    # 1. Tier 1: Model Intrinsic Check
    self_conf = extract_self_declared_confidence(content_or_reason)
    sem_ent = estimate_semantic_entropy(candidate_samples or [content_or_reason])
    t1_uncertainty = 1.0 - sem_ent["certainty"]
    if self_conf is not None:
        t1_uncertainty = max(t1_uncertainty, 1.0 - self_conf)
        
    tier_reports["model_tier"] = {
        "self_declared_confidence": self_conf,
        "semantic_entropy": sem_ent["semantic_entropy"],
        "model_uncertainty": round(t1_uncertainty, 3),
        "passed": t1_uncertainty <= 0.45
    }
    if t1_uncertainty > 0.45:
        all_issues.append(f"Tier 1 (Model): High uncertainty detected (score {t1_uncertainty:.2f}).")

    # 2. Tier 2: Context Alignment Check
    inst_check = check_instruction_compliance(decision_dict, allowed_tools)
    race_check = evaluate_race(
        reasoning=decision_dict.get("reason", ""),
        answer=decision_dict.get("content", content_or_reason)
    )
    t2_passed = inst_check["compliant"] and race_check["is_consistent"]
    tier_reports["context_tier"] = {
        "instruction_compliance": inst_check,
        "race": race_check,
        "passed": t2_passed
    }
    if not inst_check["compliant"]:
        all_issues.extend([f"Tier 2 (Context): {v}" for v in inst_check["violations"]])
    if not race_check["is_consistent"]:
        all_issues.extend([f"Tier 2 (Context RACE): {i}" for i in race_check["issues"]])

    # 3. Tier 3: Data Tier Grounding Check
    claim_text = decision_dict.get("content") or content_or_reason
    data_check = verify_factual_consistency(
        claim_text=claim_text,
        state_observations=state_observations,
        current_design=current_design,
        evaluation=evaluation,
        cost_impact=cost_impact
    )
    tier_reports["data_tier"] = {
        "factual_consistency": data_check,
        "passed": data_check["grounded"]
    }
    if not data_check["grounded"]:
        all_issues.extend([f"Tier 3 (Data Grounding): {i}" for i in data_check["issues"]])

    # Root cause classification: determine primary failure tier
    detected_tier = None
    if not tier_reports["data_tier"]["passed"]:
        detected_tier = "data_tier"
    elif not tier_reports["context_tier"]["passed"]:
        detected_tier = "context_tier"
    elif not tier_reports["model_tier"]["passed"]:
        detected_tier = "model_tier"

    # Overall risk calculation
    risk_score = 0.0
    if not tier_reports["data_tier"]["passed"]:
        risk_score += 0.45
    if not tier_reports["context_tier"]["passed"]:
        risk_score += 0.35
    if not tier_reports["model_tier"]["passed"]:
        risk_score += 0.20
    risk_score = min(1.0, round(risk_score, 3))

    return {
        "overall_risk_score": risk_score,
        "passed": risk_score < 0.30,
        "detected_tier": detected_tier,
        "tier_breakdown": tier_reports,
        "issues": all_issues,
        "needs_mitigation": risk_score >= 0.30
    }
