"""Deterministic evaluation aggregation and critic components."""
from typing import Any, Dict, List, Optional
from app.tools.cost import evaluate_cost
from app.tools.thermal import evaluate_thermal
from app.tools.structure import evaluate_structure


def critique(evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic critic identifying specific engineering and performance issues.
    
    Args:
        evaluation: Combined evaluation dictionary containing cost, thermal, structure.
        
    Returns:
        Structured critique dictionary with problem list, suggestions, and pass status.
    """
    problems: List[str] = []
    suggestions: List[str] = []

    thermal_eval = evaluation.get("thermal", {})
    if not thermal_eval.get("passed", False):
        score = thermal_eval.get("thermal_score", thermal_eval.get("score", 0.0))
        target = thermal_eval.get("target", 0.70)
        problems.append(
            f"Thermal comfort score ({score}) is below required target threshold ({target})."
        )
        suggestions.append(
            "Increase roof ridge ventilation (ventilation >= 0.8), wall opening ratio (>= 0.35), and roof overhang (>= 0.9m) to enhance airflow and shading."
        )

    cost_eval = evaluation.get("cost", {})
    if not cost_eval.get("passed", False):
        total = cost_eval.get("total_cost", 0.0)
        budget = cost_eval.get("budget", 0.0)
        problems.append(
            f"Design exceeds budget: ₹{total:,.2f} exceeds allocated ₹{budget:,.2f}."
        )
        suggestions.append(
            "Optimize structural dimensions or substitute premium envelope materials with locally sourced woven bamboo."
        )

    struct_eval = evaluation.get("structure", {})
    if not struct_eval.get("passed", False):
        violations = struct_eval.get("violations", ["Structural constraint failed."])
        problems.extend(violations)
        suggestions.append(
            "Reduce column spacing below 2.2m or reduce clear span width."
        )

    all_passed = (
        thermal_eval.get("passed", False)
        and cost_eval.get("passed", False)
        and struct_eval.get("passed", False)
    )

    return {
        "all_passed": all_passed,
        "problems": problems,
        "suggestions": suggestions,
        "summary": "All performance criteria satisfied." if all_passed else f"{len(problems)} issue(s) detected requiring design iteration."
    }


def evaluate_all(
    design: Optional[Dict[str, Any]] = None,
    budget: Optional[float] = None,
    climate: Optional[str] = "hot_humid",
    **kwargs
) -> Dict[str, Any]:
    """
    Run full multi-disciplinary evaluation pipeline (Cost, Thermal, Structure) and Critic.
    
    Args:
        design: ShelterSpec dictionary.
        budget: Maximum budget constraint.
        climate: Climate classification.
        
    Returns:
        Comprehensive multi-criteria evaluation dictionary with critic output.
    """
    cost_res = evaluate_cost(design, budget=budget)
    thermal_res = evaluate_thermal(design, climate=climate)
    struct_res = evaluate_structure(design)

    combined_eval = {
        "cost": cost_res,
        "thermal": thermal_res,
        "structure": struct_res,
        "overall_score": round((cost_res.get("score", 0.9) + thermal_res.get("score", 0.5) + struct_res.get("score", 0.8)) / 3.0, 2)
    }

    critic_res = critique(combined_eval)
    combined_eval["passed"] = critic_res["all_passed"]
    combined_eval["critique"] = critic_res

    return combined_eval
