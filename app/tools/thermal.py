"""Simplified MVP heuristic thermal evaluator tool."""
from typing import Any, Dict, Optional


def evaluate_thermal(
    design: Optional[Dict[str, Any]] = None,
    climate: Optional[str] = "hot_humid",
    target: float = 0.70,
    **kwargs
) -> Dict[str, Any]:
    """
    Evaluate thermal comfort performance using a deterministic MVP heuristic.
    
    NOTE: Clearly marked as an MVP heuristic for prototyping agent reasoning,
    NOT a certified building physics / CFD thermal simulation.
    
    Args:
        design: ShelterSpec dictionary.
        climate: Climate classification string.
        target: Required minimum comfort score threshold (default 0.70).
        
    Returns:
        Structured thermal evaluation report.
    """
    if not design:
        return {
            "thermal_score": 0.58,
            "target": target,
            "passed": False,
            "error": "No design provided, returned default baseline score."
        }

    roof = design.get("roof", {})
    walls = design.get("walls", {})

    roof_ventilation = float(roof.get("ventilation", 0.5))
    overhang = float(roof.get("overhang", 0.6))
    opening_ratio = float(walls.get("opening_ratio", 0.25))

    # Weighting heuristic for hot-humid passive cooling
    # 1. Roof ventilation effectiveness (stack effect relief)
    vent_factor = min(1.0, roof_ventilation / 0.85)
    
    # 2. Cross-ventilation airflow from wall openings
    opening_factor = min(1.0, opening_ratio / 0.38)
    
    # 3. Solar shading & monsoon protection from eaves
    shading_factor = min(1.0, overhang / 0.95)

    # Heuristic combined score (0.0 to 1.0)
    raw_score = 0.40 * vent_factor + 0.35 * opening_factor + 0.25 * shading_factor
    # Scaling to achieve ~0.58 for V1 defaults and ~0.75+ for V2 improved specs
    thermal_score = round(raw_score * 0.88, 2)

    passed = thermal_score >= target

    return {
        "thermal_score": thermal_score,
        "target": target,
        "passed": passed,
        "score": thermal_score,
        "heuristic_details": {
            "roof_ventilation_factor": round(vent_factor, 2),
            "opening_airflow_factor": round(opening_factor, 2),
            "solar_shading_factor": round(shading_factor, 2),
            "climate_assumed": climate or "hot_humid"
        },
        "diagnosis": (
            "Thermal performance meets comfort criteria."
            if passed
            else f"Thermal score ({thermal_score}) is below target threshold ({target}). Increase roof ventilation, wall opening ratio, or roof overhang."
        )
    }
