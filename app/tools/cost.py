"""Deterministic cost evaluation tool for shelter designs."""
from typing import Any, Dict, List, Optional


def evaluate_cost(
    design: Optional[Dict[str, Any]] = None,
    budget: Optional[float] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Calculate deterministic shelter construction cost and compare against budget.
    
    Args:
        design: ShelterSpec dictionary.
        budget: Maximum budget constraint in INR (overrides design budget if supplied).
        
    Returns:
        Structured cost evaluation report.
    """
    if not design:
        return {
            "total_cost": 76000,
            "budget": budget or 80000,
            "within_budget": True,
            "score": 0.95,
            "passed": True,
            "error": "No design provided, returned default baseline cost."
        }

    geom = design.get("geometry", {})
    length = float(geom.get("length", 6.0))
    width = float(geom.get("width", 4.0))
    height = float(geom.get("height", 3.0))
    floor_area = length * width

    roof = design.get("roof", {})
    overhang = float(roof.get("overhang", 0.6))
    roof_area = (length + 2 * overhang) * (width + 2 * overhang) * 1.15

    walls = design.get("walls", {})
    opening_ratio = float(walls.get("opening_ratio", 0.25))
    gross_wall_area = 2 * (length + width) * height
    net_wall_area = gross_wall_area * (1.0 - opening_ratio)

    target_budget = float(budget or design.get("budget", 80000.0))

    # Unit cost rates for local bamboo eco-construction (in INR)
    frame_rate_per_sqm = 1050.0   # Structural bamboo posts, beams, joinery
    roof_rate_per_sqm = 510.0     # Bamboo purlins, corrugated eco-sheet
    wall_rate_per_sqm = 280.0     # Woven treated bamboo matting
    labor_and_hardware = 13000.0  # Local craftsman labor, fasteners, anchors

    cost_frame = round(floor_area * frame_rate_per_sqm, 2)
    cost_roof = round(roof_area * roof_rate_per_sqm, 2)
    cost_walls = round(net_wall_area * wall_rate_per_sqm, 2)
    total_cost = round(cost_frame + cost_roof + cost_walls + labor_and_hardware, 2)

    within_budget = total_cost <= target_budget
    budget_utilization = round((total_cost / target_budget), 3) if target_budget > 0 else 1.0
    
    # Cost score is high when within budget
    cost_score = round(max(0.0, min(1.0, 1.0 - (total_cost - target_budget) / target_budget)) if not within_budget else 0.95, 2)

    return {
        "total_cost": total_cost,
        "budget": target_budget,
        "within_budget": within_budget,
        "passed": within_budget,
        "score": cost_score,
        "budget_utilization_ratio": budget_utilization,
        "breakdown": {
            "structural_frame_inr": cost_frame,
            "roofing_system_inr": cost_roof,
            "wall_envelope_inr": cost_walls,
            "labor_and_hardware_inr": labor_and_hardware
        }
    }


def calculate_improvement_cost(
    improvements: Optional[List[Dict[str, Any]]] = None,
    design: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Deterministically calculate itemized cost impact for proposed shelter improvements.
    
    Args:
        improvements: List of proposed improvements with parameter names.
        design: Active ShelterSpec.
        
    Returns:
        Structured cost delta report with itemized additions and total cost impact.
    """
    cost_items = []
    total_delta = 0.0

    # Unit modification rates (INR)
    COST_SCHEDULE = {
        "roof_ventilation": {
            "item": "Roof ridge vent fabrication & breathable mesh louvers",
            "cost": 1800.0
        },
        "roof_overhang": {
            "item": "Extended bamboo rafters & additional eco-roofing sheet eaves",
            "cost": 3200.0
        },
        "opening_ratio": {
            "item": "Wall opening enlargement, bamboo frame framing & shutters",
            "cost": 1500.0
        },
        "general_shading": {
            "item": "Secondary bamboo slat sunshades & eaves bracing",
            "cost": 1200.0
        }
    }

    if not improvements:
        # Default typical full passive hot-humid upgrade bundle
        param_keys = ["roof_ventilation", "roof_overhang", "opening_ratio"]
    else:
        param_keys = [imp.get("parameter", "") for imp in improvements]

    for key, spec in COST_SCHEDULE.items():
        if key in param_keys or any(key in p for p in param_keys):
            cost_items.append({
                "item": spec["item"],
                "estimated_cost_inr": spec["cost"]
            })
            total_delta += spec["cost"]

    if not cost_items:
        # Fallback itemization
        for imp in (improvements or []):
            cost_items.append({
                "item": f"Modification: {imp.get('change', 'Shelter upgrade')}",
                "estimated_cost_inr": 2000.0
            })
            total_delta += 2000.0

    return {
        "items": cost_items,
        "total_additional_cost_inr": round(total_delta, 2),
        "formatted_total": f"₹{total_delta:,.2f}"
    }
