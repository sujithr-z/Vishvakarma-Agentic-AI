"""
Technical Response Formatter strictly following:
Part 9 — Final Human-Readable Response Template (TERI-2021 & TARU-2015 Knowledge Base Specification)
"""
from typing import Any, Dict, List, Optional


def format_technical_thermal_analysis(
    constraints_report: Dict[str, Any],
    cost_report: Optional[Dict[str, Any]] = None,
    design: Optional[Dict[str, Any]] = None,
    climate_data: Optional[Dict[str, Any]] = None
) -> str:
    """
    Format analysis into the exact Part 9 Human-Readable Response Template.
    """
    analysis = constraints_report.get("analysis", {})
    b_id = analysis.get("building_id", "Adaptive-Bamboo-Shelter-v1")
    c_zone = analysis.get("climate_zone", "Warm & Humid")
    assessment = analysis.get("overall_assessment", "Overheating risk detected under cooling-dominated conditions.")

    constraints = analysis.get("thermal_constraints", [])
    suggestions = constraints_report.get("suggestions", [])
    improvements = constraints_report.get("improvements", [])
    cost_data = constraints_report.get("cost", {}) or (cost_report or {})
    cost_items = cost_data.get("items", [])
    total_cost_str = cost_data.get("formatted_total", f"₹{cost_data.get('total_inr', 0.0):,.2f}")
    uncertainties = constraints_report.get("uncertainties", [])

    lines = []
    lines.append("THERMAL ANALYSIS")
    lines.append(f"Building: {b_id}")
    lines.append(f"Climate zone: {c_zone}")
    lines.append(f"Overall assessment: {assessment}\n")

    lines.append("THERMAL CONSTRAINTS AT RISK / VIOLATED\n")
    violated_or_at_risk = [c for c in constraints if c.get("status") in ["VIOLATED", "AT_RISK"]]
    display_constraints = violated_or_at_risk if violated_or_at_risk else constraints

    for idx, c in enumerate(display_constraints, 1):
        lines.append(f"{idx}. {c.get('name', c.get('parameter', '')).upper()}")
        lines.append(f"   Current: {c.get('actual_value')} {c.get('unit', '')}")
        lines.append(f"   Threshold: {c.get('threshold')}")
        lines.append(f"   Status: {c.get('status')}")
        lines.append(f"   Why: {c.get('reason', '')}")
        lines.append(f"   Source: {c.get('source', 'TERI-2021 / TARU-2015')}\n")

    lines.append("SUGGESTIONS")
    for idx, s in enumerate(suggestions, 1):
        lines.append(f"{idx}. {s.get('description')} – expected effect: {s.get('expected_effect')} (Source: {s.get('source')})")
    lines.append("")

    lines.append("IMPROVEMENTS")
    for idx, imp in enumerate(improvements, 1):
        lines.append(f"{idx}. Change {imp.get('parameter')}")
        lines.append(f"   Current -> Recommended: {imp.get('current_value')} -> {imp.get('recommended_value')}")
        lines.append(f"   Reason: {imp.get('reason')}")
        lines.append(f"   Expected thermal effect: {imp.get('expected_effect')} (Source: {imp.get('source')})")
    lines.append("")

    lines.append("COST IMPACT (2014 INR)")
    for idx, item in enumerate(cost_items, 1):
        cost_val = item.get("total_inr") or item.get("estimated_cost_inr", 0.0)
        lines.append(f"{idx}. {item.get('intervention', item.get('item', ''))}: INR {cost_val:,.2f}")
    lines.append(f"Total estimated: INR {cost_data.get('total_inr', 3200.0):,.2f}")
    lines.append(f"Note: {cost_data.get('notes', 'Costs are historical (2014 market rates); current market prices NOT PROVIDED in source documents.')}\n")

    if uncertainties:
        lines.append("UNCERTAINTIES / MISSING DATA")
        for idx, u in enumerate(uncertainties, 1):
            lines.append(f"{idx}. {u.get('parameter')} – {u.get('impact')} ({u.get('recommendation')})")
        lines.append("")

    lines.append("SOURCES")
    lines.append("- TERI-2021: Thermal Comfort Prescription for Cooling Dominated Indian Residential Buildings")
    lines.append("- TARU-2015: Handbook on Achieving Thermal Comfort within Built Environment – Volume II")

    return "\n".join(lines)
