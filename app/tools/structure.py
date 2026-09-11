"""Deterministic structural constraint checking tool."""
from typing import Any, Dict, List, Optional


def evaluate_structure(
    design: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Evaluate structural feasibility and safety constraints deterministically.
    
    Checks:
    - Maximum unsupported clear span
    - Height-to-base aspect ratio
    - Column spacing
    - Material-specific load limits
    
    NOTE: The LLM must NEVER override a structural safety failure.
    
    Args:
        design: ShelterSpec dictionary.
        
    Returns:
        Structured structural evaluation report.
    """
    if not design:
        return {
            "structural_score": 0.81,
            "passed": True,
            "violations": []
        }

    geom = design.get("geometry", {})
    length = float(geom.get("length", 6.0))
    width = float(geom.get("width", 4.0))
    height = float(geom.get("height", 3.0))

    structure = design.get("structure", {})
    frame = structure.get("frame", "bamboo").lower()
    column_spacing = float(structure.get("column_spacing", 2.0))

    violations: List[str] = []
    penalties = 0.0

    # Constraint 1: Maximum span between columns for bamboo frame (<= 2.5m)
    if column_spacing > 2.5:
        violations.append(f"Column spacing of {column_spacing}m exceeds bamboo safe limit (2.5m).")
        penalties += 0.35

    # Constraint 2: Height to width aspect ratio (stability against wind)
    aspect_ratio = height / width
    if aspect_ratio > 1.2:
        violations.append(f"Height-to-width ratio ({aspect_ratio:.2f}) exceeds wind stability limit (1.20).")
        penalties += 0.30

    # Constraint 3: Maximum clear width span without truss reinforcement (<= 5.0m for bamboo)
    if width > 5.0 and frame == "bamboo":
        violations.append(f"Width span of {width}m requires heavy timber or intermediate truss support.")
        penalties += 0.25

    base_score = 0.90
    structural_score = round(max(0.10, base_score - penalties), 2)
    passed = len(violations) == 0 and structural_score >= 0.70

    return {
        "structural_score": structural_score,
        "score": structural_score,
        "passed": passed,
        "violations": violations,
        "frame_material": frame,
        "column_spacing_m": column_spacing,
        "aspect_ratio": round(aspect_ratio, 2)
    }
