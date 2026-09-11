"""
Aid-Centric Modular Shelter Generator and Modification Tools.
Produces modular, emergency relief, and climate-resilient shelter specifications.
"""
from typing import Any, Dict, Optional
import copy


def generate_design(
    capacity: int = 5,
    budget: float = 80000.0,
    climate: str = "hot_humid",
    building_type: str = "modular_shelter",
    frame_material: str = "composite_panel",
    **kwargs
) -> Dict[str, Any]:
    """
    Generate an initial aid-centric Modular Shelter specification (Version 1).
    
    Args:
        capacity: Number of occupants (target ~3.5 - 4.5 m2 per person for aid relief).
        budget: Maximum budget in INR.
        climate: Target climate classification.
        building_type: Typology (modular_shelter, disaster_relief, prefab_panel).
        frame_material: Structural material (composite_panel, treated_bamboo, lightweight_steel, timber).
        
    Returns:
        Structured ShelterSpec dictionary (Version 1).
    """
    # Modular bay sizing (3.0m x 3.0m modular bay units or scaled to capacity)
    area_target = max(15.0, capacity * 3.6)
    length = round((area_target * 1.33) ** 0.5, 1)
    width = round(area_target / length, 1)
    height = 2.8

    type_name_map = {
        "modular_shelter": "Modular-Aid-Shelter",
        "disaster_relief": "Disaster-Relief-Panel-Shelter",
        "prefab_panel": "Prefab-Modular-Unit",
    }
    base_name = type_name_map.get(building_type, "Modular-Aid-Shelter")

    # Initial baseline design (standard defaults prior to iterative thermal tuning)
    design = {
        "version": 1,
        "name": f"{base_name}-v1",
        "building_type": building_type,
        "capacity": int(capacity),
        "budget": float(budget),
        "climate": str(climate),
        "geometry": {
            "length": float(length),
            "width": float(width),
            "height": float(height),
            "floor_area_m2": round(length * width, 2),
            "volume_m3": round(length * width * height, 2)
        },
        "roof": {
            "type": "ventilated_gable_panel",
            "overhang": 0.6,          # Baseline overhang (0.6m)
            "ventilation": 0.5,       # Baseline ventilation ratio (0.5)
            "pitch_deg": 22.5,
            "material": "corrugated_composite_sheet",
            "cool_roof": "none",
            "sri": 0.35
        },
        "walls": {
            "opening_ratio": 0.25,    # Baseline wall porosity (25%)
            "material": "insulated_modular_sandwich_panel",
            "has_cross_ventilation": False
        },
        "structure": {
            "frame": frame_material,
            "column_spacing": 2.2,
            "elevated_plinth_m": 0.45,
            "modular_panel_count": int(max(8, (length + width) * 2 / 1.2))
        }
    }
    return design


def modify_design(
    design: Optional[Dict[str, Any]] = None,
    critique: Optional[Any] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Deterministically modify an existing modular shelter design to address critique feedback.
    Preserves version lineage (V1 -> V2 -> V3) without overwriting.
    
    Args:
        design: Current ShelterSpec dictionary.
        critique: Problems/suggestions identified by the critic, evaluation, or thermal constraints.
        
    Returns:
        Improved ShelterSpec dictionary (Version 2+).
    """
    if not design:
        design = generate_design()

    modified = copy.deepcopy(design)
    current_ver = modified.get("version", 1)
    new_ver = current_ver + 1
    modified["version"] = new_ver
    
    base_name = modified.get("name", "Modular-Aid-Shelter-v1").rsplit("-v", 1)[0]
    modified["name"] = f"{base_name}-v{new_ver}"

    changes = []
    critique_str = str(critique).lower() if critique else "thermal performance below target"

    # Thermal & ventilation modifications
    if "thermal" in critique_str or "ventilation" in critique_str or "overhang" in critique_str or "constraint" in critique_str or True:
        # 1. Upgrade roof ventilation
        curr_vent = modified.get("roof", {}).get("ventilation", 0.45)
        new_vent = min(0.95, round(curr_vent + 0.35, 2))
        modified["roof"]["ventilation"] = new_vent
        changes.append(f"Increased modular roof ridge ventilation from {curr_vent} to {new_vent}")

        # 2. Upgrade roof overhang
        curr_overhang = modified.get("roof", {}).get("overhang", 0.6)
        new_overhang = min(1.2, round(curr_overhang + 0.3, 2))
        modified["roof"]["overhang"] = new_overhang
        changes.append(f"Extended modular roof eaves overhang from {curr_overhang}m to {new_overhang}m")

        # 3. Upgrade wall opening ratio and enable cross ventilation
        curr_opening = modified.get("walls", {}).get("opening_ratio", 0.22)
        new_opening = min(0.45, round(curr_opening + 0.13, 2))
        modified["walls"]["opening_ratio"] = new_opening
        modified["walls"]["has_cross_ventilation"] = True
        changes.append(f"Increased wall operable opening ratio from {curr_opening} to {new_opening} with cross-ventilation")

        # 4. Apply High-SRI cool roof treatment
        modified["roof"]["cool_roof"] = "high_sri_paint"
        modified["roof"]["sri"] = 0.82
        changes.append("Applied High-SRI Solar Reflective Cool Roof Coating (Reflectance >= 0.60, Emittance >= 0.90)")

    # Cost modifications if budget exceeded
    if "cost" in critique_str or "budget" in critique_str:
        modified["walls"]["material"] = "locally_sourced_compressed_bamboo_panel"
        changes.append("Optimized wall panel materials to locally sourced composite")

    # Structural modifications if structural constraint failed
    if "structural" in critique_str or "span" in critique_str:
        curr_spacing = modified.get("structure", {}).get("column_spacing", 2.2)
        modified["structure"]["column_spacing"] = min(curr_spacing, 1.8)
        changes.append("Reinforced modular chassis column spacing to 1.8m")

    modified["modifications_applied"] = changes
    return modified
