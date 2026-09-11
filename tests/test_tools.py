"""Unit tests for deterministic shelter tools."""
import pytest
from app.tools.climate import get_climate
from app.knowledge.retriever import search_knowledge
from app.tools.shelter import generate_design, modify_design
from app.tools.cost import evaluate_cost
from app.tools.thermal import evaluate_thermal
from app.tools.structure import evaluate_structure
from app.evaluation.evaluator import critique, evaluate_all
from app.memory.memory import save_experience, retrieve_experience


def test_get_climate_success():
    res = get_climate("Kerala")
    assert res["success"] is True
    assert res["climate"] == "hot_humid"
    assert res["temperature_c"] > 25


def test_get_climate_failure_simulation():
    res = get_climate("Kerala", simulate_fail=True)
    assert res["success"] is False
    assert "error" in res


def test_search_knowledge():
    results = search_knowledge("hot humid roof ventilation")
    assert len(results) > 0
    assert any("ventilation" in r.get("text", "").lower() for r in results)


def test_generate_and_modify_design():
    d1 = generate_design(capacity=5, budget=80000, climate="hot_humid")
    assert d1["version"] == 1
    assert d1["capacity"] == 5
    assert d1["roof"]["ventilation"] == 0.5

    d2 = modify_design(d1, critique="Thermal score below target")
    assert d2["version"] == 2
    assert d2["roof"]["ventilation"] > d1["roof"]["ventilation"]
    assert d2["roof"]["overhang"] > d1["roof"]["overhang"]


def test_evaluators_and_critic():
    d1 = generate_design(capacity=5, budget=80000, climate="hot_humid")
    
    # Cost
    cost_res = evaluate_cost(d1, budget=80000)
    assert cost_res["within_budget"] is True
    assert cost_res["total_cost"] <= 80000

    # Thermal V1 should fail target 0.70
    therm_v1 = evaluate_thermal(d1, climate="hot_humid")
    assert therm_v1["passed"] is False
    assert therm_v1["thermal_score"] < 0.70

    # Structure
    struct_res = evaluate_structure(d1)
    assert struct_res["passed"] is True

    # Critic on V1
    eval_v1 = {"cost": cost_res, "thermal": therm_v1, "structure": struct_res}
    crit_v1 = critique(eval_v1)
    assert crit_v1["all_passed"] is False
    assert len(crit_v1["problems"]) > 0

def test_analyze_thermal_constraints_and_cost_delta():
    from app.tools.constraints import analyze_thermal_constraints
    from app.tools.cost import calculate_improvement_cost

    d1 = generate_design(capacity=5, budget=80000, climate="hot_humid")
    climate = get_climate("Kerala")

    analysis = analyze_thermal_constraints(design=d1, climate_data=climate, location="Kerala")
    assert "analysis" in analysis
    assert len(analysis["analysis"]["thermal_constraints"]) > 0
    
    # Check that thresholds have verified sources
    for c in analysis["analysis"]["thermal_constraints"]:
        assert "source" in c
        assert "TERI-2021" in c["source"] or "TARU-2015" in c["source"] or "NBC" in c["source"] or "ASHRAE" in c["source"]

    # Check improvement cost
    cost_res = calculate_improvement_cost(improvements=analysis["improvements"], design=d1)
    assert len(cost_res["items"]) > 0
    assert cost_res["total_additional_cost_inr"] > 0
    assert "₹" in cost_res["formatted_total"]


def test_memory_save_and_retrieve():
    save_res = save_experience(
        location="Kerala",
        climate="hot_humid",
        capacity=5,
        budget=80000,
        key_learnings="Test episodic save"
    )
    assert save_res["success"] is True

    retrieved = retrieve_experience(climate="hot_humid")
    assert len(retrieved) > 0
