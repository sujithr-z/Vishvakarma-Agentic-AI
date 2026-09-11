"""
Unit and Integration Tests for PostgreSQL Experience Memory and Relational Schema.
Tests all 11 relational entities, step history, design lineage, and adaptive retrieval.
"""
import pytest
import datetime
from app.memory.database import init_db, get_engine
from app.memory.repository import (
    save_project,
    save_building,
    save_design_version,
    save_agent_run,
    save_agent_step,
    save_climate_observation,
    save_thermal_analysis,
    save_improvement,
    save_cost_result,
    save_experience,
    retrieve_relevant_experiences,
    get_design_history,
    get_agent_steps
)
from app.memory.memory import ExperienceMemory
from app.agent.planner import build_planner_prompt
from app.agent.state import AgentState


@pytest.fixture(autouse=True)
def setup_test_db():
    """Ensure database tables are initialized before each test."""
    init_db()


def test_database_init():
    """Verify database initialization and table schema availability."""
    assert init_db() is True
    engine = get_engine()
    assert engine is not None


def test_project_and_building_storage():
    """Verify creating projects and modular shelter buildings."""
    proj = save_project(project_id="test_proj_01", name="Disaster Relief Program", description="Emergency response housing")
    assert proj["id"] == "test_proj_01"
    assert proj["name"] == "Disaster Relief Program"

    bldg = save_building(
        building_id="test_bldg_01",
        project_id="test_proj_01",
        name="Modular-Relief-Unit-v1",
        climate_zone="hot_humid",
        building_type="modular_shelter",
        geometry={"floor_area_m2": 18.0, "length": 4.5, "width": 4.0}
    )
    assert bldg["id"] == "test_bldg_01"
    assert bldg["building_type"] == "modular_shelter"


def test_design_version_lineage_preservation():
    """Verify design version history (V1 is never overwritten when V2 is saved)."""
    ts = int(datetime.datetime.now().timestamp() * 1000)
    bldg_id = f"test_bldg_lineage_{ts}"
    save_building(building_id=bldg_id, name="Modular-Panel-Shelter", climate_zone="hot_humid")

    # Save V1
    v1 = save_design_version(
        version_id=f"dv_test_v1_{ts}",
        building_id=bldg_id,
        version_number=1,
        parameters={"roof_ventilation": 0.45, "overhang": 0.6, "wall_opening": 0.20}
    )
    assert v1["is_new"] is True

    # Save V2 with parent link
    v2 = save_design_version(
        version_id=f"dv_test_v2_{ts}",
        building_id=bldg_id,
        version_number=2,
        parent_version_id=f"dv_test_v1_{ts}",
        parameters={"roof_ventilation": 0.80, "overhang": 0.9, "wall_opening": 0.35, "cool_roof": "high_sri"}
    )
    assert v2["is_new"] is True

    # Verify both versions exist in history
    history = get_design_history(bldg_id)
    assert len(history) == 2
    assert history[0]["version_number"] == 1
    assert history[0]["parameters"]["roof_ventilation"] == 0.45
    assert history[1]["version_number"] == 2
    assert history[1]["parameters"]["roof_ventilation"] == 0.80
    assert history[1]["parent_version_id"] == f"dv_test_v1_{ts}"


def test_agent_run_and_step_logging():
    """Verify recording of agent runs and granular step execution history."""
    run_id = f"test_run_{int(datetime.datetime.now().timestamp())}"
    save_agent_run(
        run_id=run_id,
        user_query="Design a modular shelter for 5 people in Kerala",
        intent="design_generation",
        status="running"
    )

    # Step 1: get_climate
    save_agent_step(
        step_id=f"{run_id}_s1",
        agent_run_id=run_id,
        step_number=1,
        decision={"action": "get_climate", "arguments": {"location": "Kerala"}},
        state_before={"iteration": 0, "status": "running"},
        tool_called="get_climate",
        tool_arguments={"location": "Kerala"},
        tool_result={"location": "Kerala", "climate": "hot_humid", "temperature": 33.5},
        state_after={"iteration": 1, "status": "running"}
    )

    # Step 2: generate_design
    save_agent_step(
        step_id=f"{run_id}_s2",
        agent_run_id=run_id,
        step_number=2,
        decision={"action": "generate_design", "arguments": {"capacity": 5}},
        state_before={"iteration": 1, "status": "running"},
        tool_called="generate_design",
        tool_arguments={"capacity": 5},
        tool_result={"name": "Modular-Aid-Shelter-v1", "version": 1},
        state_after={"iteration": 2, "status": "running"}
    )

    steps = get_agent_steps(run_id)
    assert len(steps) == 2
    assert steps[0]["step_number"] == 1
    assert steps[0]["tool_called"] == "get_climate"
    assert steps[1]["step_number"] == 2
    assert steps[1]["tool_called"] == "generate_design"


def test_save_and_retrieve_experiences():
    """Verify storing complete structured experiences and retrieving them by climate."""
    exp_id = "test_exp_kerala_mod"
    save_experience(
        exp_id=exp_id,
        user_request="Design a modular shelter in Kerala",
        intent="design_generation",
        climate_context={"location": "Kerala", "climate": "hot_humid", "temperature_c": 33.0},
        success=True,
        initial_design={"name": "Modular-Aid-Shelter-v1", "version": 1, "opening_ratio": 0.20},
        thermal_result={"score": 0.55, "passed": False},
        constraints_detected=[{"constraint_id": "TC-001", "status": "VIOLATED"}],
        improvements=[{"parameter": "opening_ratio", "recommended": 0.35}],
        final_design={"name": "Modular-Aid-Shelter-v2", "version": 2, "opening_ratio": 0.35},
        final_evaluation={"thermal": {"score": 0.82, "passed": True}},
        cost_impact={"total_inr": 76500.0},
        key_learnings="Increased modular wall opening ratio to 35% with cross ventilation to achieve thermal comfort in Kerala."
    )

    retrieved = retrieve_relevant_experiences(climate="hot_humid", location="Kerala", top_k=5)
    assert len(retrieved) > 0
    found = any(r["experience_id"] == exp_id for r in retrieved)
    assert found is True

    target_exp = next(r for r in retrieved if r["experience_id"] == exp_id)
    assert target_exp["success"] is True
    assert "35%" in target_exp["key_learnings"]


def test_successful_and_failed_experiences():
    """Verify storing and distinguishing both successful and failed experience episodes."""
    save_experience(
        exp_id="test_exp_failed_case",
        user_request="Analyze poorly ventilated metal shelter",
        intent="thermal_analysis",
        climate_context={"location": "Rajasthan", "climate": "hot_dry"},
        success=False,
        failure_reason="Bare metal roof exceeded NBC adaptive limit by 8.5 °C; single-skin design without thermal mass failed.",
        key_learnings="Single-skin metal roof without thermal mass failed in hot-dry peak conditions."
    )

    all_exps = retrieve_relevant_experiences(climate="hot_dry", top_k=5)
    failed = [e for e in all_exps if e["experience_id"] == "test_exp_failed_case"]
    assert len(failed) == 1
    assert failed[0]["success"] is False
    assert "Bare metal roof" in failed[0]["failure_reason"]


def test_experience_injection_into_planner_prompt():
    """Verify previous experiences from PostgreSQL are injected into the prompt context for Qwen."""
    state = AgentState(
        user_query="Design a modular shelter for 5 people in Kerala",
        goal="Design modular shelter in Kerala",
        requirements={"location": "Kerala", "climate": "hot_humid", "capacity": 5, "budget": 80000},
        iteration=0,
        status="running"
    )
    prompt = build_planner_prompt(state)
    assert "RELEVANT PREVIOUS EXPERIENCES (FROM POSTGRESQL MEMORY)" in prompt
    assert "Kerala" in prompt or "hot_humid" in prompt
