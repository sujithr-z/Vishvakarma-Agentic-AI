"""
The 15 Capability & Critical Reasoning Audit Test Suite for Vishvakarma Agent.

Validates:
1. Requirement Understanding
2. Unbiased Prompt Construction (No forced checklists or required actions)
3. Structured JSON & Type-based Decision Formats
4. Argument Integrity
5. Tool Failure Recovery (Observation -> Reasoning -> Recovery)
6. Invalid Tool Recovery (Environment rejects unknown tool -> Observation -> Recovery)
7. Test A — Pure Knowledge Query (No unnecessary climate/thermal calculations)
8. Test B — Simple Calculation (Cost calculation without climate fetch)
9. Test C — Diagnostic Multi-step Sequence (Observe -> Reason -> Act loop)
10. Test D — Follow-up with Existing State (Does not repeat gathered data)
11. Test E — Missing / Insufficient Data (States unavailability; no hallucination)
12. Labeled Evidence Tracking (SOURCE and STATUS tags)
13. Design Improvement Cycle (V1 -> Critique -> V2)
14. State Maintenance across Iterations
15. Conversational Welcome Handling
"""
import pytest
from app.agent.state import AgentState
from app.agent.agent import Vishvakarma, ShelterAgent, extract_initial_requirements
from app.agent.parser import parse_decision, AgentDecision
from app.agent.planner import build_planner_prompt
from app.tools.shelter import generate_design, modify_design
from app.tools.thermal import evaluate_thermal
from app.tools.climate import get_climate
from app.knowledge.retriever import search_knowledge
from app.evaluation.evaluator import critique, evaluate_all


class MockLLM:
    """Mock LLM for isolated deterministic testing of agent controller logic."""
    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0

    def generate(self, prompt: str, system: str = None, temperature: float = 0.1) -> str:
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        return '{"action": "finish", "arguments": {"summary": "Completed"}, "reason": "Default finish"}'


# Test 1 — Requirement Understanding
def test_capability_1_requirement_understanding():
    query = "Design shelter for 5 people, hot humid climate (Kerala), ₹80,000."
    reqs = extract_initial_requirements(query)
    
    assert reqs["capacity"] == 5
    assert reqs["climate"] == "hot_humid"
    assert reqs["budget"] == 80000.0
    assert reqs["location"] == "Kerala"


# Test 2 — Unbiased State-Driven Prompt (No Forced Action)
def test_capability_2_unbiased_state_prompt():
    state = AgentState(
        user_query="Analyze the model house and determine what thermal constraints it could exceed.",
        requirements={"capacity": 5, "location": "Kerala", "climate": "hot_humid"}
    )
    prompt = build_planner_prompt(state)

    # Prompt MUST provide choices and state, NOT tell Qwen what it MUST do
    assert "AVAILABLE TOOLS:" in prompt
    assert "CURRENT SITUATION & STATE" in prompt
    assert "REQUIRED NEXT ACTION" not in prompt
    assert "You must call" not in prompt


# Test 3 — Structured JSON & Type-based Decision Formats
def test_capability_3_valid_structured_json_and_type_formats():
    raw_outputs = [
        '{"action": "get_climate", "arguments": {"location": "Kerala"}, "reason": "Fetch climate conditions"}',
        '{"type": "tool", "tool_name": "search_knowledge", "arguments": {"query": "roof ventilation"}, "reason": "Need engineering rules"}',
        '{"type": "final", "content": "Thermal comfort is achieved when Tn is within acceptable limits."}',
        '```json\n{"action": "generate_design", "arguments": {"capacity": 5, "budget": 80000}, "reason": "Create shelter"}\n```'
    ]
    for raw in raw_outputs:
        decision = parse_decision(raw)
        assert isinstance(decision, AgentDecision)
        assert decision.action in ["get_climate", "search_knowledge", "answer", "generate_design"]
        assert isinstance(decision.arguments, dict)


# Test 4 — Correct Arguments
def test_capability_4_correct_arguments():
    query = "Design for 5 people under ₹80,000 in Kerala"
    reqs = extract_initial_requirements(query)
    decision = AgentDecision(
        action="generate_design",
        arguments={"capacity": reqs["capacity"], "budget": reqs["budget"]},
        reason="Model passed extracted parameters"
    )
    assert decision.arguments["capacity"] == 5
    assert decision.arguments["budget"] == 80000.0


# Test 5 — Tool Failure Recovery via Labeled Observation
def test_capability_5_tool_failure_recovery():
    fail_res = get_climate("Kerala", simulate_fail=True)
    assert fail_res["success"] is False

    mock_llm = MockLLM([
        '{"action": "get_climate", "arguments": {"location": "fail"}, "reason": "Fetch climate"}',
        '{"action": "search_knowledge", "arguments": {"query": "hot humid principles"}, "reason": "Recover from climate failure by searching knowledge"}'
    ])
    agent = Vishvakarma(llm=mock_llm, verbose=False)
    state = AgentState(user_query="Design shelter", requirements={"capacity": 5})
    
    # Step 1: fails and records TOOL_ERROR observation
    agent.run_step(state)
    assert len(state.observations) == 1
    assert state.observations[0].status == "TOOL_ERROR"

    # Step 2: Qwen reasons from observation and chooses alternative tool
    agent.run_step(state)
    assert len(state.observations) == 2
    assert state.observations[1].status == "RETRIEVED"
    assert isinstance(state.tool_results[1], list)


# Test 6 — Invalid Action Name Recovery (No silent substitution)
def test_capability_6_invalid_tool_recovery():
    mock_llm = MockLLM([
        '{"action": "calculate_thermally_acceptable", "arguments": {}, "reason": "Try invented tool name"}',
        '{"action": "evaluate_thermal", "arguments": {}, "reason": "Recover by choosing registered tool"}'
    ])
    agent = Vishvakarma(llm=mock_llm, verbose=False)
    state = AgentState(
        user_query="Evaluate house",
        current_design=generate_design(5, 80000)
    )

    # Step 1: Python environment rejects invalid tool and adds INVALID_ACTION observation
    agent.run_step(state)
    assert len(state.observations) == 1
    assert state.observations[0].status == "INVALID_ACTION"
    assert "calculate_thermally_acceptable" in state.observations[0].content

    # Step 2: Qwen receives invalid action observation and selects valid tool
    agent.run_step(state)
    assert len(state.observations) == 2
    assert state.observations[1].status == "CALCULATED"
    assert "thermal" in state.evaluation


# Test 7 — Test A: Knowledge Query (Conditional Tool Calling)
def test_capability_7_test_a_knowledge_query():
    mock_llm = MockLLM([
        '{"action": "search_knowledge", "arguments": {"query": "what is thermal comfort NBC 2016"}, "reason": "Retrieve standard definitions"}',
        '{"action": "answer", "arguments": {"message": "Thermal comfort is the condition of mind that expresses satisfaction with the thermal environment (NBC 2016 / TERI-2021)."}, "reason": "Evidence sufficient"}'
    ])
    agent = Vishvakarma(llm=mock_llm, verbose=False)
    state = agent.run(user_query="What is thermal comfort?")

    assert state.status == "completed"
    called_actions = [c["action"] for c in state.tool_calls]
    # Must NOT call climate or thermal tools for a pure knowledge query
    assert "get_climate" not in called_actions
    assert "evaluate_thermal" not in called_actions
    assert "evaluate_cost" not in called_actions
    assert "search_knowledge" in called_actions


# Test 8 — Test B: Simple Calculation (Conditional Tool Calling)
def test_capability_8_test_b_simple_calculation():
    mock_llm = MockLLM([
        '{"action": "calculate_improvement_cost", "arguments": {}, "reason": "Calculate roof overhang and cool roof cost"}',
        '{"action": "finish", "arguments": {"summary": "The cost of extending roof overhang and cool-roof is ₹3,200 (2014 INR base)."}, "reason": "Cost calculation obtained"}'
    ])
    agent = Vishvakarma(llm=mock_llm, verbose=False)
    d1 = generate_design(5, 80000)
    state = AgentState(
        user_query="Calculate the cost of increasing the roof overhang.",
        current_design=d1
    )
    agent.run_step(state)
    agent.run_step(state)

    called_actions = [c["action"] for c in state.tool_calls]
    assert "calculate_improvement_cost" in called_actions
    assert "get_climate" not in called_actions  # Did not make unnecessary climate calls


# Test 9 — Test C: Diagnostic Sequence (Observe -> Reason -> Act)
def test_capability_9_test_c_diagnostic_sequence():
    mock_llm = MockLLM([
        '{"action": "get_climate", "arguments": {"location": "Kerala"}, "reason": "Need climate data for Kerala"}',
        '{"action": "analyze_thermal_constraints", "arguments": {"location": "Kerala"}, "reason": "Analyze against NBC 2016 limits"}',
        '{"action": "finish", "arguments": {"summary": "Identified high indoor temperature TC-001 and inadequate airflow TC-004."}, "reason": "Constraints analyzed"}'
    ])
    agent = Vishvakarma(llm=mock_llm, verbose=False)
    state = agent.run(user_query="Analyze this house for thermal constraints in Kerala.")

    assert state.status == "completed"
    assert len(state.constraints) > 0
    assert len(state.observations) >= 2
    # Verify labeled observations
    obs_sources = [o.source for o in state.observations]
    assert "climate_tool" in obs_sources
    assert "constraint_engine" in obs_sources


# Test 10 — Test D: Follow-up with Existing State
def test_capability_10_test_d_follow_up_with_existing_state():
    mock_llm = MockLLM([
        '{"action": "analyze_thermal_constraints", "arguments": {}, "reason": "Use existing climate data to evaluate constraints directly"}',
        '{"action": "answer", "arguments": {"message": "The house is uncomfortable due to roof heat gain and low opening ratio."}, "reason": "Answered"}'
    ])
    agent = Vishvakarma(llm=mock_llm, verbose=False)
    # Pre-populate state with climate data already present
    state = AgentState(
        user_query="Why is the house thermally uncomfortable?",
        requirements={
            "location": "Kerala",
            "climate_data": {"location": "Kerala", "temperature": 33.0, "humidity": 82.0, "trm": 32.0, "climate": "hot_humid"}
        }
    )
    agent.run_step(state)
    agent.run_step(state)

    called_actions = [c["action"] for c in state.tool_calls]
    # Since climate was already available, get_climate was not called
    assert "get_climate" not in called_actions
    assert "analyze_thermal_constraints" in called_actions


# Test 11 — Test E: Insufficient Information / Missing Data
def test_capability_11_test_e_insufficient_information():
    mock_llm = MockLLM([
        '{"action": "answer", "arguments": {"message": "Future climate data for 6 PM tomorrow is unavailable. Current historical baseline shows 33°C."}, "reason": "Do not hallucinate future weather"}'
    ])
    agent = Vishvakarma(llm=mock_llm, verbose=False)
    state = agent.run(user_query="Is this house comfortable at 6 PM tomorrow?")

    assert state.status == "completed"
    assert "unavailable" in state.final_answer.lower() or "not available" in state.final_answer.lower()


# Test 12 — Labeled Evidence Tracking (SOURCE and STATUS)
def test_capability_12_evidence_labels_and_status():
    mock_llm = MockLLM([
        '{"action": "get_climate", "arguments": {"location": "Kerala"}, "reason": "Fetch weather"}',
        '{"action": "search_knowledge", "arguments": {"query": "roof ventilation"}, "reason": "Get principles"}'
    ])
    agent = Vishvakarma(llm=mock_llm, verbose=False)
    state = AgentState(user_query="Design climate resilient shelter in Kerala")

    agent.run_step(state)
    agent.run_step(state)

    assert len(state.observations) == 2
    assert state.observations[0].source == "climate_tool"
    assert state.observations[0].status == "OBSERVED"
    assert state.observations[1].source == "knowledge_base"
    assert state.observations[1].status == "RETRIEVED"


# Test 13 — Design Improvement Cycle (V1 -> Critique -> V2)
def test_capability_13_design_improvement_cycle():
    v1 = generate_design(capacity=5, budget=80000, climate="hot_humid")
    t1 = evaluate_thermal(v1, climate="hot_humid")
    assert t1["passed"] is False

    crit = critique({"thermal": t1, "cost": {"passed": True}, "structure": {"passed": True}})
    assert crit["all_passed"] is False

    v2 = modify_design(v1, crit)
    assert v2["version"] == 2

    t2 = evaluate_thermal(v2, climate="hot_humid")
    assert t2["thermal_score"] > t1["thermal_score"]
    assert t2["passed"] is True
    assert t2["thermal_score"] >= 0.70


# Test 14 — State Maintenance across Iterations
def test_capability_14_state_maintenance():
    mock_llm = MockLLM([
        '{"action": "get_climate", "arguments": {"location": "Kerala"}, "reason": "Get climate"}',
        '{"action": "generate_design", "arguments": {"capacity": 5, "budget": 80000}, "reason": "Generate v1"}',
        '{"action": "evaluate_thermal", "arguments": {}, "reason": "Evaluate thermal"}'
    ])
    agent = Vishvakarma(llm=mock_llm, verbose=False)
    state = AgentState(user_query="Design shelter for 5 people in Kerala")
    
    agent.run_step(state)
    agent.run_step(state)
    agent.run_step(state)

    assert state.iteration == 3
    assert state.requirements["climate"] == "hot_humid"
    assert state.current_design is not None
    assert len(state.tool_calls) == 3
    assert len(state.observations) == 3


# Test 15 — Greeting & Conversational Welcome
def test_capability_15_greeting_welcome():
    agent = Vishvakarma(verbose=False)
    state = agent.run(user_query="hi")
    assert state.status == "completed"
    assert state.final_answer is not None
    assert "Vishvakarma" in state.final_answer
    assert "NBC 2016" in state.final_answer
