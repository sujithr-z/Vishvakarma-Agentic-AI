"""
The 16 Capability & Critical Action Protocol Audit Test Suite for Vishvakarma Agent.

Validates:
1. Test 1 — "1 + 1" General Query (Direct FINAL, 0 tool calls)
2. Test 2 — "What is thermal comfort?" (Knowledge/RAG -> FINAL, No climate tools)
3. Test 3 — "Analyze the model house for thermal constraints" (Dynamic tool sequence)
4. Test 4 — "Design a low-cost shelter for 5 people in Kerala" (Dynamic design loop)
5. Test 5 — Invalid Decision Handling (Malformed syntax / placeholder -> Feedback -> Recovery)
6. Test 6 — Tool Failure Recovery (Observation -> Qwen decides next action)
7. Test 7 — Requirement Understanding & Parameter Extraction
8. Test 8 — Unbiased State-Driven Prompt (No forced checklists or required actions)
9. Test 9 — Strict 2-Type Decision Model Validation
10. Test 10 — Argument Integrity
11. Test 11 — Simple Cost Calculation (Cost tool without climate fetch)
12. Test 12 — Follow-up with Existing State (Does not repeat gathered data)
13. Test 13 — Insufficient / Missing Information (States unavailability; no hallucination)
14. Test 14 — Labeled Evidence Tracking (SOURCE and STATUS tags)
15. Test 15 — Design Improvement Cycle (V1 -> Critique -> V2)
16. Test 16 — Conversational Welcome Handling
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
from app.evaluation.evaluator import critique


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
        return '{"type": "final", "content": "Task completed successfully."}'


# Test 1 — "1 + 1" (Direct FINAL answer, 0 tool calls, no shelter tools)
def test_1_plus_1_direct_final_no_tools():
    mock_llm = MockLLM([
        '{"type": "final", "content": "2"}'
    ])
    agent = Vishvakarma(llm=mock_llm, verbose=False)
    state = agent.run(user_query="1 + 1")

    assert state.status == "completed"
    assert state.final_answer == "2"
    # Ensure zero tool calls were made
    assert len(state.tool_results) == 0
    called_tools = [c.get("tool_name") for c in state.tool_calls if c.get("type") == "tool"]
    assert len(called_tools) == 0


# Test 2 — "What is thermal comfort?" (Knowledge/RAG -> FINAL, no climate tools)
def test_knowledge_query_no_climate_tools():
    mock_llm = MockLLM([
        '{"type": "tool", "tool_name": "search_knowledge", "arguments": {"query": "what is thermal comfort NBC 2016"}, "reason": "Retrieve standard definitions"}',
        '{"type": "final", "content": "Thermal comfort is the condition of mind that expresses satisfaction with the thermal environment (NBC 2016 / TERI-2021)."}'
    ])
    agent = Vishvakarma(llm=mock_llm, verbose=False)
    state = agent.run(user_query="What is thermal comfort?")

    assert state.status == "completed"
    called_tools = [c.get("tool_name") for c in state.tool_calls if c.get("type") == "tool"]
    assert "get_climate" not in called_tools
    assert "evaluate_thermal" not in called_tools
    assert "evaluate_cost" not in called_tools
    assert "search_knowledge" in called_tools


# Test 3 — "Analyze the model house for thermal constraints" (Dynamic tool sequence)
def test_diagnostic_sequence_dynamic():
    mock_llm = MockLLM([
        '{"type": "tool", "tool_name": "get_climate", "arguments": {"location": "Kerala"}, "reason": "Need climate data for Kerala"}',
        '{"type": "tool", "tool_name": "analyze_thermal_constraints", "arguments": {"location": "Kerala"}, "reason": "Analyze against NBC 2016 limits"}',
        '{"type": "final", "content": "Identified high indoor temperature TC-001 and inadequate airflow TC-004."}'
    ])
    agent = Vishvakarma(llm=mock_llm, verbose=False)
    state = agent.run(user_query="Analyze this house for thermal constraints in Kerala.")

    assert state.status == "completed"
    assert len(state.constraints) > 0
    obs_sources = [o.source for o in state.observations]
    assert "climate_tool" in obs_sources
    assert "constraint_engine" in obs_sources


# Test 4 — "Design a low-cost shelter for 5 people in Kerala" (Dynamic design loop)
def test_design_shelter_dynamic_loop():
    mock_llm = MockLLM([
        '{"type": "tool", "tool_name": "get_climate", "arguments": {"location": "Kerala"}, "reason": "Get environmental parameters"}',
        '{"type": "tool", "tool_name": "generate_design", "arguments": {"capacity": 5, "budget": 80000}, "reason": "Generate v1"}',
        '{"type": "tool", "tool_name": "evaluate_thermal", "arguments": {}, "reason": "Evaluate thermal"}',
        '{"type": "final", "content": "Shelter design completed and evaluated."}'
    ])
    agent = Vishvakarma(llm=mock_llm, verbose=False)
    state = agent.run(user_query="Design a low-cost shelter for 5 people in Kerala with a budget of ₹80,000.")

    assert state.status == "completed"
    assert state.current_design is not None
    assert "thermal" in state.evaluation


# Test 5 — Invalid Decision Handling (Malformed syntax / placeholder -> Feedback -> Recovery)
def test_invalid_decision_feedback_and_recovery():
    mock_llm = MockLLM([
        '{"action": "<finish tool name> (or <answer tool name>)", "arguments": {}}',  # Malformed placeholder output
        '{"type": "tool", "tool_name": "evaluate_thermal", "arguments": {}, "reason": "Recover by providing valid tool JSON"}'
    ])
    agent = Vishvakarma(llm=mock_llm, verbose=False)
    state = AgentState(
        user_query="Evaluate house",
        current_design=generate_design(5, 80000)
    )

    # Step 1: Rejected as INVALID_MODEL_DECISION observation
    agent.run_step(state)
    assert len(state.observations) == 1
    assert state.observations[0].status == "INVALID_MODEL_DECISION"

    # Step 2: Qwen receives feedback observation and recovers with valid tool call
    agent.run_step(state)
    assert len(state.observations) == 2
    assert state.observations[1].status == "CALCULATED"
    assert "thermal" in state.evaluation


# Test 6 — Tool Failure Recovery (Observation -> Qwen decides next action)
def test_tool_failure_feedback_and_recovery():
    mock_llm = MockLLM([
        '{"type": "tool", "tool_name": "get_climate", "arguments": {"location": "fail"}, "reason": "Fetch climate"}',
        '{"type": "tool", "tool_name": "search_knowledge", "arguments": {"query": "hot humid principles"}, "reason": "Recover from climate failure by searching knowledge"}'
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


# Test 7 — Requirement Understanding & Parameter Extraction
def test_requirement_understanding():
    query = "Design shelter for 5 people, hot humid climate (Kerala), ₹80,000."
    reqs = extract_initial_requirements(query)
    
    assert reqs["capacity"] == 5
    assert reqs["climate"] == "hot_humid"
    assert reqs["budget"] == 80000.0
    assert reqs["location"] == "Kerala"


# Test 8 — Unbiased State-Driven Prompt (No forced checklists or required actions)
def test_unbiased_state_prompt():
    state = AgentState(
        user_query="Analyze the model house and determine what thermal constraints it could exceed.",
        requirements={"capacity": 5, "location": "Kerala", "climate": "hot_humid"}
    )
    prompt = build_planner_prompt(state)

    assert "AVAILABLE REGISTERED TOOLS:" in prompt
    assert "CURRENT SITUATION & STATE" in prompt
    assert "REQUIRED NEXT ACTION" not in prompt
    assert "<tool_name or finish" not in prompt
    assert "<finish tool name>" not in prompt


# Test 9 — Strict 2-Type Decision Model Validation
def test_strict_2_type_decision_validation():
    # Valid tool
    d_tool = AgentDecision(type="tool", tool_name="get_climate", arguments={"location": "Kerala"}, reason="test")
    assert d_tool.type == "tool"
    assert d_tool.tool_name == "get_climate"
    assert d_tool.content is None

    # Valid final
    d_final = AgentDecision(type="final", content="Final report content")
    assert d_final.type == "final"
    assert d_final.content == "Final report content"
    assert d_final.tool_name is None

    # Invalid tool (missing tool_name)
    with pytest.raises(ValueError):
        AgentDecision(type="tool", content="bad")

    # Invalid final (missing content)
    with pytest.raises(ValueError):
        AgentDecision(type="final", tool_name="bad")


# Test 10 — Argument Integrity
def test_argument_integrity():
    query = "Design for 5 people under ₹80,000 in Kerala"
    reqs = extract_initial_requirements(query)
    decision = AgentDecision(
        type="tool",
        tool_name="generate_design",
        arguments={"capacity": reqs["capacity"], "budget": reqs["budget"]},
        reason="Model passed extracted parameters"
    )
    assert decision.arguments["capacity"] == 5
    assert decision.arguments["budget"] == 80000.0


# Test 11 — Simple Cost Calculation (Cost tool without climate fetch)
def test_simple_calculation_conditional():
    mock_llm = MockLLM([
        '{"type": "tool", "tool_name": "calculate_improvement_cost", "arguments": {}, "reason": "Calculate roof overhang and cool roof cost"}',
        '{"type": "final", "content": "The cost of extending roof overhang and cool-roof is ₹3,200 (2014 INR base)."}'
    ])
    agent = Vishvakarma(llm=mock_llm, verbose=False)
    d1 = generate_design(5, 80000)
    state = AgentState(
        user_query="Calculate the cost of increasing the roof overhang.",
        current_design=d1
    )
    agent.run_step(state)
    agent.run_step(state)

    called_tools = [c.get("tool_name") for c in state.tool_calls if c.get("type") == "tool"]
    assert "calculate_improvement_cost" in called_tools
    assert "get_climate" not in called_tools


# Test 12 — Follow-up with Existing State (Does not repeat gathered data)
def test_follow_up_with_existing_state():
    mock_llm = MockLLM([
        '{"type": "tool", "tool_name": "analyze_thermal_constraints", "arguments": {}, "reason": "Use existing climate data to evaluate constraints directly"}',
        '{"type": "final", "content": "The house is uncomfortable due to roof heat gain and low opening ratio."}'
    ])
    agent = Vishvakarma(llm=mock_llm, verbose=False)
    state = AgentState(
        user_query="Why is the house thermally uncomfortable?",
        requirements={
            "location": "Kerala",
            "climate_data": {"location": "Kerala", "temperature": 33.0, "humidity": 82.0, "trm": 32.0, "climate": "hot_humid"}
        }
    )
    agent.run_step(state)
    agent.run_step(state)

    called_tools = [c.get("tool_name") for c in state.tool_calls if c.get("type") == "tool"]
    assert "get_climate" not in called_tools
    assert "analyze_thermal_constraints" in called_tools


# Test 13 — Insufficient / Missing Information (States unavailability; no hallucination)
def test_insufficient_information_no_hallucination():
    mock_llm = MockLLM([
        '{"type": "final", "content": "Future climate data for 6 PM tomorrow is unavailable. Current historical baseline shows 33°C."}'
    ])
    agent = Vishvakarma(llm=mock_llm, verbose=False)
    state = agent.run(user_query="Is this house comfortable at 6 PM tomorrow?")

    assert state.status == "completed"
    assert "unavailable" in state.final_answer.lower() or "not available" in state.final_answer.lower()


# Test 14 — Labeled Evidence Tracking (SOURCE and STATUS tags)
def test_labeled_evidence_tracking():
    mock_llm = MockLLM([
        '{"type": "tool", "tool_name": "get_climate", "arguments": {"location": "Kerala"}, "reason": "Fetch weather"}',
        '{"type": "tool", "tool_name": "search_knowledge", "arguments": {"query": "roof ventilation"}, "reason": "Get principles"}'
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


# Test 15 — Design Improvement Cycle (V1 -> Critique -> V2)
def test_design_improvement_cycle():
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


# Test 16 — Conversational Welcome Handling
def test_greeting_welcome():
    agent = Vishvakarma(verbose=False)
    state = agent.run(user_query="hi")
    assert state.status == "completed"
    assert state.final_answer is not None
    assert "Vishvakarma" in state.final_answer
    assert "NBC 2016" in state.final_answer
