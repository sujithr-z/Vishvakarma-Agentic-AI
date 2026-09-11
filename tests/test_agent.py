"""
The 10 Capability Test Suite for Adaptive Shelter Agent v0.1 (Section 24).

Tests:
1. Understand shelter requirements
2. Select the correct tool
3. Generate valid structured JSON
4. Provide correct tool arguments
5. Recover from tool failure
6. Use retrieved knowledge
7. Inspect evaluation results
8. Choose another action after failure
9. Maintain state
10. Improve a design through iteration (V2 > V1)
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
from app.llm.ollama_client import OllamaClient


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


# Test 2 — Tool Selection
def test_capability_2_tool_selection():
    state = AgentState(
        user_query="What is the thermal performance of this design?",
        requirements={"capacity": 5, "climate": "hot_humid"},
        current_design=generate_design(5, 80000)
    )
    prompt = build_planner_prompt(state)
    # The prompt explicitly presents available tools and design status
    assert "evaluate_thermal" in prompt
    assert "Version 1" in prompt


# Test 3 — Valid Structured JSON
def test_capability_3_valid_structured_json():
    raw_outputs = [
        '{"action": "get_climate", "arguments": {"location": "Kerala"}, "reason": "Fetch climate conditions"}',
        '```json\n{"action": "generate_design", "arguments": {"capacity": 5, "budget": 80000}, "reason": "Create shelter"}\n```'
    ]
    for raw in raw_outputs:
        decision = parse_decision(raw)
        assert isinstance(decision, AgentDecision)
        assert decision.action in ["get_climate", "generate_design"]
        assert isinstance(decision.arguments, dict)
        assert len(decision.reason) > 0


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


# Test 5 — Tool Failure Recovery
def test_capability_5_tool_failure_recovery():
    # Force climate failure
    fail_res = get_climate("Kerala", simulate_fail=True)
    assert fail_res["success"] is False
    assert "error" in fail_res

    # Mock agent step handling failure without hallucinating success
    mock_llm = MockLLM([
        '{"action": "get_climate", "arguments": {"location": "fail"}, "reason": "Fetch climate"}',
        '{"action": "search_knowledge", "arguments": {"query": "hot humid principles"}, "reason": "Recover from climate failure by searching knowledge"}'
    ])
    agent = ShelterAgent(llm=mock_llm, verbose=False)
    state = AgentState(user_query="Design shelter", requirements={"capacity": 5})
    
    # Step 1: fails
    agent.run_step(state)
    assert len(state.tool_results) == 1
    assert state.tool_results[0]["success"] is False

    # Step 2: recovers with search_knowledge
    agent.run_step(state)
    assert len(state.tool_results) == 2
    assert isinstance(state.tool_results[1], list)  # Knowledge list


# Test 6 — Knowledge Retrieval
def test_capability_6_knowledge_retrieval():
    docs = search_knowledge("Why is roof ventilation useful in a hot-humid shelter?")
    assert len(docs) > 0
    found_ventilation_topic = any("ventilation" in d.get("topic", "").lower() or "ventilation" in d.get("text", "").lower() for d in docs)
    assert found_ventilation_topic is True


# Test 7 — Evaluation Reasoning
def test_capability_7_evaluation_reasoning():
    design = generate_design(5, 80000)
    thermal_eval = evaluate_thermal(design, climate="hot_humid")
    
    assert thermal_eval["thermal_score"] < 0.70
    assert thermal_eval["passed"] is False

    eval_dict = {"thermal": thermal_eval, "cost": {"passed": True}, "structure": {"passed": True}}
    crit = critique(eval_dict)
    assert crit["all_passed"] is False
    assert any("thermal" in p.lower() for p in crit["problems"])


# Test 8 — Next Action after Failure
def test_capability_8_next_action_after_failure():
    # If thermal failed, next action should be modify_design
    mock_llm = MockLLM([
        '{"action": "modify_design", "arguments": {}, "reason": "Thermal score 0.58 below 0.70 threshold; modifying parameters"}'
    ])
    agent = ShelterAgent(llm=mock_llm, verbose=False)
    
    d1 = generate_design(5, 80000)
    state = AgentState(
        user_query="Design shelter",
        current_design=d1,
        evaluation={"thermal": {"thermal_score": 0.58, "target": 0.70, "passed": False}},
        critique={"all_passed": False, "problems": ["Thermal performance below target."]}
    )
    decision = agent.run_step(state)
    assert decision.action == "modify_design"
    assert state.current_design["version"] == 2


# Test 9 — State Maintenance
def test_capability_9_state_maintenance():
    mock_llm = MockLLM([
        '{"action": "get_climate", "arguments": {"location": "Kerala"}, "reason": "Get climate"}',
        '{"action": "generate_design", "arguments": {"capacity": 5, "budget": 80000}, "reason": "Generate v1"}',
        '{"action": "evaluate_thermal", "arguments": {}, "reason": "Evaluate thermal"}'
    ])
    agent = ShelterAgent(llm=mock_llm, verbose=False)
    state = AgentState(user_query="Design shelter for 5 people in Kerala")
    
    agent.run_step(state)
    agent.run_step(state)
    agent.run_step(state)

    assert state.iteration == 3
    assert state.requirements["climate"] == "hot_humid"
    assert state.current_design is not None
    assert len(state.tool_calls) == 3
    assert len(state.tool_results) == 3
    assert "thermal" in state.evaluation


# Test 10 — Design Improvement Cycle (V2 > V1)
def test_capability_10_design_improvement_cycle():
    # V1 initial design
    v1 = generate_design(capacity=5, budget=80000, climate="hot_humid")
    t1 = evaluate_thermal(v1, climate="hot_humid")
    assert t1["passed"] is False

    # Critic identifies issue
    crit = critique({"thermal": t1, "cost": {"passed": True}, "structure": {"passed": True}})
    assert crit["all_passed"] is False

    # Modify to V2
    v2 = modify_design(v1, crit)
    assert v2["version"] == 2

    # Re-evaluate V2
    t2 = evaluate_thermal(v2, climate="hot_humid")
    assert t2["thermal_score"] > t1["thermal_score"]
    assert t2["passed"] is True
    assert t2["thermal_score"] >= 0.70


# Test 11 — Greeting & Conversational Welcome
def test_capability_11_greeting_welcome():
    agent = Vishvakarma(verbose=False)
    state = agent.run(user_query="hi")
    assert state.status == "completed"
    assert state.final_answer is not None
    assert "Vishvakarma" in state.final_answer
    assert "NBC 2016" in state.final_answer
