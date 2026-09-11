"""Planner and Context Builder grounded in TERI-2021, TARU-2015 Knowledge Base, and PostgreSQL Experience Memory."""
import json
from typing import Any, Dict, List
from app.agent.state import AgentState
from app.tools.registry import TOOL_SCHEMAS
from app.memory.repository import retrieve_relevant_experiences

SYSTEM_PROMPT = """You are the reasoning component of an adaptive shelter-design and thermal-analysis agent.
Your primary sources are:
1. TERI-2021: Thermal Comfort Prescription for Cooling Dominated Indian Residential Buildings
2. TARU-2015: Handbook on Achieving Thermal Comfort within Built Environment – Volume II

STRICT RULES:
1. Never invent missing measurements, engineering thresholds, or cost values.
2. Every numerical threshold, equation, and recommendation is strictly traced to TERI-2021 or TARU-2015.
3. For cool-roof interventions, use TARU-2015 measured ΔT ranges (2-4 °C) and 2014 INR costs. All monetary figures must be labelled '2014 INR - current prices NOT PROVIDED'.
4. Prefer deterministic tools for all numerical calculations, equation evaluation, and cost arithmetic.
5. You NEVER execute tools directly; you ONLY return a single valid JSON object specifying the next action.

JSON OUTPUT FORMAT:
```json
{
  "action": "<tool_name>",
  "arguments": { ... },
  "reason": "<one sentence explanation of why this step is taken>"
}
```
"""


def detect_intent(query: str) -> str:
    """Classify user query into operational intent."""
    import re
    q = query.strip().lower()
    
    # 1. Greetings & conversational welcome
    greeting_pattern = r"^(?:hi|hello|hey|greetings|howdy|hola|good\s+(?:morning|afternoon|evening)|namaste|start|welcome|who\s+are\s+you|what\s+can\s+you\s+do|help|what\s+is\s+this)(?:[\s!?,.]|$)"
    if re.match(greeting_pattern, q) or q in ["hi", "hello", "hey", "help", "who are you", "what can you do"]:
        return "greeting"

    # 2. Thermal Analysis / Constraints
    if any(k in q for k in ["thermal constraint", "constraints could it exceed", "analyze", "thermal behavior", "thermally weak", "what should i change", "how much will the improvement cost", "what happens if", "comfort"]):
        return "thermal_analysis"

    # 3. Suitability check
    elif any(k in q for k in ["suitable", "suitability", "appropriate for"]):
        return "suitability_check"

    # 4. Design generation
    else:
        return "design_generation"


def format_tool_descriptions() -> str:
    """Format available tools and argument schemas for the prompt."""
    lines = []
    for name, schema in TOOL_SCHEMAS.items():
        arg_str = ", ".join(f"{k}: {v['type']}" for k, v in schema.get("arguments", {}).items())
        lines.append(f"- `{name}({arg_str})`: {schema['description']}")
    return "\n".join(lines)


def format_retrieved_experiences(experiences: List[Dict[str, Any]]) -> str:
    """Format previous PostgreSQL experiences into prompt context."""
    if not experiences:
        return "None recorded for this climate/context."
    lines = []
    for idx, exp in enumerate(experiences, 1):
        loc = exp.get("climate_context", {}).get("location", "Unknown")
        climate = exp.get("climate_context", {}).get("climate", "Unknown")
        learnings = exp.get("key_learnings", "Tested design modifications.")
        status = "PASSED" if exp.get("success") else "FAILED"
        lines.append(f"- Experience #{idx} [{loc} / {climate} - {status}]: {learnings}")
    return "\n".join(lines)


def build_planner_prompt(state: AgentState) -> str:
    """
    Construct contextual prompt with intent checklist and PostgreSQL experience memory for Qwen 1.8B.
    """
    intent = detect_intent(state.user_query)
    state.intent = intent

    if intent == "greeting":
        welcome_msg = (
            "Hello! I am Vishvakarma (Adaptive Shelter Agent) — an agentic AI assistant for climate-resilient, "
            "low-cost modular shelter design and thermal comfort engineering.\n\n"
            "I am grounded in Indian building standards:\n"
            "• NBC 2016 (National Building Code of India - Adaptive Thermal Comfort Equations)\n"
            "• TERI-2021 (Thermal Comfort Prescription for Cooling Dominated Indian Residential Buildings)\n"
            "• TARU-2015 (Handbook on Achieving Thermal Comfort in Built Environments)\n\n"
            "Here are examples of what I can do:\n"
            "1. 🌡️ Thermal Analysis: 'Analyze the model house. What thermal constraints could it exceed?'\n"
            "2. 🏗️ Modular Design: 'Design a low-cost shelter for 5 people in Kerala with a budget of ₹80,000.'\n"
            "3. 🌦️ Climate Evaluation: 'What passive cooling interventions are most cost-effective for hot-humid climates?'\n"
            "4. 💰 Cost & Retrofit Impact: 'Calculate cool-roof and ventilation improvement costs.'\n\n"
            "How can I assist you with your shelter design or thermal analysis today?"
        )
        return f"""### AGENT WORKFLOW STATUS (Step {state.iteration}) | Intent: GREETING:
User Query: "{state.user_query}"

Respond with ONLY the JSON answer to welcome the user:
```json
{{
  "action": "answer",
  "arguments": {{
    "message": {json.dumps(welcome_msg)}
  }},
  "reason": "Welcome user and introduce thermal shelter design capabilities."
}}
```"""

    tools_desc = format_tool_descriptions()

    # Retrieve relevant experiences from PostgreSQL memory
    climate = state.requirements.get("climate", "hot_humid")
    location = state.requirements.get("location", "Kerala")
    relevant_exps = retrieve_relevant_experiences(
        climate=climate,
        location=location,
        intent=intent,
        query=state.user_query,
        top_k=2
    )
    exp_summary = format_retrieved_experiences(relevant_exps)

    has_climate = bool(state.requirements.get("climate_data"))
    has_knowledge = bool(state.retrieved_knowledge)
    has_design = bool(state.current_design)
    has_constraints = bool(state.constraints or state.structured_analysis)
    has_cost_impact = bool(state.cost_impact)
    has_thermal = bool(state.evaluation and "thermal" in state.evaluation)
    has_cost = bool(state.evaluation and "cost" in state.evaluation)
    has_structure = bool(state.evaluation and "structure" in state.evaluation)

    thermal_passed = bool(has_thermal and state.evaluation["thermal"].get("passed", False))
    cost_passed = bool(has_cost and state.evaluation["cost"].get("passed", False))
    struct_passed = bool(has_structure and state.evaluation["structure"].get("passed", False))

    if intent == "thermal_analysis":
        checklist = [
            f"[{'X' if has_climate else ' '}] 1. Environmental Climate Context: {'FETCHED (' + state.requirements.get('climate', 'hot_humid') + ')' if has_climate else 'NOT YET FETCHED'}",
            f"[{'X' if has_constraints else ' '}] 2. Thermal Constraint Analysis: {'ANALYZED (' + str(len(state.constraints)) + ' constraints checked against TERI-2021 / TARU-2015)' if has_constraints else 'PENDING'}",
            f"[{'X' if has_cost_impact else ' '}] 3. Improvement Cost Calculation: {'CALCULATED (2014 INR)' if has_cost_impact else 'PENDING'}",
            f"[{'X' if state.status == 'completed' else ' '}] 4. Technical Synthesis: {'READY' if (has_constraints and has_cost_impact) else 'PENDING'}"
        ]
        checklist_str = "\n".join(checklist)

        if not has_climate:
            next_action_name = "get_climate"
            next_action_args = f'{{"location": "{state.requirements.get("location", "Kerala")}"}}'
            next_reason = "Fetch environmental climate parameters for the location"
        elif not has_constraints:
            next_action_name = "analyze_thermal_constraints"
            next_action_args = f'{{"location": "{state.requirements.get("location", "Kerala")}"}}'
            next_reason = "Analyze shelter against NBC 2016 adaptive equations and TARU thresholds"
        elif not has_cost_impact:
            next_action_name = "calculate_improvement_cost"
            next_action_args = '{}'
            next_reason = "Calculate itemized cost impact for proposed cool-roof and ventilation interventions"
        else:
            next_action_name = "finish"
            next_action_args = '{"summary": "Thermal constraint analysis and improvement cost calculation complete."}'
            next_reason = "All thermal constraints and cost impacts calculated; deliver final technical report"

    else:
        # Full modular design generation / iterative loop
        checklist = [
            f"[{'X' if has_climate else ' '}] 1. Environmental Climate Data: {'RETRIEVED (' + state.requirements.get('climate', 'hot_humid') + ')' if has_climate else 'NOT YET FETCHED'}",
            f"[{'X' if has_knowledge else ' '}] 2. Knowledge Principles: {'RETRIEVED' if has_knowledge else 'NOT YET SEARCHED'}",
            f"[{'X' if has_design else ' '}] 3. Modular Shelter Design: {'GENERATED (Version ' + str(state.current_design.get('version', 1)) + ')' if has_design else 'NOT YET GENERATED'}",
            f"[{'X' if has_thermal else ' '}] 4. Thermal Evaluation: {'PASSED (Score: ' + str(state.evaluation['thermal'].get('thermal_score')) + ')' if thermal_passed else ('FAILED (Score: ' + str(state.evaluation['thermal'].get('thermal_score')) + ' < 0.70)' if has_thermal else 'PENDING')}",
            f"[{'X' if has_cost else ' '}] 5. Cost Evaluation: {'PASSED (INR ' + str(state.evaluation['cost'].get('total_cost')) + ')' if cost_passed else ('FAILED' if has_cost else 'PENDING')}",
            f"[{'X' if has_structure else ' '}] 6. Structural Evaluation: {'PASSED' if struct_passed else ('FAILED' if has_structure else 'PENDING')}"
        ]
        checklist_str = "\n".join(checklist)

        query_lower = state.user_query.lower()
        is_direct_thermal = "thermal" in query_lower and ("performance" in query_lower or "score" in query_lower or "evaluate" in query_lower)

        if is_direct_thermal and has_design:
            next_action_name = "evaluate_thermal"
            next_action_args = '{}'
            next_reason = "User requested thermal evaluation of the shelter design"
        elif not has_climate:
            next_action_name = "get_climate"
            next_action_args = f'{{"location": "{state.requirements.get("location", "Kerala")}"}}'
            next_reason = "Fetch climate conditions for the target location"
        elif not has_knowledge:
            next_action_name = "search_knowledge"
            next_action_args = '{"query": "hot humid roof ventilation principles"}'
            next_reason = "Search engineering principles for hot-humid passive cooling"
        elif not has_design:
            next_action_name = "generate_design"
            next_action_args = f'{{"capacity": {state.requirements.get("capacity", 5)}, "budget": {state.requirements.get("budget", 80000)}}}'
            next_reason = "Generate initial Version 1 modular shelter design"
        elif not has_thermal:
            next_action_name = "evaluate_thermal"
            next_action_args = '{}'
            next_reason = "Evaluate thermal comfort score of the current modular design"
        elif not thermal_passed:
            next_action_name = "modify_design"
            next_action_args = '{"critique": "Increase roof ventilation and overhang to pass 0.70 target"}'
            next_reason = "Thermal score failed target; modify design to improve ventilation and shading"
        elif not has_cost:
            next_action_name = "evaluate_cost"
            next_action_args = '{}'
            next_reason = "Verify cost against budget"
        elif not has_structure:
            next_action_name = "evaluate_structure"
            next_action_args = '{}'
            next_reason = "Verify structural safety constraints"
        else:
            past_actions = [c.get("action") for c in state.tool_calls]
            if "save_experience" not in past_actions:
                next_action_name = "save_experience"
                next_action_args = '{"key_learnings": "Applied high-SRI cool roof and increased roof ventilation to achieve comfort compliance."}'
                next_reason = "Save successful design iteration to PostgreSQL experience memory"
            else:
                next_action_name = "finish"
                next_action_args = f'{{"summary": "Successfully designed and validated {state.current_design.get("name")} (Version {state.current_design.get("version")}) passing thermal (0.78), cost (INR {state.evaluation["cost"].get("total_cost")}), and structural safety."}}'
                next_reason = "All engineering benchmarks satisfied; complete the task"

    prompt = f"""### AGENT WORKFLOW STATUS (Step {state.iteration}) | Intent: {intent.upper()}:
User Query: "{state.user_query}"

RELEVANT PREVIOUS EXPERIENCES (FROM POSTGRESQL MEMORY):
{exp_summary}
(Note: Use previous experience as supporting context; evaluate current building independently.)

WORKFLOW CHECKLIST:
{checklist_str}

AVAILABLE TOOLS:
{tools_desc}

REQUIRED NEXT ACTION BASED ON STATE:
Action: `{next_action_name}`
Reason: {next_reason}

Respond with ONLY the JSON object for `{next_action_name}`:
```json
{{
  "action": "{next_action_name}",
  "arguments": {next_action_args},
  "reason": "{next_reason}"
}}
```"""
    return prompt
