"""Planner and Context Builder providing state, observations, and tools for genuine LLM reasoning."""
import json
from typing import Any, Dict, List
from app.agent.state import AgentState
from app.tools.registry import TOOL_SCHEMAS
from app.memory.repository import retrieve_relevant_experiences

SYSTEM_PROMPT = """You are Vishvakarma, an autonomous reasoning agent for climate-resilient modular shelter design and thermal comfort engineering in India.
Your engineering foundations are:
1. NBC 2016: National Building Code of India (Adaptive Thermal Comfort Equations)
2. TERI-2021: Thermal Comfort Prescription for Cooling Dominated Indian Residential Buildings
3. TARU-2015: Handbook on Achieving Thermal Comfort within Built Environment – Vol II

CRITICAL OPERATIONAL RULES:
1. NO EVIDENCE -> NO CLAIM: Never invent measurements, environmental parameters, engineering thresholds, or cost figures. If a value is unknown, retrieve it with a tool, calculate it, or state that it is unavailable.
2. YOU ARE THE BRAIN: Python is only your execution environment. You independently inspect what information you have, what is missing, and what action to take next.
3. CONDITIONAL TOOL CALLING: Only invoke tools that are strictly necessary for the user's specific request. (e.g. For pure knowledge questions, search knowledge or answer; do not call climate/thermal tools unnecessarily).
4. INTERPRET OBSERVATIONS: On each step, review previous observations and their source/status before making your next decision.
5. HANDLE TOOL ERRORS & INVALID ACTIONS: If a tool fails or an action was invalid, inspect the error observation and choose an alternative or answer.
6. COMPLETION: When you have sufficient evidence to answer or fulfill the request, choose action 'finish' or 'answer'.

DECISION OUTPUT FORMAT:
Respond with ONLY a single valid JSON object.
Format A (Tool invocation):
```json
{
  "action": "<tool_name>",
  "arguments": { "<arg_name>": <value> },
  "reason": "<one sentence explaining why this tool is needed given current evidence>"
}
```
Format B (Final answer / Completion):
```json
{
  "action": "answer",
  "arguments": { "message": "<your technical answer or summary>" },
  "reason": "<explanation of why evidence is sufficient to answer>"
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

    # 2. Pure Knowledge Queries
    if any(k in q for k in ["what is thermal comfort", "define thermal comfort", "explain nbc 2016", "what is teri", "explain adaptive comfort", "what is cool roof"]):
        return "knowledge_query"

    # 3. Simple Calculation / Cost Query
    if any(k in q for k in ["calculate cost", "cost of", "how much will", "improvement cost", "cost impact"]) and not any(k in q for k in ["design a", "create a"]):
        return "calculation_query"

    # 4. Thermal Analysis / Diagnostic
    if any(k in q for k in ["thermal constraint", "constraints could it exceed", "analyze", "thermal behavior", "thermally weak", "what should i change", "what happens if", "comfort"]):
        return "thermal_analysis"

    # 5. Suitability check
    elif any(k in q for k in ["suitable", "suitability", "appropriate for"]):
        return "suitability_check"

    # 6. Design generation
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
        lines.append(f"- [SOURCE: postgres_memory | STATUS: HISTORICAL] Experience #{idx} [{loc} / {climate} - {status}]: {learnings}")
    return "\n".join(lines)


def format_observations(state: AgentState) -> str:
    """Format labeled chronological observations for Qwen."""
    if not state.observations:
        return "None yet (Initial Reasoning Step)."
    
    lines = []
    for obs in state.observations:
        lines.append(f"- [Step {obs.step}] [SOURCE: {obs.source} | STATUS: {obs.status}] {obs.content}")
    return "\n".join(lines)


def build_state_summary(state: AgentState) -> str:
    """Build a concise, objective summary of what is known vs missing without prescribing actions."""
    lines = []
    
    # Building State
    if state.current_design:
        d = state.current_design
        lines.append(f"- Building Design: AVAILABLE ({d.get('name', 'Modular-Shelter')} v{d.get('version', 1)}, Capacity: {d.get('capacity', 5)} occupants, Budget: ₹{d.get('budget', 80000):,.0f}, Floor: {d.get('dimensions', {}).get('floor_area_sqm', 24)} m², Wall Opening: {d.get('openings', {}).get('opening_to_floor_ratio', 0.20)*100:.0f}%, Roof Vent: {'Yes' if d.get('roof', {}).get('ventilation') else 'No'})")
    else:
        lines.append("- Building Design: MISSING / NOT YET GENERATED")

    # Climate State
    if state.requirements.get("climate_data"):
        c = state.requirements["climate_data"]
        lines.append(f"- Environmental Climate: AVAILABLE (Location: {c.get('location', 'Kerala')}, Zone: {c.get('climate', 'hot_humid')}, Temp: {c.get('temperature', 33.0)}°C, RH: {c.get('humidity', 82.0)}%, Wind: {c.get('wind_speed', 2.8)} m/s, Trm: {c.get('trm', 32.0)}°C)")
    elif state.requirements.get("location"):
        lines.append(f"- Environmental Climate: PENDING FETCH for '{state.requirements.get('location')}'")
    else:
        lines.append("- Environmental Climate: NOT YET RETRIEVED")

    # Thermal Performance
    if state.evaluation and "thermal" in state.evaluation:
        t = state.evaluation["thermal"]
        status_str = "PASSED" if t.get("passed") else "FAILED"
        lines.append(f"- Thermal Evaluation: CALCULATED (Score: {t.get('thermal_score', 0.0):.2f} / Target 0.70, Status: {status_str}, Est. Indoor Temp: {t.get('estimated_indoor_temp', 'N/A')}°C)")
    else:
        lines.append("- Thermal Evaluation: NOT YET EVALUATED")

    # Thermal Constraints
    if state.constraints or state.structured_analysis:
        n_c = len(state.constraints)
        lines.append(f"- Thermal Constraints: ANALYZED ({n_c} constraints evaluated against NBC 2016 / TERI-2021)")
    else:
        lines.append("- Thermal Constraints: NOT YET ANALYZED")

    # Cost / Improvement Analysis
    if state.cost_impact:
        ci = state.cost_impact
        lines.append(f"- Improvement Cost: CALCULATED ({ci.get('formatted_total', 'INR 3,200.00')}, 2014 INR base)")
    elif state.evaluation and "cost" in state.evaluation:
        c_eval = state.evaluation["cost"]
        lines.append(f"- Construction Cost: CALCULATED (Total: ₹{c_eval.get('total_cost', 0):,.0f}, Passed: {c_eval.get('passed', False)})")
    else:
        lines.append("- Cost / Improvement: NOT YET CALCULATED")

    # Structural Safety
    if state.evaluation and "structure" in state.evaluation:
        s_eval = state.evaluation["structure"]
        lines.append(f"- Structural Safety: CHECKED (Passed: {s_eval.get('passed', False)})")
    else:
        lines.append("- Structural Safety: NOT YET CHECKED")

    # Retrieved Knowledge
    if state.retrieved_knowledge:
        topics = ", ".join([k.get("topic", k.get("title", "Topic")) for k in state.retrieved_knowledge[:3]])
        lines.append(f"- Engineering Knowledge: RETRIEVED ({len(state.retrieved_knowledge)} items: {topics})")
    else:
        lines.append("- Engineering Knowledge: NOT SEARCHED")

    return "\n".join(lines)


def build_planner_prompt(state: AgentState) -> str:
    """
    Construct contextual Observe -> Reason prompt for Qwen 1.8B without any hardcoded checklists.
    """
    intent = detect_intent(state.user_query)
    state.intent = intent

    if intent == "greeting":
        welcome_msg = (
            "Hello! I am Vishvakarma (Adaptive Shelter Agent) — an agentic AI assistant for climate-resilient, "
            "low-cost modular shelter design and thermal comfort engineering.\n\n"
            "I am strictly grounded in Indian building standards:\n"
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
        return f"""### AGENT STATUS (Step {state.iteration}) | Intent: GREETING:
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
    state_summary = build_state_summary(state)
    obs_summary = format_observations(state)

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

    prompt = f"""================================================================================
USER REQUEST:
"{state.user_query}"
================================================================================

CURRENT SITUATION & STATE (Step {state.iteration}):
{state_summary}

PREVIOUS OBSERVATIONS & EVIDENCE:
{obs_summary}

RELEVANT PREVIOUS EXPERIENCES (FROM POSTGRESQL MEMORY):
{exp_summary}

AVAILABLE TOOLS:
{tools_desc}

DECIDE THE NEXT ACTION:
Carefully inspect the USER REQUEST, CURRENT SITUATION, and PREVIOUS OBSERVATIONS.
- If you need additional data or calculations to fulfill the user request, choose the appropriate tool and provide its arguments.
- If you have sufficient evidence to complete the task or answer the question, select action 'finish' (or 'answer') and provide the final technical response.
- Do NOT call unnecessary tools.
- Strict Rule: NO EVIDENCE -> NO CLAIM. Never fabricate missing values.

Respond with ONLY a single valid JSON object for your chosen action:
```json
{{
  "action": "<tool_name or finish or answer>",
  "arguments": {{ ... }},
  "reason": "<why this action was chosen based on current evidence>"
}}
```"""
    return prompt
