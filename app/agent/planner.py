"""Planner and Context Builder providing state, observations, and tools for genuine LLM reasoning."""
import json
import re
from typing import Any, Dict, List
from app.agent.state import AgentState
from app.tools.registry import TOOL_SCHEMAS
from app.memory.repository import retrieve_relevant_experiences

SYSTEM_PROMPT = """You are Vishvakarma, an autonomous reasoning agent for climate-resilient modular shelter design and thermal engineering in India.

You interact with an external environment by making decisions. On each turn, you must return EXACTLY ONE valid JSON object of type 'tool' OR type 'final'.

DECISION FORMAT 1 — INVOKE A TOOL:
Use this when you need to fetch information, perform calculations, evaluate models, or search knowledge.
```json
{
  "type": "tool",
  "tool_name": "get_climate",
  "arguments": {"location": "Kerala"},
  "reason": "Climate conditions are required for thermal engineering analysis."
}
```

DECISION FORMAT 2 — RETURN FINAL ANSWER:
Use this when you have sufficient evidence to answer, or when no tools are required (e.g. general math, definitions, or finished reports).
```json
{
  "type": "final",
  "content": "Your complete technical answer or final response goes here."
}
```

OPERATIONAL RULES:
1. ONLY use registered tool names in tool_name. Never invent tool names.
2. Never output placeholder names like <tool_name>, <finish tool name>, or <answer tool name>.
3. If sufficient evidence already exists or no tool is needed, return type 'final'.
4. NO EVIDENCE -> NO CLAIM: Never invent measurements, thresholds, or costs. If data is unknown, retrieve it or state that it is unavailable.
5. All cool-roof costs are based on TARU-2015 2014 INR rates.
"""


def detect_intent(query: str) -> str:
    """Classify user query into operational intent."""
    q = query.strip().lower()
    
    # 1. Greetings & conversational welcome
    greeting_pattern = r"^(?:hi|hello|hey|greetings|howdy|hola|good\s+(?:morning|afternoon|evening)|namaste|start|welcome|who\s+are\s+you|what\s+can\s+you\s+do|help|what\s+is\s+this)(?:[\s!?,.]|$)"
    if re.match(greeting_pattern, q) or q in ["hi", "hello", "hey", "help", "who are you", "what can you do"]:
        return "greeting"

    # 2. General / Out of domain queries (Math, General knowledge, Python, etc.)
    # Check for basic arithmetic like "1 + 1", "2*5", "sqrt(16)"
    if re.match(r"^[\d\s+\-*/().^=]+$", q) or any(q.startswith(p) for p in ["what is python", "what is 2", "calculate 1", "who was", "who is"]):
        if not any(k in q for k in ["shelter", "thermal", "climate", "comfort", "teri", "taru", "nbc", "roof", "insulation", "cost"]):
            return "general_query"

    # 3. Visualization / Graph Request
    if any(k in q for k in ["graph", "plot", "dashboard", "visualize", "visualization", "show chart", "show figure", "show me the"]):
        if any(k in q for k in ["graph", "plot", "dashboard", "visual", "figure", "chart"]):
            return "visualization_request"

    # 4. Pure Knowledge Queries
    if any(k in q for k in ["what is thermal comfort", "define thermal comfort", "explain nbc 2016", "what is teri", "explain adaptive comfort", "what is cool roof"]):
        return "knowledge_query"

    # 5. Simple Calculation / Cost Query
    if any(k in q for k in ["calculate cost", "cost of", "how much will", "improvement cost", "cost impact"]) and not any(k in q for k in ["design a", "create a"]):
        return "calculation_query"

    # 6. Thermal Analysis / Diagnostic
    if any(k in q for k in ["thermal constraint", "constraints could it exceed", "analyze", "thermal behavior", "thermally weak", "what should i change", "what happens if", "comfort"]):
        return "thermal_analysis"

    # 7. Suitability check
    if any(k in q for k in ["suitable", "suitability", "appropriate for"]):
        return "suitability_check"

    # 8. Design generation
    if any(k in q for k in ["design", "shelter", "modular", "house", "building", "roof", "kerala", "assam", "rajasthan"]):
        return "design_generation"

    # 9. General fallback for other queries
    return "general_query"


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
        return "None recorded for this context."
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
        return "None yet (Initial Step)."
    
    lines = []
    for obs in state.observations:
        lines.append(f"- [Step {obs.step}] [SOURCE: {obs.source} | STATUS: {obs.status}] {obs.content}")
    return "\n".join(lines)


def build_state_summary(state: AgentState) -> str:
    """Build a concise, objective summary of what is known vs missing without prescribing actions."""
    if state.intent == "general_query":
        return "- Context: General Query (No shelter design state required unless requested)"

    lines = []
    # Building State
    if state.current_design:
        d = state.current_design
        lines.append(f"- Building Design: AVAILABLE ({d.get('name', 'Modular-Shelter')} v{d.get('version', 1)}, Capacity: {d.get('capacity', 5)} occupants, Budget: ₹{d.get('budget', 80000):,.0f}, Floor: {d.get('dimensions', {}).get('floor_area_sqm', 24)} m², Wall Opening: {d.get('openings', {}).get('opening_to_floor_ratio', 0.20)*100:.0f}%, Roof Vent: {'Yes' if d.get('roof', {}).get('ventilation') else 'No'})")
    else:
        lines.append("- Building Design: MISSING / NOT YET SPECIFIED")

    # Climate State
    if state.requirements.get("climate_data"):
        c = state.requirements["climate_data"]
        lines.append(f"- Environmental Climate: AVAILABLE (Location: {c.get('location', 'Kerala')}, Zone: {c.get('climate', 'hot_humid')}, Temp: {c.get('temperature', 33.0)}°C, RH: {c.get('humidity', 82.0)}%, Wind: {c.get('wind_speed', 2.8)} m/s, Trm: {c.get('trm', 32.0)}°C)")
    elif state.requirements.get("location"):
        lines.append(f"- Environmental Climate: NOT YET RETRIEVED for '{state.requirements.get('location')}'")
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
    Construct contextual Observe -> Reason prompt for Qwen 1.8B without placeholder syntax.
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

Respond with ONLY this JSON final decision:
```json
{{
  "type": "final",
  "content": {json.dumps(welcome_msg)}
}}
```"""

    if intent == "general_query":
        return f"""================================================================================
USER REQUEST:
"{state.user_query}"
================================================================================

This request is a general question or calculation.
If no shelter tools are needed, answer directly using type 'final'.

Respond with ONLY a single valid JSON object:
```json
{{
  "type": "final",
  "content": "Your direct answer here."
}}
```"""

    tools_desc = format_tool_descriptions()
    state_summary = build_state_summary(state)
    obs_summary = format_observations(state)

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

AVAILABLE REGISTERED TOOLS:
{tools_desc}

DECIDE THE NEXT ACTION:
Carefully inspect the USER REQUEST, CURRENT SITUATION, and PREVIOUS OBSERVATIONS.
- If you need to fetch data or run engineering calculations, output type 'tool' with a valid tool_name from the registered tools list.
- If you have gathered sufficient evidence to fulfill the user request, output type 'final' with your complete technical response in 'content'.
- Do NOT call unnecessary tools.
- Strict Rule: NO EVIDENCE -> NO CLAIM. Never fabricate missing values.

Respond with ONLY a single valid JSON object in one of the two formats:
```json
{{
  "type": "tool",
  "tool_name": "get_climate",
  "arguments": {{"location": "Kerala"}},
  "reason": "Need climate observations before running thermal evaluation."
}}
```
OR
```json
{{
  "type": "final",
  "content": "Final technical report or response."
}}
```"""
    return prompt
