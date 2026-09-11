"""Agent Controller orchestrating genuine LLM Observe -> Reason -> Act loop, deterministic tools, and PostgreSQL situation memory."""
import re
import sys
import json
import logging
import datetime
from typing import Any, Dict, Optional

from app.llm.ollama_client import OllamaClient
from app.agent.state import AgentState
from app.agent.planner import SYSTEM_PROMPT, build_planner_prompt, detect_intent
from app.agent.parser import parse_decision, AgentDecision
from app.tools.registry import get_tool, TOOLS
from app.evaluation.evaluator import evaluate_all, critique
from app.memory.repository import (
    save_agent_run,
    save_agent_step,
    save_project,
    save_building,
    save_design_version,
    save_experience,
    save_climate_observation,
    save_thermal_analysis,
    save_cost_result
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Vishvakarma")


def extract_initial_requirements(query: str) -> Dict[str, Any]:
    """Extract basic numerical parameters from query to populate initial state."""
    reqs: Dict[str, Any] = {
        "capacity": 5,
        "budget": 80000.0,
        "climate": "hot_humid",
        "location": "Kerala",
        "building_type": "modular_shelter"
    }
    
    # Extract capacity
    cap_match = re.search(r"(\d+)\s*(?:people|persons|occupants|capacity)", query, re.IGNORECASE)
    if cap_match:
        reqs["capacity"] = int(cap_match.group(1))

    # Extract budget
    budget_match = re.search(r"(?:₹|rs\.?|inr|budget\s*(?:of|is)?\s*)?\s*(\d+[\d,]*)(?:\s*(?:inr|rs|rupees))?", query, re.IGNORECASE)
    if budget_match:
        val_str = budget_match.group(1).replace(",", "")
        val = float(val_str)
        if val >= 1000:
            reqs["budget"] = val

    # Extract climate / location
    query_lower = query.lower()
    if "kerala" in query_lower:
        reqs["location"] = "Kerala"
        reqs["climate"] = "hot_humid"
    elif "rajasthan" in query_lower:
        reqs["location"] = "Rajasthan"
        reqs["climate"] = "hot_dry"
    elif "assam" in query_lower:
        reqs["location"] = "Assam"
        reqs["climate"] = "hot_humid"
        reqs["building_type"] = "disaster_relief"
    elif "ladakh" in query_lower:
        reqs["location"] = "Ladakh"
        reqs["climate"] = "cold_arid"

    if "hot-humid" in query_lower or "hot humid" in query_lower:
        reqs["climate"] = "hot_humid"
    elif "hot-arid" in query_lower or "desert" in query_lower or "hot-dry" in query_lower or "hot dry" in query_lower:
        reqs["climate"] = "hot_dry"
    elif "cold" in query_lower:
        reqs["climate"] = "cold"

    if "modular" in query_lower:
        reqs["building_type"] = "modular_shelter"
    elif "disaster" in query_lower or "flood" in query_lower or "relief" in query_lower:
        reqs["building_type"] = "disaster_relief"

    return reqs


class Vishvakarma:
    """
    The central agent orchestrator implementing an autonomous Observe -> Reason -> Act loop.
    Python serves as the execution environment; Qwen 1.8B serves as the reasoning brain.
    """

    def __init__(
        self,
        llm: Optional[OllamaClient] = None,
        max_iterations: int = 10,
        verbose: bool = True
    ):
        self.llm = llm or OllamaClient()
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.active_run_id: Optional[str] = None
        self.active_building_id: Optional[str] = None

    def _log(self, tag: str, message: str, color: str = ""):
        """Formatted console logging for visual agent traces."""
        if not self.verbose:
            return
        prefix = f"[{tag}]"
        safe_msg = message.encode("ascii", errors="backslashreplace").decode("ascii") if sys.platform == "win32" and not sys.stdout.encoding.lower().startswith("utf") else message
        try:
            print(f"{prefix:12} {message}")
        except UnicodeEncodeError:
            print(f"{prefix:12} {safe_msg}")

    def execute_action(self, action: str, args: Dict[str, Any], state: AgentState) -> Any:
        """
        Execute deterministic tool call with contextual parameter fallback from state.
        """
        args_with_context = dict(args)

        if action in ["generate_design"]:
            if "capacity" not in args_with_context and "capacity" in state.requirements:
                args_with_context["capacity"] = state.requirements["capacity"]
            if "budget" not in args_with_context and "budget" in state.requirements:
                args_with_context["budget"] = state.requirements["budget"]
            if "climate" not in args_with_context and "climate" in state.requirements:
                args_with_context["climate"] = state.requirements["climate"]
            if "building_type" not in args_with_context and "building_type" in state.requirements:
                args_with_context["building_type"] = state.requirements["building_type"]

        elif action in ["get_climate"]:
            if "location" not in args_with_context and "location" in state.requirements:
                args_with_context["location"] = state.requirements["location"]

        elif action in ["evaluate_cost"]:
            if "design" not in args_with_context and state.current_design:
                args_with_context["design"] = state.current_design
            if "budget" not in args_with_context and "budget" in state.requirements:
                args_with_context["budget"] = state.requirements["budget"]

        elif action in ["evaluate_thermal"]:
            if "design" not in args_with_context and state.current_design:
                args_with_context["design"] = state.current_design
            if "climate" not in args_with_context and "climate" in state.requirements:
                args_with_context["climate"] = state.requirements["climate"]

        elif action in ["evaluate_structure"]:
            if "design" not in args_with_context and state.current_design:
                args_with_context["design"] = state.current_design

        elif action in ["modify_design"]:
            if "design" not in args_with_context and state.current_design:
                args_with_context["design"] = state.current_design
            if "critique" not in args_with_context and state.critique:
                args_with_context["critique"] = state.critique.get("problems", [])

        elif action in ["analyze_thermal_constraints"]:
            if "design" not in args_with_context and state.current_design:
                args_with_context["design"] = state.current_design
            if "climate_data" not in args_with_context and "climate_data" in state.requirements:
                args_with_context["climate_data"] = state.requirements["climate_data"]
            if "location" not in args_with_context and "location" in state.requirements:
                args_with_context["location"] = state.requirements["location"]

        elif action in ["calculate_improvement_cost"]:
            if "improvements" not in args_with_context and state.improvements:
                args_with_context["improvements"] = state.improvements
            if "design" not in args_with_context and state.current_design:
                args_with_context["design"] = state.current_design

        elif action in ["save_experience"]:
            if "design" not in args_with_context and state.current_design:
                args_with_context["design"] = state.current_design
            if "evaluation" not in args_with_context and state.evaluation:
                args_with_context["evaluation"] = state.evaluation
            if "climate" not in args_with_context and "climate" in state.requirements:
                args_with_context["climate"] = state.requirements["climate"]
            if "location" not in args_with_context and "location" in state.requirements:
                args_with_context["location"] = state.requirements["location"]
            if "capacity" not in args_with_context and "capacity" in state.requirements:
                args_with_context["capacity"] = state.requirements["capacity"]
            if "budget" not in args_with_context and "budget" in state.requirements:
                args_with_context["budget"] = state.requirements["budget"]
            if "user_request" not in args_with_context:
                args_with_context["user_request"] = state.user_query
            if "intent" not in args_with_context:
                args_with_context["intent"] = state.intent
            if "improvements" not in args_with_context and state.improvements:
                args_with_context["improvements"] = state.improvements

        tool_fn = get_tool(action)
        return tool_fn(**args_with_context)

    def run_step(self, state: AgentState) -> AgentDecision:
        """
        Execute a single Observe -> Reason -> Act step:
        1. Contextual Prompt (State + Observations) -> Qwen 1.8B
        2. Qwen decides next action autonomously
        3. Python Environment executes action & records Labeled Observation
        4. State updated for next reasoning step
        """
        state.iteration += 1
        run_id = getattr(state, "run_id", self.active_run_id or f"run_{int(datetime.datetime.now().timestamp())}")
        b_id = self.active_building_id or f"bldg_{run_id}"

        # 1. Capture state before
        state_before = {
            "iteration": state.iteration,
            "status": state.status,
            "intent": state.intent,
            "has_design": bool(state.current_design),
            "design_version": state.current_design.get("version") if state.current_design else None
        }

        # Check for greeting intent
        state.intent = detect_intent(state.user_query)
        if state.intent == "greeting" and state.iteration == 1:
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
            decision = AgentDecision(
                action="answer",
                arguments={"message": welcome_msg},
                reason="Provide welcome greeting and capabilities overview."
            )
            state.final_answer = welcome_msg
            state.status = "completed"
            state.add_observation("agent", "COMPLETED", "Welcome greeting delivered.")
            self._log("ANSWER", welcome_msg)
            state.add_trace("AGENT", "Welcome Greeting", {"message": welcome_msg})
            return decision

        # 2. Build Context for LLM (OBSERVE)
        user_prompt = build_planner_prompt(state)

        # 3. Call LLM for Autonomous Decision (REASON)
        decision: Optional[AgentDecision] = None
        try:
            raw_response = self.llm.generate(
                prompt=user_prompt,
                system=SYSTEM_PROMPT,
                temperature=0.1
            )
            decision = parse_decision(raw_response)
        except Exception as e:
            # When parsing fails, do NOT substitute a hardcoded tool. Return observation to Qwen!
            err_msg = f"Failed to parse decision JSON from model output: {e}"
            self._log("PARSER_ERROR", err_msg)
            state.add_observation(
                source="environment",
                status="INVALID_ACTION",
                content=f"Your previous output was not valid JSON ({e}). Please return a single JSON object specifying action, arguments, and reason. Available tools: {list(TOOLS.keys())} + finish, answer."
            )
            decision = AgentDecision(
                action="invalid_syntax",
                arguments={},
                reason="Invalid JSON syntax from model."
            )

        state.tool_calls.append({
            "iteration": state.iteration,
            "action": decision.action,
            "arguments": decision.arguments,
            "reason": decision.reason
        })
        self._log("QWEN DECISION", f"Action: {decision.action} | Reason: {decision.reason}")
        state.add_trace("AGENT", f"Action: {decision.action} ({decision.reason})", decision.model_dump())

        # 4. Handle Terminal Decisions (finish / answer)
        if decision.action in ["finish", "answer"]:
            if state.intent == "thermal_analysis" and state.structured_analysis:
                from app.agent.formatter import format_technical_thermal_analysis
                formatted_summary = format_technical_thermal_analysis(
                    constraints_report=state.structured_analysis,
                    cost_report=state.cost_impact,
                    design=state.current_design,
                    climate_data=state.requirements.get("climate_data")
                )
                state.final_answer = formatted_summary
                self._log("FINISH", "Thermal analysis technical report generated.")
            else:
                summary = (
                    decision.arguments.get("message")
                    or decision.arguments.get("summary")
                    or "Task completed successfully based on accumulated evidence."
                )
                state.final_answer = summary
                self._log("FINISH", summary)

            state.status = "completed"
            state.add_observation("agent", "COMPLETED", f"Task finalized: {state.final_answer[:120]}...")
            state.add_trace("AGENT", "Task Completed", {"summary": state.final_answer})

            # Save step to PostgreSQL
            try:
                save_agent_step(
                    step_id=f"{run_id}_s_{state.iteration}",
                    agent_run_id=run_id,
                    step_number=state.iteration,
                    decision=decision.model_dump(),
                    state_before=state_before,
                    tool_called=decision.action,
                    tool_arguments=decision.arguments,
                    tool_result={"final_answer": state.final_answer},
                    state_after={"iteration": state.iteration, "status": state.status}
                )
            except Exception:
                pass

            return decision

        # 5. Handle Invalid / Unregistered Action Names (Do not silently replace!)
        if decision.action not in TOOLS:
            if decision.action != "invalid_syntax":
                obs_content = f"Requested tool '{decision.action}' does not exist. Available tools: {list(TOOLS.keys())} + finish, answer. Please reason and select an available tool or finish."
                self._log("ENV_NOTICE", obs_content)
                state.add_observation(source="environment", status="INVALID_ACTION", content=obs_content)
            return decision

        # 6. Execute Deterministic Tool (ACT & OBSERVE)
        result = None
        try:
            result = self.execute_action(decision.action, decision.arguments, state)
            state.tool_results.append(result)
        except Exception as tool_err:
            err_str = str(tool_err)
            self._log("TOOL_ERROR", f"Error in {decision.action}: {err_str}")
            state.add_observation(
                source=decision.action,
                status="TOOL_ERROR",
                content=f"Tool '{decision.action}' execution failed with error: {err_str}. You must decide what to do next (e.g. try another tool, search knowledge, or answer with existing evidence)."
            )
            state.add_trace("TOOL", f"Tool Failure ({decision.action})", {"error": err_str})
            return decision

        # 7. Process Tool Output & Record Labeled Observations
        if decision.action == "get_climate":
            if isinstance(result, dict) and result.get("success", True):
                state.requirements["climate"] = result.get("climate", state.requirements.get("climate", "hot_humid"))
                state.requirements["climate_data"] = result
                obs_text = f"Location: {result.get('location')}, Climate Zone: {result.get('climate')}, Temp: {result.get('temperature')}°C, RH: {result.get('humidity')}%, Wind: {result.get('wind_speed')} m/s, Trm: {result.get('trm')}°C"
                state.add_observation(source="climate_tool", status="OBSERVED", content=obs_text, data=result)
                self._log("OBSERVATION", obs_text)
                state.add_trace("TOOL", f"Retrieved climate for {result.get('location')}", result)

                try:
                    save_climate_observation(
                        obs_id=f"obs_{run_id}_{state.iteration}",
                        agent_run_id=run_id,
                        location=result.get("location", "Unknown"),
                        climate_zone=result.get("climate", "hot_humid"),
                        temperature_c=result.get("temperature"),
                        rh_percent=result.get("humidity"),
                        wind_speed_ms=result.get("wind_speed"),
                        solar_radiation_w_m2=result.get("solar_radiation"),
                        trm_c=result.get("trm")
                    )
                except Exception:
                    pass
            else:
                err_text = result.get("error", "Unknown climate error") if isinstance(result, dict) else "Climate service unavailable"
                obs_text = f"Climate lookup failed: {err_text}. Available alternatives: search_knowledge. You must decide what to do next."
                state.add_observation(source="climate_tool", status="TOOL_ERROR", content=obs_text, data=result)
                self._log("OBSERVATION", obs_text)
                state.add_trace("TOOL", f"Climate Tool Failure: {err_text}", result)

        elif decision.action == "analyze_thermal_constraints":
            state.structured_analysis = result
            state.constraints = result.get("thermal_constraints", result.get("analysis", {}).get("thermal_constraints", []))
            state.suggestions = result.get("suggestions", [])
            state.improvements = result.get("improvements", [])
            constraint_names = [c.get("constraint_id", "") + " (" + c.get("name", "") + ")" for c in state.constraints]
            obs_text = f"Identified {len(state.constraints)} thermal constraints at risk: {', '.join(constraint_names[:3])}"
            state.add_observation(source="constraint_engine", status="INFERRED", content=obs_text, data=result)
            self._log("OBSERVATION", obs_text)
            state.add_trace("TOOL", f"Thermal constraint analysis ({len(state.constraints)} at risk)", result)

            try:
                save_thermal_analysis(
                    analysis_id=f"ta_{run_id}_{state.iteration}",
                    agent_run_id=run_id,
                    indoor_temp_c=result.get("metrics", {}).get("indoor_operative_temp_c", 34.2),
                    neutral_temp_c=result.get("metrics", {}).get("neutral_temp_c", 29.57),
                    upper_90_limit=result.get("metrics", {}).get("upper_90_limit", 31.95),
                    comfort_score=0.55,
                    passed=False,
                    constraints=state.constraints
                )
            except Exception:
                pass

        elif decision.action == "calculate_improvement_cost":
            state.cost_impact = result
            obs_text = f"Total improvement cost: {result.get('formatted_total', 'INR 3,200.00')} (TARU-2015 2014 INR base). Items: {', '.join([it.get('description', '') for it in result.get('items', [])])}"
            state.add_observation(source="cost_tool", status="CALCULATED", content=obs_text, data=result)
            self._log("OBSERVATION", obs_text)
            state.add_trace("TOOL", "Improvement cost calculated", result)

            try:
                save_cost_result(
                    cost_id=f"cost_{run_id}_{state.iteration}",
                    agent_run_id=run_id,
                    total_cost_inr=result.get("total_inr", 3200.0),
                    itemized=result.get("items", []),
                    is_historical=True,
                    cost_year=2014,
                    notes=result.get("notes")
                )
            except Exception:
                pass

        elif decision.action == "search_knowledge":
            if isinstance(result, list):
                state.retrieved_knowledge.extend(result)
                topics = [r.get("title", r.get("topic", "Principle")) for r in result]
                obs_text = f"Retrieved {len(result)} engineering principles: {', '.join(topics)}"
                state.add_observation(source="knowledge_base", status="RETRIEVED", content=obs_text, data=result)
                self._log("OBSERVATION", obs_text)
                state.add_trace("RAG", f"Knowledge retrieved: {', '.join(topics)}", result)

        elif decision.action == "generate_design":
            state.current_design = result
            state.design_history.append(result)
            obs_text = f"Generated {result.get('name')} (v{result.get('version')}), Capacity: {result.get('capacity')}P, Floor area: {result.get('dimensions', {}).get('floor_area_sqm')} m², Openings: {result.get('openings', {}).get('opening_to_floor_ratio', 0.20)*100:.0f}%, Roof vent: {'Yes' if result.get('roof', {}).get('ventilation') else 'No'}"
            state.add_observation(source="shelter_generator", status="CALCULATED", content=obs_text, data=result)
            self._log("OBSERVATION", obs_text)
            state.add_trace("TOOL", f"Generated initial modular design {result.get('name')}", result)

            try:
                save_building(
                    building_id=b_id,
                    name=result.get("name", "Modular-Aid-Shelter-v1"),
                    climate_zone=state.requirements.get("climate", "hot_humid"),
                    building_type=result.get("building_type", "modular_shelter"),
                    geometry=result.get("geometry", {})
                )
                save_design_version(
                    version_id=f"dv_{b_id}_v1",
                    building_id=b_id,
                    version_number=1,
                    parameters=result
                )
            except Exception:
                pass

        elif decision.action == "modify_design":
            state.current_design = result
            state.design_history.append(result)
            v_num = result.get("version", 2)
            mods = ", ".join(result.get("modifications_applied", ["Updated parameters"]))
            obs_text = f"Adapted design to Version {v_num}: {mods}. Openings: {result.get('openings', {}).get('opening_to_floor_ratio')*100:.0f}%, Roof vent: {result.get('roof', {}).get('ventilation')}"
            state.add_observation(source="shelter_modifier", status="CALCULATED", content=obs_text, data=result)
            self._log("OBSERVATION", obs_text)
            state.add_trace("TOOL", f"Modified modular design to v{v_num}", result)
            state.evaluation = None
            state.critique = None

            try:
                save_design_version(
                    version_id=f"dv_{b_id}_v{v_num}",
                    building_id=b_id,
                    version_number=v_num,
                    parent_version_id=f"dv_{b_id}_v{v_num - 1}",
                    parameters=result
                )
            except Exception:
                pass

        elif decision.action in ["evaluate_cost", "evaluate_thermal", "evaluate_structure"]:
            if not state.evaluation:
                state.evaluation = {}

            if decision.action == "evaluate_cost":
                state.evaluation["cost"] = result
                obs_text = f"Cost evaluated: ₹{result.get('total_cost'):,.0f} (Passed: {result.get('passed')})"
                state.add_observation(source="cost_evaluator", status="CALCULATED", content=obs_text, data=result)
                self._log("OBSERVATION", obs_text)
            elif decision.action == "evaluate_thermal":
                state.evaluation["thermal"] = result
                obs_text = f"Thermal comfort score: {result.get('thermal_score')} / 0.70 target. Status: {'PASSED' if result.get('passed') else 'FAILED'}. Est. operative temp: {result.get('estimated_indoor_temp')}°C vs 90% limit {result.get('upper_90_limit')}°C"
                state.add_observation(source="thermal_evaluator", status="CALCULATED", content=obs_text, data=result)
                self._log("OBSERVATION", obs_text)
            elif decision.action == "evaluate_structure":
                state.evaluation["structure"] = result
                obs_text = f"Structural safety score: {result.get('structural_score')}. Status: {'PASSED' if result.get('passed') else 'FAILED'}"
                state.add_observation(source="structure_evaluator", status="CALCULATED", content=obs_text, data=result)
                self._log("OBSERVATION", obs_text)

            state.add_trace("TOOL", f"Evaluation completed for {decision.action}", result)

            if "thermal" in state.evaluation and "cost" in state.evaluation and "structure" in state.evaluation:
                state.critique = critique(state.evaluation)
                state.evaluation["passed"] = state.critique["all_passed"]
                critique_summary = "; ".join(state.critique.get("problems", ["All benchmarks satisfied"]))
                obs_text = f"Comprehensive critique: {'PASSED ALL' if state.critique['all_passed'] else 'ISSUES FOUND: ' + critique_summary}"
                state.add_observation(source="critic_engine", status="INFERRED", content=obs_text, data=state.critique)
                self._log("OBSERVATION", obs_text)

        elif decision.action == "save_experience":
            obs_text = f"Saved design experience episode '{result.get('experience_id', 'exp_live')}' into PostgreSQL long-term memory."
            state.add_observation(source="postgres_memory", status="HISTORICAL", content=obs_text, data=result)
            self._log("OBSERVATION", obs_text)
            state.add_trace("TOOL", "Experience saved to PostgreSQL", result)

        elif decision.action == "retrieve_experience":
            obs_text = f"Retrieved {len(result)} past design experiences from PostgreSQL memory."
            state.add_observation(source="postgres_memory", status="HISTORICAL", content=obs_text, data=result)
            self._log("OBSERVATION", obs_text)
            state.add_trace("TOOL", "Retrieved experiences from PostgreSQL", result)

        # 8. Save Step to PostgreSQL
        try:
            state_after = {
                "iteration": state.iteration,
                "status": state.status,
                "design_version": state.current_design.get("version") if state.current_design else None
            }
            save_agent_step(
                step_id=f"{run_id}_s_{state.iteration}",
                agent_run_id=run_id,
                step_number=state.iteration,
                decision=decision.model_dump(),
                state_before=state_before,
                tool_called=decision.action,
                tool_arguments=decision.arguments,
                tool_result=result,
                state_after=state_after
            )
        except Exception:
            pass

        return decision

    def run(self, user_query: str, max_iterations: Optional[int] = None) -> AgentState:
        """
        Execute full autonomous agent Observe -> Reason -> Act loop.
        """
        limit = max_iterations or self.max_iterations
        reqs = extract_initial_requirements(user_query)
        run_id = f"run_{int(datetime.datetime.now().timestamp())}"
        self.active_run_id = run_id
        self.active_building_id = f"bldg_{run_id}"

        state = AgentState(
            user_query=user_query,
            goal=user_query,
            requirements=reqs,
            iteration=0,
            status="running"
        )
        setattr(state, "run_id", run_id)
        state.add_trace("USER", user_query, {"requirements": reqs})

        # Register initial run in PostgreSQL
        try:
            save_project(project_id="proj_shelter_agent", name="Aid-Centric Modular Shelter Project")
            save_agent_run(
                run_id=run_id,
                project_id="proj_shelter_agent",
                user_query=user_query,
                intent=detect_intent(user_query),
                status="running"
            )
        except Exception:
            pass

        if self.verbose:
            print("\n" + "="*70)
            self._log("USER", user_query)
            print("="*70)

        while state.status == "running" and state.iteration < limit:
            self.run_step(state)

        if state.iteration >= limit and state.status == "running":
            state.status = "max_iterations"
            state.final_answer = "Max iterations limit reached before full validation completion."
            self._log("SYSTEM", f"Agent halted after reaching maximum iterations limit ({limit}).")
            state.add_trace("SYSTEM", f"Max iterations reached ({limit})")

        # Finalize run in PostgreSQL and save experience episode
        try:
            save_agent_run(
                run_id=run_id,
                user_query=user_query,
                intent=state.intent,
                status=state.status,
                total_steps=state.iteration,
                final_response=state.final_answer
            )

            is_success = (state.status == "completed")
            save_experience(
                exp_id=f"exp_{run_id}",
                project_id="proj_shelter_agent",
                building_id=self.active_building_id,
                agent_run_id=run_id,
                user_request=user_query,
                intent=state.intent,
                climate_context=state.requirements,
                initial_design=state.design_history[0] if state.design_history else state.current_design,
                thermal_result=state.evaluation.get("thermal") if state.evaluation else None,
                constraints_detected=state.constraints,
                actions_taken=[c.get("action") for c in state.tool_calls],
                improvements=state.improvements,
                final_design=state.current_design,
                final_evaluation=state.evaluation,
                cost_impact=state.cost_impact or (state.evaluation.get("cost") if state.evaluation else None),
                success=is_success,
                failure_reason=None if is_success else "Halted before full validation",
                key_learnings=f"Completed {state.intent} for {state.requirements.get('location')} ({state.requirements.get('climate')})."
            )
        except Exception:
            pass

        if self.verbose:
            print("="*70)
            print(f"Status: {state.status.upper()} | Total Steps: {state.iteration}")
            if state.current_design:
                print(f"Final Modular Design: {state.current_design.get('name')} (v{state.current_design.get('version')})")
            print("="*70 + "\n")

        return state


# Backward compatibility alias
ShelterAgent = Vishvakarma
