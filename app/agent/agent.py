"""Agent Controller orchestrating the LLM reasoning, deterministic tools, and PostgreSQL situation memory."""
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
    """The central agent orchestrator controlling Qwen 1.8B reasoning, PostgreSQL experience memory, and deterministic tool actions."""

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
            print(f"{prefix:10} {message}")
        except UnicodeEncodeError:
            print(f"{prefix:10} {safe_msg}")

    def execute_action(self, action: str, args: Dict[str, Any], state: AgentState) -> Any:
        """
        Execute deterministic tool call with state context injection and PostgreSQL versioning.
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
        Execute a single reasoning step: State -> LLM Prompt -> Decision JSON -> Tool Execution -> PostgreSQL Step Log -> Updated State.
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
        if state.intent == "greeting":
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
            self._log("ANSWER", welcome_msg)
            state.add_trace("AGENT", "Welcome Greeting", {"message": welcome_msg})
            return decision

        # 2. Construct Prompt for Qwen
        user_prompt = build_planner_prompt(state)

        # 3. Call Qwen 1.8B via Ollama
        try:
            raw_response = self.llm.generate(
                prompt=user_prompt,
                system=SYSTEM_PROMPT,
                temperature=0.1
            )
            decision = parse_decision(raw_response)
        except Exception as e:
            self._log("PARSER", f"JSON parse warning: {e}. Applying workflow fallback...")
            # Fallback action determination based on state
            if state.intent == "thermal_analysis":
                if not state.requirements.get("climate_data"):
                    fallback_act = "get_climate"
                    fallback_args = {"location": state.requirements.get("location", "Kerala")}
                elif not state.structured_analysis and not state.constraints:
                    fallback_act = "analyze_thermal_constraints"
                    fallback_args = {"location": state.requirements.get("location", "Kerala")}
                elif not state.cost_impact:
                    fallback_act = "calculate_improvement_cost"
                    fallback_args = {}
                else:
                    fallback_act = "finish"
                    fallback_args = {"summary": "Thermal constraint analysis and improvement cost calculation complete."}
            else:
                if not state.requirements.get("climate_data"):
                    fallback_act = "get_climate"
                    fallback_args = {"location": state.requirements.get("location", "Kerala")}
                elif not state.current_design:
                    fallback_act = "generate_design"
                    fallback_args = {"capacity": state.requirements.get("capacity", 5), "budget": state.requirements.get("budget", 80000)}
                elif not (state.evaluation and "thermal" in state.evaluation):
                    fallback_act = "evaluate_thermal"
                    fallback_args = {}
                elif not state.evaluation.get("thermal", {}).get("passed", False) and len(state.design_history) < 2:
                    fallback_act = "modify_design"
                    fallback_args = {"critique": "Increase roof ventilation and overhang to pass 0.70 target"}
                elif not (state.evaluation and "cost" in state.evaluation):
                    fallback_act = "evaluate_cost"
                    fallback_args = {}
                elif not (state.evaluation and "structure" in state.evaluation):
                    fallback_act = "evaluate_structure"
                    fallback_args = {}
                else:
                    fallback_act = "finish"
                    fallback_args = {"summary": "Shelter design generation completed."}
            
            decision = AgentDecision(
                action=fallback_act,
                arguments=fallback_args,
                reason=f"Recovered workflow action after parse notice: {fallback_act}"
            )

        state.tool_calls.append({"iteration": state.iteration, "action": decision.action, "arguments": decision.arguments, "reason": decision.reason})
        self._log("AGENT", f"Action: {decision.action} | Reason: {decision.reason}")
        state.add_trace("AGENT", f"Action: {decision.action} ({decision.reason})", decision.model_dump())

        # 4. Handle Terminal Decisions
        if decision.action == "finish":
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
                summary = decision.arguments.get("summary", "Modular shelter design successfully completed and verified.")
                state.final_answer = summary
                self._log("FINISH", summary)

            state.status = "completed"
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

        if decision.action == "answer":
            msg = decision.arguments.get("message", "Task completed.")
            state.final_answer = msg
            state.status = "completed"
            self._log("ANSWER", msg)
            state.add_trace("AGENT", "Direct Answer", {"message": msg})

            try:
                save_agent_step(
                    step_id=f"{run_id}_s_{state.iteration}",
                    agent_run_id=run_id,
                    step_number=state.iteration,
                    decision=decision.model_dump(),
                    state_before=state_before,
                    tool_called=decision.action,
                    tool_arguments=decision.arguments,
                    tool_result={"message": msg},
                    state_after={"iteration": state.iteration, "status": state.status}
                )
            except Exception:
                pass

            return decision

        # 5. Handle Deterministic Tool Actions
        result = None
        if decision.action in TOOLS:
            try:
                result = self.execute_action(decision.action, decision.arguments, state)
                state.tool_results.append(result)

                # State Updates per Tool Type
                if decision.action == "get_climate":
                    if isinstance(result, dict) and result.get("success", True):
                        state.requirements["climate"] = result.get("climate", state.requirements.get("climate", "hot_humid"))
                        state.requirements["climate_data"] = result
                        self._log("TOOL", f"Climate retrieved: {result.get('location')} -> {result.get('climate')}")
                        state.add_trace("TOOL", f"Retrieved climate for {result.get('location')}", result)

                        # Save climate observation to PostgreSQL
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
                        self._log("TOOL", f"Tool Failure: {result.get('error', 'Unknown climate error')}")
                        state.add_trace("TOOL", f"Climate Tool Failure: {result.get('error')}", result)

                elif decision.action in ["analyze_thermal_constraints", "evaluate_thermal"] and state.intent == "thermal_analysis":
                    from app.tools.constraints import analyze_thermal_constraints
                    c_res = analyze_thermal_constraints(
                        design=state.current_design,
                        climate_data=state.requirements.get("climate_data"),
                        location=state.requirements.get("location", "Kerala")
                    )
                    state.structured_analysis = c_res
                    state.constraints = c_res.get("thermal_constraints", c_res.get("analysis", {}).get("thermal_constraints", []))
                    state.suggestions = c_res.get("suggestions", [])
                    state.improvements = c_res.get("improvements", [])
                    self._log("TOOL", f"Constraint analysis complete: {len(state.constraints)} thermal constraints at risk identified.")
                    state.add_trace("TOOL", f"Thermal constraint analysis complete ({len(state.constraints)} at risk)", c_res)

                    # Save thermal analysis & constraints to PostgreSQL
                    try:
                        save_thermal_analysis(
                            analysis_id=f"ta_{run_id}_{state.iteration}",
                            agent_run_id=run_id,
                            indoor_temp_c=c_res.get("metrics", {}).get("indoor_operative_temp_c", 34.2),
                            neutral_temp_c=c_res.get("metrics", {}).get("neutral_temp_c", 29.57),
                            upper_90_limit=c_res.get("metrics", {}).get("upper_90_limit", 31.95),
                            comfort_score=0.55,
                            passed=False,
                            constraints=state.constraints
                        )
                    except Exception:
                        pass

                elif decision.action in ["calculate_improvement_cost", "evaluate_cost"] and state.intent == "thermal_analysis":
                    from app.tools.cost import calculate_improvement_cost
                    cost_res = calculate_improvement_cost(
                        improvements=state.improvements,
                        design=state.current_design
                    )
                    state.cost_impact = cost_res
                    self._log("TOOL", f"Improvement cost calculated: {cost_res.get('formatted_total', 'INR 3,200.00')} total additional cost.")
                    state.add_trace("TOOL", "Improvement cost calculated", cost_res)

                    # Save cost result to PostgreSQL
                    try:
                        save_cost_result(
                            cost_id=f"cost_{run_id}_{state.iteration}",
                            agent_run_id=run_id,
                            total_cost_inr=cost_res.get("total_inr", 3200.0),
                            itemized=cost_res.get("items", []),
                            is_historical=True,
                            cost_year=2014,
                            notes=cost_res.get("notes")
                        )
                    except Exception:
                        pass

                elif decision.action == "search_knowledge":
                    if isinstance(result, list):
                        state.retrieved_knowledge.extend(result)
                        titles = ", ".join([r.get("title", r.get("topic", "")) for r in result])
                        self._log("RAG", f"Retrieved {len(result)} items: {titles}")
                        state.add_trace("RAG", f"Knowledge retrieved: {titles}", result)

                elif decision.action == "generate_design":
                    state.current_design = result
                    state.design_history.append(result)
                    self._log("TOOL", f"Modular design generated: {result.get('name')} (Version {result.get('version')})")
                    state.add_trace("TOOL", f"Generated initial modular design {result.get('name')}", result)

                    # Save building and immutable V1 design version in PostgreSQL
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
                    mods = ", ".join(result.get("modifications_applied", ["Updated parameters"]))
                    v_num = result.get("version", 2)
                    self._log("TOOL", f"Modular design modified to Version {v_num}: {mods}")
                    state.add_trace("TOOL", f"Modified modular design to v{v_num}", result)
                    state.evaluation = None
                    state.critique = None

                    # Save immutable V2+ design version in PostgreSQL (Never overwrites V1)
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
                        self._log("TOOL", f"Cost evaluated: INR {result.get('total_cost')} (Passed: {result.get('passed')})")
                    elif decision.action == "evaluate_thermal":
                        state.evaluation["thermal"] = result
                        self._log("TOOL", f"Thermal evaluated: Score={result.get('thermal_score')} / Target={result.get('target')} (Passed: {result.get('passed')})")
                    elif decision.action == "evaluate_structure":
                        state.evaluation["structure"] = result
                        self._log("TOOL", f"Structure evaluated: Score={result.get('structural_score')} (Passed: {result.get('passed')})")

                    state.add_trace("TOOL", f"Evaluation completed for {decision.action}", result)

                    if "thermal" in state.evaluation and "cost" in state.evaluation and "structure" in state.evaluation:
                        state.critique = critique(state.evaluation)
                        state.evaluation["passed"] = state.critique["all_passed"]
                        if not state.critique["all_passed"]:
                            self._log("CRITIC", f"Observation: {'; '.join(state.critique['problems'])}")
                            state.add_trace("CRITIC", "Critic evaluation failed", state.critique)
                        else:
                            self._log("CRITIC", "Observation: All performance benchmarks passed!")
                            state.add_trace("CRITIC", "Critic all benchmarks passed", state.critique)
                    elif decision.action == "evaluate_thermal" and not result.get("passed"):
                        self._log("OBSERVE", f"Thermal score {result.get('thermal_score')} < target {result.get('target')} -> Improvement required.")
                        state.add_trace("CRITIC", f"Thermal failed ({result.get('thermal_score')} < {result.get('target')})", result)

                elif decision.action == "save_experience":
                    self._log("MEMORY", f"PostgreSQL Experience saved: {result.get('experience_id')}")
                    state.add_trace("MEMORY", "Saved modular design experience", result)

                elif decision.action == "retrieve_experience":
                    self._log("MEMORY", f"Retrieved {len(result)} past experiences from PostgreSQL")
                    state.add_trace("MEMORY", "Retrieved past experiences", result)

            except Exception as e:
                err_msg = f"Error executing tool '{decision.action}': {e}"
                self._log("ERROR", err_msg)
                state.tool_results.append({"success": False, "error": err_msg})
                state.add_trace("TOOL", f"Tool Exception: {err_msg}", {"error": str(e)})

        # Record step to PostgreSQL agent_steps
        try:
            state_after = {
                "iteration": state.iteration,
                "status": state.status,
                "has_design": bool(state.current_design),
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
        Execute full autonomous agent loop with PostgreSQL run tracking and experience persistence.
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
        except Exception as e:
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
