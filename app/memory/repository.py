"""
Dedicated Database Repository Layer for PostgreSQL Experience Memory.
Provides structured CRUD and similarity retrieval without scattering raw SQL.
"""
import datetime
import json
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc, or_, and_

from app.memory.database import get_db, init_db
from app.memory.models import (
    Project, Building, DesignVersion, AgentRun, AgentStep,
    ClimateObservation, ThermalAnalysis, ThermalConstraintRecord,
    ImprovementRecord, CostResultRecord, ExperienceRecord
)


def ensure_db_initialized():
    """Ensure all database tables exist."""
    init_db()


def save_project(
    project_id: str,
    name: str,
    description: Optional[str] = None,
    session: Optional[Session] = None
) -> Dict[str, Any]:
    """Create or update a Project."""
    def _op(db: Session):
        proj = db.query(Project).filter(Project.id == project_id).first()
        if not proj:
            proj = Project(id=project_id, name=name, description=description)
            db.add(proj)
        else:
            proj.name = name
            proj.description = description
        db.flush()
        return {"id": proj.id, "name": proj.name}

    if session:
        return _op(session)
    with get_db() as db:
        return _op(db)


def save_building(
    building_id: str,
    name: str,
    climate_zone: str,
    project_id: Optional[str] = None,
    building_type: str = "modular_shelter",
    geometry: Optional[Dict[str, Any]] = None,
    session: Optional[Session] = None
) -> Dict[str, Any]:
    """Create or update a Building instance."""
    def _op(db: Session):
        b = db.query(Building).filter(Building.id == building_id).first()
        if not b:
            b = Building(
                id=building_id,
                project_id=project_id,
                name=name,
                building_type=building_type,
                climate_zone=climate_zone,
                geometry=geometry or {}
            )
            db.add(b)
        else:
            b.name = name
            b.climate_zone = climate_zone
            b.building_type = building_type
            if geometry:
                b.geometry = geometry
        db.flush()
        return {"id": b.id, "name": b.name, "building_type": b.building_type}

    if session:
        return _op(session)
    with get_db() as db:
        return _op(db)


def save_design_version(
    version_id: str,
    building_id: str,
    version_number: int,
    parameters: Dict[str, Any],
    parent_version_id: Optional[str] = None,
    session: Optional[Session] = None
) -> Dict[str, Any]:
    """Save an immutable design version. Never overwrites prior versions."""
    def _op(db: Session):
        existing = db.query(DesignVersion).filter(
            DesignVersion.building_id == building_id,
            DesignVersion.version_number == version_number
        ).first()
        if existing:
            return {"id": existing.id, "version_number": existing.version_number, "is_new": False}

        dv = DesignVersion(
            id=version_id,
            building_id=building_id,
            version_number=version_number,
            parent_version_id=parent_version_id,
            parameters=parameters
        )
        db.add(dv)
        db.flush()
        return {"id": dv.id, "version_number": dv.version_number, "is_new": True}

    if session:
        return _op(session)
    with get_db() as db:
        return _op(db)


def save_agent_run(
    run_id: str,
    user_query: str,
    intent: str,
    status: str = "running",
    project_id: Optional[str] = None,
    building_id: Optional[str] = None,
    total_steps: int = 0,
    final_response: Optional[str] = None,
    session: Optional[Session] = None
) -> Dict[str, Any]:
    """Record or update an agent execution run."""
    def _op(db: Session):
        run = db.query(AgentRun).filter(AgentRun.id == run_id).first()
        if not run:
            run = AgentRun(
                id=run_id,
                project_id=project_id,
                building_id=building_id,
                user_query=user_query,
                intent=intent,
                status=status,
                total_steps=total_steps,
                final_response=final_response
            )
            db.add(run)
        else:
            run.status = status
            run.total_steps = total_steps
            run.final_response = final_response
        db.flush()
        return {"id": run.id, "status": run.status}

    if session:
        return _op(session)
    with get_db() as db:
        return _op(db)


def save_agent_step(
    step_id: str,
    agent_run_id: str,
    step_number: int,
    decision: Dict[str, Any],
    state_before: Optional[Dict[str, Any]] = None,
    tool_called: Optional[str] = None,
    tool_arguments: Optional[Dict[str, Any]] = None,
    tool_result: Optional[Any] = None,
    state_after: Optional[Dict[str, Any]] = None,
    session: Optional[Session] = None
) -> Dict[str, Any]:
    """Record an individual agent reasoning and tool execution step."""
    def _op(db: Session):
        existing = db.query(AgentStep).filter(
            AgentStep.agent_run_id == agent_run_id,
            AgentStep.step_number == step_number
        ).first()
        if existing:
            return {"id": existing.id, "step_number": existing.step_number}

        step = AgentStep(
            id=step_id,
            agent_run_id=agent_run_id,
            step_number=step_number,
            state_before=state_before,
            decision=decision,
            tool_called=tool_called,
            tool_arguments=tool_arguments,
            tool_result=tool_result if isinstance(tool_result, (dict, list)) else {"value": str(tool_result)},
            state_after=state_after
        )
        db.add(step)
        db.flush()
        return {"id": step.id, "step_number": step.step_number}

    if session:
        return _op(session)
    with get_db() as db:
        return _op(db)


def save_climate_observation(
    obs_id: str,
    agent_run_id: str,
    location: str,
    climate_zone: str,
    temperature_c: Optional[float] = None,
    rh_percent: Optional[float] = None,
    wind_speed_ms: Optional[float] = None,
    solar_radiation_w_m2: Optional[float] = None,
    trm_c: Optional[float] = None,
    session: Optional[Session] = None
) -> Dict[str, Any]:
    """Save an environmental observation."""
    def _op(db: Session):
        obs = ClimateObservation(
            id=obs_id,
            agent_run_id=agent_run_id,
            location=location,
            climate_zone=climate_zone,
            temperature_c=temperature_c,
            rh_percent=rh_percent,
            wind_speed_ms=wind_speed_ms,
            solar_radiation_w_m2=solar_radiation_w_m2,
            trm_c=trm_c
        )
        db.add(obs)
        db.flush()
        return {"id": obs.id, "location": obs.location}

    if session:
        return _op(session)
    with get_db() as db:
        return _op(db)


def save_thermal_analysis(
    analysis_id: str,
    agent_run_id: str,
    design_version_id: Optional[str] = None,
    indoor_temp_c: Optional[float] = None,
    neutral_temp_c: Optional[float] = None,
    upper_90_limit: Optional[float] = None,
    lower_90_limit: Optional[float] = None,
    comfort_score: Optional[float] = None,
    passed: bool = False,
    constraints: Optional[List[Dict[str, Any]]] = None,
    session: Optional[Session] = None
) -> Dict[str, Any]:
    """Save a thermal analysis record and its evaluated constraints."""
    def _op(db: Session):
        ta = ThermalAnalysis(
            id=analysis_id,
            agent_run_id=agent_run_id,
            design_version_id=design_version_id,
            indoor_temp_c=indoor_temp_c,
            neutral_temp_c=neutral_temp_c,
            upper_90_limit=upper_90_limit,
            lower_90_limit=lower_90_limit,
            comfort_score=comfort_score,
            passed=passed
        )
        db.add(ta)

        if constraints:
            for idx, c in enumerate(constraints, 1):
                tc_rec = ThermalConstraintRecord(
                    id=f"{analysis_id}_c_{idx}",
                    thermal_analysis_id=analysis_id,
                    constraint_id=c.get("constraint_id", f"TC-{idx:03d}"),
                    parameter=c.get("parameter", "unknown"),
                    actual_value=str(c.get("actual_value")),
                    threshold_value=str(c.get("threshold")),
                    status=c.get("status", "UNKNOWN"),
                    reason=c.get("reason", ""),
                    source=c.get("source", "TERI-2021 / TARU-2015")
                )
                db.add(tc_rec)

        db.flush()
        return {"id": ta.id, "passed": ta.passed}

    if session:
        return _op(session)
    with get_db() as db:
        return _op(db)


def save_improvement(
    imp_id: str,
    agent_run_id: str,
    parameter: str,
    current_value: Optional[str] = None,
    recommended_value: Optional[str] = None,
    expected_effect: Optional[str] = None,
    reason: Optional[str] = None,
    source: Optional[str] = None,
    session: Optional[Session] = None
) -> Dict[str, Any]:
    """Save an improvement recommendation."""
    def _op(db: Session):
        imp = ImprovementRecord(
            id=imp_id,
            agent_run_id=agent_run_id,
            parameter=parameter,
            current_value=current_value,
            recommended_value=recommended_value,
            expected_effect=expected_effect,
            reason=reason,
            source=source
        )
        db.add(imp)
        db.flush()
        return {"id": imp.id, "parameter": imp.parameter}

    if session:
        return _op(session)
    with get_db() as db:
        return _op(db)


def save_cost_result(
    cost_id: str,
    agent_run_id: str,
    total_cost_inr: float,
    itemized: List[Dict[str, Any]],
    is_historical: bool = True,
    cost_year: int = 2014,
    notes: Optional[str] = None,
    session: Optional[Session] = None
) -> Dict[str, Any]:
    """Save cost evaluation record."""
    def _op(db: Session):
        cr = CostResultRecord(
            id=cost_id,
            agent_run_id=agent_run_id,
            total_cost_inr=total_cost_inr,
            itemized=itemized,
            is_historical=is_historical,
            cost_year=cost_year,
            notes=notes
        )
        db.add(cr)
        db.flush()
        return {"id": cr.id, "total_cost_inr": cr.total_cost_inr}

    if session:
        return _op(session)
    with get_db() as db:
        return _op(db)


def save_experience(
    exp_id: str,
    user_request: str,
    intent: str,
    climate_context: Dict[str, Any],
    success: bool,
    project_id: Optional[str] = None,
    building_id: Optional[str] = None,
    agent_run_id: Optional[str] = None,
    initial_design: Optional[Dict[str, Any]] = None,
    thermal_result: Optional[Dict[str, Any]] = None,
    constraints_detected: Optional[List[Any]] = None,
    actions_taken: Optional[List[Any]] = None,
    improvements: Optional[List[Any]] = None,
    final_design: Optional[Dict[str, Any]] = None,
    final_evaluation: Optional[Dict[str, Any]] = None,
    cost_impact: Optional[Dict[str, Any]] = None,
    failure_reason: Optional[str] = None,
    key_learnings: Optional[str] = None,
    session: Optional[Session] = None
) -> Dict[str, Any]:
    """
    Save complete episodic experience record to PostgreSQL long-term memory.
    Stores both successful and failed episodes for adaptive retrieval.
    """
    def _op(db: Session):
        existing = db.query(ExperienceRecord).filter(ExperienceRecord.id == exp_id).first()
        if existing:
            existing.user_request = user_request
            existing.intent = intent
            existing.climate_context = climate_context
            existing.initial_design = initial_design
            existing.thermal_result = thermal_result
            existing.constraints_detected = constraints_detected
            existing.actions_taken = actions_taken
            existing.improvements = improvements
            existing.final_design = final_design
            existing.final_evaluation = final_evaluation
            existing.cost_impact = cost_impact
            existing.success = success
            existing.failure_reason = failure_reason
            existing.key_learnings = key_learnings
            existing.created_at = datetime.datetime.now(datetime.timezone.utc)
            db.flush()
            return {"id": existing.id, "success": existing.success, "is_new": False}

        exp = ExperienceRecord(
            id=exp_id,
            project_id=project_id,
            building_id=building_id,
            agent_run_id=agent_run_id,
            user_request=user_request,
            intent=intent,
            climate_context=climate_context,
            initial_design=initial_design,
            thermal_result=thermal_result,
            constraints_detected=constraints_detected,
            actions_taken=actions_taken,
            improvements=improvements,
            final_design=final_design,
            final_evaluation=final_evaluation,
            cost_impact=cost_impact,
            success=success,
            failure_reason=failure_reason,
            key_learnings=key_learnings
        )
        db.add(exp)
        db.flush()
        return {"id": exp.id, "success": exp.success, "is_new": True}

    if session:
        return _op(session)
    with get_db() as db:
        return _op(db)


def retrieve_relevant_experiences(
    climate: Optional[str] = None,
    location: Optional[str] = None,
    building_type: Optional[str] = None,
    intent: Optional[str] = None,
    query: Optional[str] = None,
    success_only: bool = False,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant past experiences from PostgreSQL long-term memory matching
    the current climate context, building type, or intent.
    """
    ensure_db_initialized()
    with get_db() as db:
        q = db.query(ExperienceRecord)

        if success_only:
            q = q.filter(ExperienceRecord.success == True)

        if intent:
            q = q.filter(ExperienceRecord.intent == intent)

        experiences = q.order_by(desc(ExperienceRecord.created_at)).all()

        scored_results = []
        for exp in experiences:
            score = 0
            c_ctx = exp.climate_context or {}
            exp_climate = str(c_ctx.get("climate", "")).lower()
            exp_loc = str(c_ctx.get("location", "")).lower()

            if climate and (climate.lower() in exp_climate or exp_climate in climate.lower()):
                score += 5
            if location and (location.lower() in exp_loc or exp_loc in location.lower()):
                score += 4

            if query:
                q_words = set(query.lower().split())
                req_words = set(exp.user_request.lower().split())
                overlap = len(q_words.intersection(req_words))
                score += overlap * 2

            score += 1
            created_ts = exp.created_at.timestamp() if exp.created_at else 0.0
            scored_results.append(((score, created_ts), exp))

        scored_results.sort(key=lambda x: x[0], reverse=True)
        top_records = [item for _, item in scored_results[:top_k]]

        results = []
        for r in top_records:
            results.append({
                "experience_id": r.id,
                "user_request": r.user_request,
                "intent": r.intent,
                "climate_context": r.climate_context,
                "initial_design": r.initial_design,
                "thermal_result": r.thermal_result,
                "constraints_detected": r.constraints_detected,
                "improvements": r.improvements,
                "final_design": r.final_design,
                "final_evaluation": r.final_evaluation,
                "cost_impact": r.cost_impact,
                "success": r.success,
                "failure_reason": r.failure_reason,
                "key_learnings": r.key_learnings,
                "created_at": r.created_at.isoformat() if r.created_at else None
            })

        return results


def get_design_history(building_id: str) -> List[Dict[str, Any]]:
    """Retrieve full design version history for a building without overwriting."""
    ensure_db_initialized()
    with get_db() as db:
        versions = db.query(DesignVersion).filter(
            DesignVersion.building_id == building_id
        ).order_by(DesignVersion.version_number.asc()).all()

        return [
            {
                "id": v.id,
                "building_id": v.building_id,
                "version_number": v.version_number,
                "parent_version_id": v.parent_version_id,
                "parameters": v.parameters,
                "created_at": v.created_at.isoformat() if v.created_at else None
            }
            for v in versions
        ]


def get_agent_steps(agent_run_id: str) -> List[Dict[str, Any]]:
    """Retrieve step-by-step execution history of an agent run."""
    ensure_db_initialized()
    with get_db() as db:
        steps = db.query(AgentStep).filter(
            AgentStep.agent_run_id == agent_run_id
        ).order_by(AgentStep.step_number.asc()).all()

        return [
            {
                "id": s.id,
                "step_number": s.step_number,
                "decision": s.decision,
                "tool_called": s.tool_called,
                "tool_arguments": s.tool_arguments,
                "tool_result": s.tool_result,
                "created_at": s.created_at.isoformat() if s.created_at else None
            }
            for s in steps
        ]
