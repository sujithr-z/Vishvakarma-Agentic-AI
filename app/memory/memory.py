"""
Experience Memory Manager backed by PostgreSQL Repository.
Maintains persistent long-term episodic memory for shelter design and thermal analysis.
"""
import os
import json
import datetime
from typing import Any, Dict, List, Optional

from app.memory.database import init_db, is_using_postgres
from app.memory.repository import (
    save_experience as repo_save_experience,
    retrieve_relevant_experiences as repo_retrieve_experiences,
    save_project,
    save_building,
    save_design_version
)

EXPERIENCES_JSON = os.path.join(os.path.dirname(__file__), "experiences.json")


class ExperienceMemory:
    """Stores and retrieves past design episodes, solutions, failures, and critiques via PostgreSQL."""

    def __init__(self, json_path: str = EXPERIENCES_JSON):
        self.json_path = json_path
        self._init_storage()
        self._seed_experiences()

    def _init_storage(self):
        """Initialize PostgreSQL tables."""
        try:
            init_db()
        except Exception as e:
            print(f"[Memory Warning] Database init failed: {e}")

    def _seed_experiences(self):
        """Seed relational database from experiences.json if no experiences exist."""
        if not os.path.exists(self.json_path):
            return

        try:
            existing = repo_retrieve_experiences(top_k=1)
            if existing:
                return

            with open(self.json_path, "r", encoding="utf-8") as f:
                items = json.load(f)

            for item in items:
                p_id = item.get("project_id", "proj_default")
                b_id = item.get("building_id", "bldg_default")
                save_project(project_id=p_id, name="Aid Modular Shelter Project")
                save_building(
                    building_id=b_id,
                    project_id=p_id,
                    name=item.get("initial_design", {}).get("name", "Modular-Aid-Shelter-v1"),
                    climate_zone=item.get("climate_context", {}).get("climate", "hot_humid"),
                    building_type=item.get("initial_design", {}).get("building_type", "modular_shelter"),
                    geometry=item.get("initial_design", {}).get("dimensions", {})
                )

                repo_save_experience(
                    exp_id=item.get("id", f"exp_{int(datetime.datetime.now().timestamp())}"),
                    project_id=p_id,
                    building_id=b_id,
                    user_request=item.get("user_request", "Design a modular shelter"),
                    intent=item.get("intent", "design_generation"),
                    climate_context=item.get("climate_context", {}),
                    initial_design=item.get("initial_design"),
                    thermal_result=item.get("thermal_result"),
                    constraints_detected=item.get("constraints_detected"),
                    actions_taken=item.get("actions_taken"),
                    improvements=item.get("improvements"),
                    final_design=item.get("final_design"),
                    final_evaluation=item.get("final_evaluation"),
                    cost_impact=item.get("cost_impact"),
                    success=item.get("success", True),
                    failure_reason=item.get("failure_reason"),
                    key_learnings=item.get("key_learnings")
                )
        except Exception as e:
            print(f"[Memory Warning] Experience seeding skipped: {e}")

    def save_experience(
        self,
        design: Optional[Dict[str, Any]] = None,
        evaluation: Optional[Dict[str, Any]] = None,
        key_learnings: Optional[str] = None,
        location: str = "Unknown",
        climate: str = "hot_humid",
        capacity: int = 5,
        budget: float = 80000.0,
        user_request: Optional[str] = None,
        intent: str = "design_generation",
        improvements: Optional[List[Any]] = None,
        success: bool = True,
        failure_reason: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Save a new design or thermal analysis episode to PostgreSQL experience memory."""
        exp_id = f"exp_{int(datetime.datetime.now().timestamp())}"
        req_text = user_request or f"Design modular shelter for {capacity} people in {location} with budget {budget} INR"
        learnings_text = key_learnings or "Optimized modular panel ventilation, overhang, and cool roof to meet thermal requirements."

        c_ctx = {
            "location": location,
            "climate": climate,
            "capacity": capacity,
            "budget": budget
        }

        # Delegate to repository
        res = repo_save_experience(
            exp_id=exp_id,
            user_request=req_text,
            intent=intent,
            climate_context=c_ctx,
            initial_design=design,
            thermal_result=evaluation.get("thermal") if evaluation else None,
            constraints_detected=kwargs.get("constraints"),
            actions_taken=kwargs.get("actions_taken"),
            improvements=improvements or kwargs.get("improvements"),
            final_design=design,
            final_evaluation=evaluation,
            cost_impact=evaluation.get("cost") if evaluation else None,
            success=success,
            failure_reason=failure_reason,
            key_learnings=learnings_text
        )

        return {
            "success": True,
            "experience_id": exp_id,
            "storage_backend": "PostgreSQL" if is_using_postgres() else "Relational SQLite (Fallback)",
            "message": f"Successfully saved experience {exp_id} to PostgreSQL experience memory."
        }

    def retrieve_experience(
        self,
        query: str = "",
        climate: Optional[str] = None,
        location: Optional[str] = None,
        building_type: Optional[str] = None,
        intent: Optional[str] = None,
        top_k: int = 2,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant past experiences matching climate context, location, or query."""
        results = repo_retrieve_experiences(
            climate=climate,
            location=location,
            building_type=building_type,
            intent=intent,
            query=query,
            top_k=top_k
        )

        # Backward compatibility format for existing tests/tools
        formatted = []
        for r in results:
            formatted.append({
                "id": r["experience_id"],
                "timestamp": r["created_at"],
                "climate": r["climate_context"].get("climate", "hot_humid"),
                "location": r["climate_context"].get("location", "Unknown"),
                "capacity": r["climate_context"].get("capacity", 5),
                "budget": r["climate_context"].get("budget", 80000.0),
                "design_summary": r["final_design"] or r["initial_design"] or {},
                "evaluation_scores": r["final_evaluation"] or {},
                "key_learnings": r["key_learnings"],
                "improvements": r["improvements"],
                "success": r["success"],
                "failure_reason": r["failure_reason"]
            })

        return formatted


# Global memory singleton instance
memory_instance = ExperienceMemory()


def save_experience(**kwargs) -> Dict[str, Any]:
    """Tool function wrapper for saving experience to PostgreSQL."""
    return memory_instance.save_experience(**kwargs)


def retrieve_experience(**kwargs) -> List[Dict[str, Any]]:
    """Tool function wrapper for retrieving experience from PostgreSQL."""
    return memory_instance.retrieve_experience(**kwargs)
