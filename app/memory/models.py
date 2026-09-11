"""
SQLAlchemy ORM Models for PostgreSQL Long-Term Experience Memory.
Maps all 11 relational entities for projects, buildings, design versions, agent runs, steps, and experiences.
"""
import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, Text, DateTime,
    ForeignKey, UniqueConstraint, Index, JSON
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Project(Base):
    """Top-level design project container."""
    __tablename__ = "projects"

    id = Column(String(64), primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc))

    buildings = relationship("Building", back_populates="project", cascade="all, delete-orphan")
    agent_runs = relationship("AgentRun", back_populates="project")
    experiences = relationship("ExperienceRecord", back_populates="project")


class Building(Base):
    """Building or Shelter instance (e.g. Modular Relief Shelter, Prefab Panel Unit)."""
    __tablename__ = "buildings"

    id = Column(String(64), primary_key=True)
    project_id = Column(String(64), ForeignKey("projects.id", ondelete="CASCADE"), nullable=True)
    name = Column(String(255), nullable=False)
    building_type = Column(String(64), default="modular_shelter")
    climate_zone = Column(String(64), nullable=False)
    geometry = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc))

    project = relationship("Project", back_populates="buildings")
    design_versions = relationship("DesignVersion", back_populates="building", cascade="all, delete-orphan")
    agent_runs = relationship("AgentRun", back_populates="building")
    experiences = relationship("ExperienceRecord", back_populates="building")


class DesignVersion(Base):
    """Immutable design version (V1, V2, V3 lineage). Never overwritten."""
    __tablename__ = "design_versions"

    id = Column(String(64), primary_key=True)
    building_id = Column(String(64), ForeignKey("buildings.id", ondelete="CASCADE"), nullable=False)
    version_number = Column(Integer, nullable=False)
    parent_version_id = Column(String(64), ForeignKey("design_versions.id", ondelete="SET NULL"), nullable=True)
    parameters = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc))

    building = relationship("Building", back_populates="design_versions")
    thermal_analyses = relationship("ThermalAnalysis", back_populates="design_version")

    __table_args__ = (
        UniqueConstraint("building_id", "version_number", name="uq_building_version"),
        Index("idx_design_versions_building", "building_id", "version_number"),
    )


class AgentRun(Base):
    """Single autonomous execution episode of the agent controller."""
    __tablename__ = "agent_runs"

    id = Column(String(64), primary_key=True)
    project_id = Column(String(64), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True)
    building_id = Column(String(64), ForeignKey("buildings.id", ondelete="SET NULL"), nullable=True)
    user_query = Column(Text, nullable=False)
    intent = Column(String(64), nullable=False)
    status = Column(String(32), nullable=False)
    total_steps = Column(Integer, default=0)
    final_response = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc))

    project = relationship("Project", back_populates="agent_runs")
    building = relationship("Building", back_populates="agent_runs")
    steps = relationship("AgentStep", back_populates="agent_run", cascade="all, delete-orphan")
    climate_observations = relationship("ClimateObservation", back_populates="agent_run", cascade="all, delete-orphan")
    thermal_analyses = relationship("ThermalAnalysis", back_populates="agent_run", cascade="all, delete-orphan")
    improvements = relationship("ImprovementRecord", back_populates="agent_run", cascade="all, delete-orphan")
    cost_results = relationship("CostResultRecord", back_populates="agent_run", cascade="all, delete-orphan")
    experiences = relationship("ExperienceRecord", back_populates="agent_run")


class AgentStep(Base):
    """Granular record of an individual reasoning and tool step."""
    __tablename__ = "agent_steps"

    id = Column(String(64), primary_key=True)
    agent_run_id = Column(String(64), ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False)
    step_number = Column(Integer, nullable=False)
    state_before = Column(JSON, nullable=True)
    decision = Column(JSON, nullable=False)
    tool_called = Column(String(64), nullable=True)
    tool_arguments = Column(JSON, nullable=True)
    tool_result = Column(JSON, nullable=True)
    state_after = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc))

    agent_run = relationship("AgentRun", back_populates="steps")

    __table_args__ = (
        UniqueConstraint("agent_run_id", "step_number", name="uq_run_step"),
        Index("idx_agent_steps_run", "agent_run_id", "step_number"),
    )


class ClimateObservation(Base):
    """Environmental and climate observation associated with an agent run."""
    __tablename__ = "climate_observations"

    id = Column(String(64), primary_key=True)
    agent_run_id = Column(String(64), ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False)
    location = Column(String(128), nullable=False)
    climate_zone = Column(String(64), nullable=False)
    temperature_c = Column(Float, nullable=True)
    rh_percent = Column(Float, nullable=True)
    wind_speed_ms = Column(Float, nullable=True)
    solar_radiation_w_m2 = Column(Float, nullable=True)
    trm_c = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc))

    agent_run = relationship("AgentRun", back_populates="climate_observations")


class ThermalAnalysis(Base):
    """Deterministic thermal evaluation results."""
    __tablename__ = "thermal_analyses"

    id = Column(String(64), primary_key=True)
    agent_run_id = Column(String(64), ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False)
    design_version_id = Column(String(64), ForeignKey("design_versions.id", ondelete="SET NULL"), nullable=True)
    indoor_temp_c = Column(Float, nullable=True)
    neutral_temp_c = Column(Float, nullable=True)
    upper_90_limit = Column(Float, nullable=True)
    lower_90_limit = Column(Float, nullable=True)
    comfort_score = Column(Float, nullable=True)
    passed = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc))

    agent_run = relationship("AgentRun", back_populates="thermal_analyses")
    design_version = relationship("DesignVersion", back_populates="thermal_analyses")
    constraints = relationship("ThermalConstraintRecord", back_populates="thermal_analysis", cascade="all, delete-orphan")


class ThermalConstraintRecord(Base):
    """Specific TC-001..TC-009 constraint evaluation record."""
    __tablename__ = "thermal_constraints"

    id = Column(String(64), primary_key=True)
    thermal_analysis_id = Column(String(64), ForeignKey("thermal_analyses.id", ondelete="CASCADE"), nullable=False)
    constraint_id = Column(String(32), nullable=False)
    parameter = Column(String(64), nullable=False)
    actual_value = Column(Text, nullable=True)
    threshold_value = Column(Text, nullable=True)
    status = Column(String(32), nullable=False)
    reason = Column(Text, nullable=True)
    source = Column(String(128), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc))

    thermal_analysis = relationship("ThermalAnalysis", back_populates="constraints")


class ImprovementRecord(Base):
    """Modification proposed or applied by the agent."""
    __tablename__ = "improvements"

    id = Column(String(64), primary_key=True)
    agent_run_id = Column(String(64), ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False)
    parameter = Column(String(64), nullable=False)
    current_value = Column(Text, nullable=True)
    recommended_value = Column(Text, nullable=True)
    expected_effect = Column(Text, nullable=True)
    reason = Column(Text, nullable=True)
    source = Column(String(128), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc))

    agent_run = relationship("AgentRun", back_populates="improvements")


class CostResultRecord(Base):
    """Cost calculation record (2014 INR historical baseline)."""
    __tablename__ = "cost_results"

    id = Column(String(64), primary_key=True)
    agent_run_id = Column(String(64), ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False)
    total_cost_inr = Column(Float, nullable=False)
    itemized = Column(JSON, nullable=False)
    is_historical = Column(Boolean, default=True)
    cost_year = Column(Integer, default=2014)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc))

    agent_run = relationship("AgentRun", back_populates="cost_results")


class ExperienceRecord(Base):
    """Complete episodic design and analysis experience for long-term memory retrieval."""
    __tablename__ = "experiences"

    id = Column(String(64), primary_key=True)
    project_id = Column(String(64), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True)
    building_id = Column(String(64), ForeignKey("buildings.id", ondelete="SET NULL"), nullable=True)
    agent_run_id = Column(String(64), ForeignKey("agent_runs.id", ondelete="SET NULL"), nullable=True)
    user_request = Column(Text, nullable=False)
    intent = Column(String(64), nullable=False)
    climate_context = Column(JSON, nullable=False)
    initial_design = Column(JSON, nullable=True)
    thermal_result = Column(JSON, nullable=True)
    constraints_detected = Column(JSON, nullable=True)
    actions_taken = Column(JSON, nullable=True)
    improvements = Column(JSON, nullable=True)
    final_design = Column(JSON, nullable=True)
    final_evaluation = Column(JSON, nullable=True)
    cost_impact = Column(JSON, nullable=True)
    success = Column(Boolean, nullable=False)
    failure_reason = Column(Text, nullable=True)
    key_learnings = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc))

    project = relationship("Project", back_populates="experiences")
    building = relationship("Building", back_populates="experiences")
    agent_run = relationship("AgentRun", back_populates="experiences")

    __table_args__ = (
        Index("idx_experiences_intent", "intent"),
        Index("idx_experiences_success", "success"),
    )
