"""Agent State data structures for tracking current situation and context."""
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class TraceEntry(BaseModel):
    """Single step trace entry for auditing agent decisions."""
    step: int
    actor: str  # USER, AGENT, TOOL, RAG, CRITIC, SYSTEM
    message: str
    data: Optional[Any] = None
    timestamp: Optional[str] = None


class AgentState(BaseModel):
    """
    The central Situation State passed through every agent iteration.
    Maintains the complete working context, requirements, current design,
    evaluations, constraints, cost impacts, and history.
    """
    user_query: str
    run_id: Optional[str] = None
    intent: str = "design_generation"  # "thermal_analysis" | "design_generation" | "suitability_check" | "general_query" | "greeting"
    goal: Optional[str] = None
    requirements: Dict[str, Any] = Field(default_factory=dict)
    current_design: Optional[Dict[str, Any]] = None
    design_history: List[Dict[str, Any]] = Field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    tool_results: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_knowledge: List[Dict[str, Any]] = Field(default_factory=list)
    evaluation: Optional[Dict[str, Any]] = None
    critique: Optional[Dict[str, Any]] = None
    constraints: List[Dict[str, Any]] = Field(default_factory=list)
    suggestions: List[Union[str, Dict[str, Any]]] = Field(default_factory=list)
    improvements: List[Dict[str, Any]] = Field(default_factory=list)
    cost_impact: Optional[Dict[str, Any]] = None
    structured_analysis: Optional[Dict[str, Any]] = None
    iteration: int = 0
    status: str = "running"  # "running" | "completed" | "failed" | "max_iterations"
    final_answer: Optional[str] = None
    trace: List[TraceEntry] = Field(default_factory=list)

    def add_trace(self, actor: str, message: str, data: Optional[Any] = None):
        """Add a chronological trace entry."""
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        entry = TraceEntry(
            step=self.iteration,
            actor=actor,
            message=message,
            data=data,
            timestamp=ts
        )
        self.trace.append(entry)
