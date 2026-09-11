"""Tool Registry mapping action names to executable Python functions and providing metadata."""
from typing import Any, Callable, Dict
from app.tools.climate import get_climate
from app.knowledge.retriever import search_knowledge
from app.tools.shelter import generate_design, modify_design
from app.tools.cost import evaluate_cost, calculate_improvement_cost
from app.tools.thermal import evaluate_thermal
from app.tools.structure import evaluate_structure
from app.tools.constraints import analyze_thermal_constraints
from app.memory.memory import save_experience, retrieve_experience
from app.visualization.dashboard import generate_analysis_plot

TOOLS: Dict[str, Callable[..., Any]] = {
    "get_climate": get_climate,
    "search_knowledge": search_knowledge,
    "generate_design": generate_design,
    "evaluate_thermal": evaluate_thermal,
    "evaluate_cost": evaluate_cost,
    "evaluate_structure": evaluate_structure,
    "modify_design": modify_design,
    "analyze_thermal_constraints": analyze_thermal_constraints,
    "calculate_improvement_cost": calculate_improvement_cost,
    "generate_analysis_plot": generate_analysis_plot,
    "save_experience": save_experience,
    "retrieve_experience": retrieve_experience,
}

TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "get_climate": {
        "description": "Fetch climate conditions (temperature, humidity, wind, rainfall) for a given location or region.",
        "arguments": {
            "location": {"type": "string", "description": "Name of city, state, or region (e.g. 'Kerala', 'Rajasthan')"}
        }
    },
    "search_knowledge": {
        "description": "Search curated engineering and passive design principles for shelters (e.g. hot-humid ventilation, overhangs, bamboo framing).",
        "arguments": {
            "query": {"type": "string", "description": "Search query terms (e.g. 'hot humid roof ventilation principles')"}
        }
    },
    "generate_design": {
        "description": "Generate an initial structured ShelterSpec dictionary matching occupancy capacity and budget.",
        "arguments": {
            "capacity": {"type": "integer", "description": "Number of occupants (e.g. 5)"},
            "budget": {"type": "number", "description": "Maximum budget in INR (e.g. 80000)"},
            "climate": {"type": "string", "description": "Climate type (e.g. 'hot_humid')"}
        }
    },
    "analyze_thermal_constraints": {
        "description": "Perform comprehensive grounded thermal constraint analysis against physical engineering thresholds (NBC 2016, ASHRAE 55, BIS).",
        "arguments": {
            "location": {"type": "string", "description": "Geographic location (e.g. 'Kerala')"}
        }
    },
    "calculate_improvement_cost": {
        "description": "Deterministically calculate itemized cost impact for proposed shelter improvements (roof vent, overhang, openings).",
        "arguments": {}
    },
    "evaluate_cost": {
        "description": "Deterministically calculate construction costs of the current shelter design against budget.",
        "arguments": {}
    },
    "evaluate_thermal": {
        "description": "Evaluate thermal comfort and ventilation performance score against the target threshold (0.70).",
        "arguments": {}
    },
    "evaluate_structure": {
        "description": "Perform structural constraint checking on spans, height, column spacing, and load safety.",
        "arguments": {}
    },
    "modify_design": {
        "description": "Deterministically adapt/improve shelter design parameters (ventilation, overhang, opening ratio) based on evaluation critique.",
        "arguments": {
            "critique": {"type": "string", "description": "Summary of issues or critique feedback to resolve"}
        }
    },
    "generate_analysis_plot": {
        "description": "Render and export live 3-panel Matplotlib visualization plots and multi-panel dashboards from CURRENT agent state metrics (thermal operative temp vs NBC 2016 limits, airflow/RH, comfort evolution, architecture blueprint). Call when user requests a graph, plot, dashboard, or visual analysis.",
        "arguments": {
            "plot_type": {"type": "string", "description": "Type of visualization (e.g. 'thermal_analysis', 'cost_analysis', 'full_dashboard')"}
        }
    },
    "save_experience": {
        "description": "Save completed successful shelter design and learnings into episodic memory for future retrieval.",
        "arguments": {
            "key_learnings": {"type": "string", "description": "Summary of what worked well and parameters tuned"}
        }
    },
    "retrieve_experience": {
        "description": "Retrieve past successful shelter designs and episode experiences from memory.",
        "arguments": {
            "query": {"type": "string", "description": "Search query for past cases"},
            "climate": {"type": "string", "description": "Target climate filter"}
        }
    }
}


def get_tool(action: str) -> Callable[..., Any]:
    """Retrieve executable tool callable by action name."""
    if action not in TOOLS:
        raise KeyError(f"Tool '{action}' is not registered. Available tools: {list(TOOLS.keys())}")
    return TOOLS[action]
