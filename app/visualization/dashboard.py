"""
Matplotlib 3-Panel Visualization Dashboard for Adaptive Shelter Agent.
Combines:
- Figure 1: Thermal & Climate Analysis (Operative Temp vs NBC 2016 limit, Airflow/RH, Evolution)
- Figure 2: Shelter Architecture Blueprint (img_plot/arch_plan.jpeg)
- Figure 3: Shelter Walkthrough & Live 3D Simulation Video Player (video_plot/model_3d.mp4)
"""
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from app.visualization.thermal_plots import render_thermal_panel, extract_thermal_metrics
from app.visualization.architecture_plot import render_architecture_panel
from app.visualization.video_placeholder import render_video_placeholder, attach_video_animation
from app.tools.thermal import evaluate_thermal
from app.tools.cost import evaluate_cost


def generate_analysis_plot(
    state: Optional[Any] = None,
    plot_type: Optional[str] = "thermal_analysis",
    save_path: Optional[str] = "data/analysis_dashboard.png",
    **kwargs
) -> Dict[str, Any]:
    """
    Agent Tool: Generate live Matplotlib visualization plots and multi-panel dashboard strictly from CURRENT AgentState.
    
    Args:
        state: Active AgentState containing live design, climate observations, and evaluations.
        plot_type: Type of visualization requested ('thermal_analysis', 'cost_analysis', 'full_dashboard').
        save_path: Destination path for exported PNG image.
        
    Returns:
        Structured observation dictionary with status, file path, and live calculated metrics.
    """
    if state is None:
        return {
            "status": "warning",
            "visualization_type": plot_type or "thermal_analysis",
            "data_source": "current_agent_state",
            "message": "No agent state provided for visualization.",
            "file_path": None
        }

    curr_design = getattr(state, "current_design", None) if not isinstance(state, dict) else state.get("current_design")
    design_hist = getattr(state, "design_history", []) if not isinstance(state, dict) else state.get("design_history", [])
    constraints = getattr(state, "constraints", []) if not isinstance(state, dict) else state.get("constraints", [])

    if not curr_design and not design_hist and not constraints:
        return {
            "status": "warning",
            "visualization_type": plot_type or "thermal_analysis",
            "data_source": "current_agent_state",
            "message": "No active shelter design or thermal analysis found in agent state to visualize. Please run generate_design or analyze_thermal_constraints first.",
            "file_path": None
        }

    # Ensure live evaluations are performed deterministically against current design
    if curr_design and hasattr(state, "evaluation"):
        reqs = getattr(state, "requirements", {})
        climate = reqs.get("climate", "hot_humid")
        budget = float(reqs.get("budget", 80000.0))

        if state.evaluation is None:
            state.evaluation = {}
        if "thermal" not in state.evaluation:
            state.evaluation["thermal"] = evaluate_thermal(design=curr_design, climate=climate)
        if "cost" not in state.evaluation:
            state.evaluation["cost"] = evaluate_cost(design=curr_design, budget=budget)

    out_file = generate_analysis_dashboard(state=state, save_path=save_path)
    metrics = extract_thermal_metrics(state)

    total_cost = None
    if hasattr(state, "evaluation") and isinstance(state.evaluation, dict):
        total_cost = state.evaluation.get("cost", {}).get("total_cost")

    return {
        "status": "success",
        "visualization_type": plot_type or "thermal_analysis",
        "data_source": "current_agent_state",
        "step": getattr(state, "iteration", 0),
        "file_path": str(out_file),
        "metrics": {
            "outdoor_temp_c": metrics.get("outdoor_temp"),
            "indoor_temp_c": metrics.get("indoor_temp"),
            "neutral_temp_c": metrics.get("neutral_temp"),
            "upper_90_limit_c": metrics.get("upper_90_limit"),
            "thermal_score": metrics.get("thermal_score"),
            "indoor_air_speed_ms": metrics.get("indoor_air_speed"),
            "design_version": curr_design.get("version", 1) if curr_design else None,
            "total_cost_inr": total_cost,
            "versions_tracked": metrics.get("versions", [])
        }
    }



def generate_analysis_dashboard(
    state: Any,
    show_gui: bool = False,
    save_path: Optional[str] = "data/analysis_dashboard.png",
    arch_image_path: Optional[str] = "img_plot/arch_plan.jpeg",
    video_path: Optional[str] = "video_plot/model_3d.mp4"
) -> str:
    """
    Generate the 3-panel Matplotlib analysis dashboard and export to PNG.
    
    Args:
        state: AgentState object or dictionary containing analysis results and history.
        show_gui: If True and running in interactive environment, displays the window.
        save_path: Path to save the exported dashboard image.
        arch_image_path: Path to the architectural blueprint image.
        video_path: Path to the 3D walkthrough / CFD video asset.
        
    Returns:
        String path to the saved dashboard PNG image.
    """
    # Create Figure with 3 distinct panel zones using GridSpec
    fig = plt.figure(figsize=(18, 10), facecolor="#f8fafc")

    # Main Grid: Top Row = Visual Assets (Fig 2 & Fig 3), Bottom Row = Thermal Analytics (Fig 1: 3 sub-plots)
    gs = GridSpec(2, 3, height_ratios=[1.1, 1.0], hspace=0.35, wspace=0.25, figure=fig)

    # Figure 2: Architecture Blueprint (Spans Top Left 2 columns)
    ax_arch = fig.add_subplot(gs[0, :2])
    design_spec = getattr(state, "current_design", None) if not isinstance(state, dict) else state.get("current_design")
    render_architecture_panel(ax_arch, image_path=arch_image_path, design_spec=design_spec)

    # Figure 3: Simulation & Video Walkthrough (Top Right column)
    ax_video = fig.add_subplot(gs[0, 2])
    render_video_placeholder(ax_video, video_path=video_path)

    # Figure 1: Thermal & Climate Analytics (3 subplots across the bottom row)
    ax_temp = fig.add_subplot(gs[1, 0])
    ax_air = fig.add_subplot(gs[1, 1])
    ax_evol = fig.add_subplot(gs[1, 2])
    render_thermal_panel(ax_temp, ax_air, ax_evol, state=state)

    # Dashboard Supertitle
    run_id = getattr(state, "run_id", "Run-Live") if not isinstance(state, dict) else state.get("run_id", "Run-Live")
    status = getattr(state, "status", "Completed") if not isinstance(state, dict) else state.get("status", "Completed")
    fig.suptitle(
        f"VISHVAKARMA-AGENTIC-AI — CLIMATE RESILIENT DESIGN & THERMAL ANALYSIS DASHBOARD\n"
        f"[Status: {str(status).upper()} | Run ID: {run_id or 'Live'} | Standards: NBC 2016 / TERI-2021 / TARU-2015]",
        fontsize=14,
        fontweight="bold",
        color="#0f172a",
        y=0.98
    )

    # Ensure destination directory exists
    out_file = Path(save_path) if save_path else Path("data/analysis_dashboard.png")
    if not out_file.is_absolute():
        workspace_root = Path(__file__).resolve().parent.parent.parent
        out_file = workspace_root / out_file
    
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Save high-resolution figure
    fig.savefig(str(out_file), dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")

    if show_gui:
        try:
            plt.show()
        except Exception:
            pass

    plt.close(fig)
    return str(out_file)


def show_interactive_dashboard(
    state: Any,
    save_path: Optional[str] = "data/analysis_dashboard.png",
    arch_image_path: Optional[str] = "img_plot/arch_plan.jpeg",
    video_path: Optional[str] = "video_plot/model_3d.mp4"
) -> None:
    """
    Launch interactive Matplotlib GUI window with live looping 3D video simulation in Figure 3,
    architectural blueprint in Figure 2, and thermal analytics graphs in Figure 1.
    """
    # Create interactive figure window
    fig = plt.figure(num="Vishvakarma-Agentic-AI - Live 3-Panel Visual Analysis Dashboard", figsize=(17, 9.5), facecolor="#f8fafc")
    gs = GridSpec(2, 3, height_ratios=[1.1, 1.0], hspace=0.35, wspace=0.25, figure=fig)

    # Figure 2: Architecture Blueprint (Top Left)
    ax_arch = fig.add_subplot(gs[0, :2])
    design_spec = getattr(state, "current_design", None) if not isinstance(state, dict) else state.get("current_design")
    render_architecture_panel(ax_arch, image_path=arch_image_path, design_spec=design_spec)

    # Figure 3: Live 3D Simulation Video Player with Animation (Top Right)
    ax_video = fig.add_subplot(gs[0, 2])
    anim = attach_video_animation(fig, ax_video, video_path=video_path, interval_ms=40)
    if anim is None:
        render_video_placeholder(ax_video, video_path=video_path)

    # Figure 1: Thermal & Climate Analytics (Bottom Row)
    ax_temp = fig.add_subplot(gs[1, 0])
    ax_air = fig.add_subplot(gs[1, 1])
    ax_evol = fig.add_subplot(gs[1, 2])
    render_thermal_panel(ax_temp, ax_air, ax_evol, state=state)

    run_id = getattr(state, "run_id", "Live-GUI") if not isinstance(state, dict) else state.get("run_id", "Live-GUI")
    fig.suptitle(
        f"VISHVAKARMA-AGENTIC-AI — LIVE DESIGN & THERMAL ANALYSIS DASHBOARD\n"
        f"[Figure 1: Thermal Graphs | Figure 2: Architecture Plan | Figure 3: 3D Video Simulation]",
        fontsize=13,
        fontweight="bold",
        color="#0f172a",
        y=0.98
    )

    # Also save static export
    try:
        out_file = Path(save_path) if save_path else Path("data/analysis_dashboard.png")
        if not out_file.is_absolute():
            workspace_root = Path(__file__).resolve().parent.parent.parent
            out_file = workspace_root / out_file
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_file), dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    except Exception:
        pass

    try:
        plt.show()
    except Exception as e:
        print(f"[GUI Notice] Matplotlib interactive display: {e}")
    finally:
        plt.close(fig)


def launch_dashboard_gui_process(save_path: Optional[str] = "data/analysis_dashboard.png") -> None:
    """
    Launch interactive dashboard viewer as an independent GUI process so it pops up
    seamlessly beside the terminal without blocking CLI execution.
    """
    workspace_root = Path(__file__).resolve().parent.parent.parent
    runner_code = (
        "import json\n"
        "from app.visualization.dashboard import show_interactive_dashboard\n"
        "state = {'user_query': 'Live Analysis', 'status': 'Completed', 'run_id': 'Live-Viewer'}\n"
        "try:\n"
        "    with open('data/last_state.json', 'r', encoding='utf-8') as f:\n"
        "        state = json.load(f)\n"
        "except Exception:\n"
        "    pass\n"
        "show_interactive_dashboard(state)\n"
    )
    try:
        subprocess.Popen([sys.executable, "-c", runner_code], cwd=str(workspace_root))
    except Exception as e:
        print(f"[GUI Notice] Could not spawn background GUI window: {e}")
