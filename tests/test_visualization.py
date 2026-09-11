"""
Unit and Integration Tests for Matplotlib Visualization Module.
Validates:
1. Figure 1: Thermal and Climate Analytics Plots (Temperature, Limits, Airflow, RH, Evolution)
2. Figure 2: Architecture Plan Image loading & Schematic fallback
3. Figure 3: Video Simulation Placeholder & Metadata
4. 3-Panel Combined Dashboard Generator & Export
"""
from pathlib import Path
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.agent.state import AgentState
from app.visualization.thermal_plots import extract_thermal_metrics, render_thermal_panel
from app.visualization.architecture_plot import render_architecture_panel
from app.visualization.video_placeholder import render_video_placeholder
from app.visualization.dashboard import generate_analysis_dashboard


def test_extract_thermal_metrics_defaults():
    """Verify extraction handles empty or minimal state gracefully."""
    state = AgentState(user_query="Test empty state")
    metrics = extract_thermal_metrics(state)

    assert "outdoor_temp" in metrics
    assert "indoor_temp" in metrics
    assert "upper_90_limit" in metrics
    assert "scores" in metrics
    assert len(metrics["versions"]) >= 1


def test_extract_thermal_metrics_populated():
    """Verify extraction pulls real constraints and multi-version design history."""
    state = AgentState(
        user_query="Analyze Kerala shelter",
        run_id="test-run-123",
        status="completed",
        constraints=[
            {"constraint_id": "TC-001", "actual_value": 34.5},
            {"constraint_id": "TC-004", "actual_value": 0.35}
        ],
        design_history=[
            {"version": "V1 (Initial)", "thermal_score": 0.58},
            {"version": "V2 (Optimized)", "thermal_score": 0.78},
            {"version": "V3 (Cool-Roof)", "thermal_score": 0.84}
        ],
        tool_results=[
            {
                "tool": "get_climate",
                "result": {
                    "temperature_c": 33.0,
                    "humidity_percent": 82.0,
                    "wind_speed_ms": 2.8,
                    "trm_c": 32.0,
                    "climate": "hot_humid"
                }
            }
        ]
    )
    metrics = extract_thermal_metrics(state)

    assert metrics["indoor_temp"] == 34.5
    assert metrics["indoor_air_speed"] == 0.35
    assert metrics["outdoor_temp"] == 33.0
    assert metrics["rh_percent"] == 82.0
    assert len(metrics["versions"]) == 3
    assert metrics["scores"] == [0.58, 0.78, 0.84]


def test_render_thermal_panel():
    """Verify thermal panel renders onto sub-axes without errors."""
    fig, (ax_temp, ax_air, ax_evol) = plt.subplots(1, 3, figsize=(15, 4))
    state = AgentState(user_query="Test thermal render")
    render_thermal_panel(ax_temp, ax_air, ax_evol, state=state)

    assert ax_temp.get_title() != ""
    assert ax_air.get_title() != ""
    assert ax_evol.get_title() != ""
    plt.close(fig)


def test_render_architecture_panel_image():
    """Verify architecture image loads from img_plot/arch_plan.jpeg."""
    fig, ax = plt.subplots(figsize=(6, 4))
    render_architecture_panel(ax, image_path="img_plot/arch_plan.jpeg")
    assert ax.get_title() == "Shelter Architecture"
    plt.close(fig)


def test_render_architecture_panel_fallback():
    """Verify architecture schematic fallback works when image is missing."""
    fig, ax = plt.subplots(figsize=(6, 4))
    render_architecture_panel(ax, image_path="non_existent/path.png")
    assert "Schematic" in ax.get_title()
    plt.close(fig)


def test_render_video_placeholder():
    """Verify video placeholder renders for existing or missing asset."""
    fig, ax = plt.subplots(figsize=(6, 4))
    render_video_placeholder(ax, video_path="video_plot/model_3d.mp4")
    assert "Figure 3" in ax.get_title() or "3D Shelter" in ax.get_title() or "Simulation" in ax.get_title()
    plt.close(fig)


def test_attach_video_animation():
    """Verify live video animation attaches properly to figure axis."""
    from app.visualization.video_placeholder import attach_video_animation
    fig, ax = plt.subplots(figsize=(6, 4))
    anim = attach_video_animation(fig, ax, video_path="video_plot/model_3d.mp4", interval_ms=50)
    assert anim is not None
    plt.close(fig)


def test_generate_analysis_dashboard(tmp_path):
    """Verify full 3-panel dashboard generation and PNG file export."""
    state = AgentState(
        user_query="Analyze the model house. What thermal constraints could it exceed?",
        run_id="run-test-vis",
        status="completed",
        design_history=[
            {"version": "V1", "thermal_score": 0.58},
            {"version": "V2", "thermal_score": 0.76}
        ]
    )
    test_out = tmp_path / "dashboard_test.png"
    out_path = generate_analysis_dashboard(state, save_path=str(test_out))

    assert Path(out_path).exists()
    assert Path(out_path).stat().st_size > 10000  # Non-trivial image size
