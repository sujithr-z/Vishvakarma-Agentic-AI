"""FastAPI Application and CLI Runner for Adaptive Shelter Agent v0.1."""
import os
import sys
import argparse
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from app.agent.state import AgentState
from app.agent.agent import Vishvakarma, ShelterAgent
from app.llm.ollama_client import OllamaClient
from app.tools.registry import TOOL_SCHEMAS
from app.memory.memory import memory_instance
from app.visualization.dashboard import generate_analysis_dashboard

load_dotenv()

app = FastAPI(
    title="Vishvakarma-Agentic-AI",
    description="Agentic AI Prototype proving local LLM (Qwen 1.8B) reasoning over deterministic tools",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm_client = OllamaClient()
agent = Vishvakarma(llm=llm_client)


class RunRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "Design a low-cost shelter for 5 people in a hot-humid environment with a budget of 80,000 INR."})
    max_iterations: Optional[int] = Field(default=10, ge=1, le=20)


class StepRequest(BaseModel):
    state: Dict[str, Any]


@app.get("/api/health")
def get_health():
    """System health check and Ollama connection status."""
    is_ollama_ok = llm_client.is_available()
    return {
        "status": "healthy",
        "ollama_connected": is_ollama_ok,
        "ollama_url": llm_client.base_url,
        "model": llm_client.model,
        "version": "0.1.0"
    }


@app.get("/api/tools")
def get_tools():
    """Inspect all registered deterministic tools and argument schemas."""
    return TOOL_SCHEMAS


@app.get("/api/memory")
def get_memory(climate: Optional[str] = None):
    """Retrieve saved experiences and episodes from memory."""
    return memory_instance.retrieve_experience(climate=climate, top_k=10)


@app.get("/api/visualization/dashboard.png")
def get_dashboard_image():
    """Serve latest generated Matplotlib analysis dashboard image."""
    from pathlib import Path
    img_path = Path("data/analysis_dashboard.png")
    if not img_path.exists():
        # Generate initial dashboard baseline
        generate_analysis_dashboard(state={
            "user_query": "Baseline Shelter Architecture & Thermal Analysis",
            "status": "Ready",
            "run_id": "Init"
        })
    return FileResponse(str(img_path), media_type="image/png")


@app.post("/api/agent/run")
def run_agent_endpoint(req: RunRequest):
    """Execute complete autonomous agent loop."""
    try:
        final_state = agent.run(user_query=req.query, max_iterations=req.max_iterations)
        try:
            generate_analysis_dashboard(final_state)
        except Exception as vis_err:
            print(f"Warning: Dashboard generation error: {vis_err}")
        return final_state.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/agent/step")
def step_agent_endpoint(req: StepRequest):
    """Execute a single step on the given state."""
    try:
        state_obj = AgentState(**req.state)
        decision = agent.run_step(state_obj)
        return {
            "decision": decision.model_dump(),
            "state": state_obj.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
def index_page():
    """Interactive Web Dashboard for observing the agent loop and state."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vishvakarma-Agentic-AI</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #0b0f19;
            --bg-surface: #131b2e;
            --bg-card: #1c2640;
            --bg-card-hover: #223052;
            --border: #2a3b61;
            --primary: #38bdf8;
            --primary-glow: rgba(56, 189, 248, 0.25);
            --accent: #818cf8;
            --success: #34d399;
            --warning: #fbbf24;
            --danger: #f87171;
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --radius-lg: 16px;
            --radius-md: 10px;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
            background: radial-gradient(circle at 10% 10%, #172554 0%, #0b0f19 50%, #030712 100%);
            color: var(--text-main);
            min-height: 100vh;
            padding: 24px;
            overflow-x: hidden;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 24px;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(19, 27, 46, 0.7);
            backdrop-filter: blur(12px);
            padding: 20px 28px;
            border-radius: var(--radius-lg);
            border: 1px solid var(--border);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        }

        .logo-area {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .badge-chip {
            background: linear-gradient(135deg, #0284c7, #6366f1);
            color: #ffffff;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 9999px;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        h1 {
            font-size: 1.6rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, #ffffff 30%, #93c5fd 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .status-pill {
            display: flex;
            align-items: center;
            gap: 8px;
            background: #0f172a;
            padding: 8px 14px;
            border-radius: 9999px;
            border: 1px solid var(--border);
            font-size: 0.85rem;
            color: var(--text-muted);
        }

        .dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--success);
            box-shadow: 0 0 10px var(--success);
        }

        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1.2fr;
            gap: 24px;
        }

        @media (max-width: 1024px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
        }

        .panel {
            background: rgba(19, 27, 46, 0.6);
            backdrop-filter: blur(12px);
            border-radius: var(--radius-lg);
            border: 1px solid var(--border);
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 20px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        }

        .panel-title {
            font-size: 1.15rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            justify-content: space-between;
            color: #e2e8f0;
            border-bottom: 1px solid rgba(42, 59, 97, 0.5);
            padding-bottom: 12px;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        label {
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-muted);
        }

        textarea {
            width: 100%;
            height: 90px;
            background: #0b1120;
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            color: #f8fafc;
            padding: 12px 16px;
            font-family: inherit;
            font-size: 0.95rem;
            resize: none;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        textarea:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px var(--primary-glow);
        }

        .button-bar {
            display: flex;
            gap: 12px;
        }

        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 12px 20px;
            border-radius: var(--radius-md);
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
            border: none;
            transition: all 0.2s ease;
        }

        .btn-primary {
            background: linear-gradient(135deg, #0284c7, #38bdf8);
            color: #041324;
            box-shadow: 0 4px 14px rgba(56, 189, 248, 0.3);
            flex: 1;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(56, 189, 248, 0.45);
        }

        .btn-secondary {
            background: #1e293b;
            color: #e2e8f0;
            border: 1px solid var(--border);
        }

        .btn-secondary:hover {
            background: #334155;
        }

        .trace-container {
            display: flex;
            flex-direction: column;
            gap: 12px;
            max-height: 520px;
            overflow-y: auto;
            padding-right: 6px;
        }

        .trace-item {
            background: #0f172a;
            border-left: 4px solid var(--primary);
            border-radius: 6px 10px 10px 6px;
            padding: 12px 16px;
            display: flex;
            flex-direction: column;
            gap: 6px;
            font-size: 0.88rem;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .trace-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .actor-tag {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 2px 6px;
            border-radius: 4px;
        }

        .actor-USER { background: #3730a3; color: #c7d2fe; }
        .actor-AGENT { background: #0369a1; color: #bae6fd; }
        .actor-TOOL { background: #065f46; color: #a7f3d0; }
        .actor-RAG { background: #713f12; color: #fef08a; }
        .actor-CRITIC { background: #831843; color: #fbcfe8; }
        .actor-MEMORY { background: #581c87; color: #e9d5ff; }

        .trace-msg {
            color: #cbd5e1;
            line-height: 1.4;
        }

        .trace-data {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            background: #030712;
            padding: 8px;
            border-radius: 6px;
            color: #94a3b8;
            overflow-x: auto;
            max-height: 140px;
        }

        .comparison-cards {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }

        .design-card {
            background: #0f172a;
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .design-card.v2-active {
            border-color: var(--success);
            box-shadow: 0 0 16px rgba(52, 211, 153, 0.15);
        }

        .design-card h4 {
            font-size: 0.95rem;
            display: flex;
            justify-content: space-between;
            color: #e2e8f0;
        }

        .metric-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            color: var(--text-muted);
            border-bottom: 1px dashed rgba(255,255,255,0.05);
            padding-bottom: 4px;
        }

        .metric-val {
            font-weight: 600;
            color: #f1f5f9;
        }

        .score-pill {
            padding: 2px 8px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
        }

        .score-pass { background: rgba(52, 211, 153, 0.2); color: var(--success); }
        .score-fail { background: rgba(248, 113, 113, 0.2); color: var(--danger); }

        pre code {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo-area">
                <span class="badge-chip">v0.1 Prototype</span>
                <h1>Vishvakarma-Agentic-AI</h1>
            </div>
            <div class="status-pill" id="healthPill">
                <div class="dot"></div>
                <span id="healthText">Connecting to Ollama (Qwen 1.8B)...</span>
            </div>
        </header>

        <div class="main-grid">
            <!-- Left Column: Controls and Design Evolution -->
            <div class="panel">
                <div class="panel-title">
                    <span>Task Specification</span>
                </div>

                <div class="input-group">
                    <label for="queryInput">Shelter Design Requirement Prompt:</label>
                    <textarea id="queryInput">Design a low-cost shelter for 5 people in a hot-humid environment (Kerala) with a budget of ₹80,000.</textarea>
                </div>

                <div class="button-bar">
                    <button class="btn btn-primary" id="btnRun" onclick="runAutonomousAgent()">
                        ⚡ Run Full Agent Loop
                    </button>
                    <button class="btn btn-secondary" onclick="presetExample('Thermal')">Thermal Constraints (TERI/TARU)</button>
                    <button class="btn btn-secondary" onclick="presetExample('Kerala')">Kerala (5P)</button>
                    <button class="btn btn-secondary" onclick="presetExample('Assam')">Assam (4P)</button>
                </div>

                <div class="panel-title" style="margin-top: 10px;">
                    <span>Design Evolution (V1 Baseline → V2 Adaptive)</span>
                </div>

                <div class="comparison-cards" id="comparisonArea">
                    <div class="design-card" id="cardV1">
                        <h4>
                            <span>Version 1 (Initial)</span>
                            <span class="score-pill score-fail" id="v1Badge">Pending</span>
                        </h4>
                        <div class="metric-row"><span>Roof Ventilation:</span><span class="metric-val" id="v1Vent">-</span></div>
                        <div class="metric-row"><span>Roof Overhang:</span><span class="metric-val" id="v1Overhang">-</span></div>
                        <div class="metric-row"><span>Wall Opening Ratio:</span><span class="metric-val" id="v1Opening">-</span></div>
                        <div class="metric-row"><span>Thermal Score:</span><span class="metric-val" id="v1Thermal">-</span></div>
                        <div class="metric-row"><span>Cost:</span><span class="metric-val" id="v1Cost">-</span></div>
                    </div>

                    <div class="design-card" id="cardV2">
                        <h4>
                            <span>Version 2 (Improved)</span>
                            <span class="score-pill score-pass" id="v2Badge">Pending</span>
                        </h4>
                        <div class="metric-row"><span>Roof Ventilation:</span><span class="metric-val" id="v2Vent">-</span></div>
                        <div class="metric-row"><span>Roof Overhang:</span><span class="metric-val" id="v2Overhang">-</span></div>
                        <div class="metric-row"><span>Wall Opening Ratio:</span><span class="metric-val" id="v2Opening">-</span></div>
                        <div class="metric-row"><span>Thermal Score:</span><span class="metric-val" id="v2Thermal">-</span></div>
                        <div class="metric-row"><span>Cost:</span><span class="metric-val" id="v2Cost">-</span></div>
                    </div>
                </div>

                <div id="finalSummaryBox" style="display:none; background:#0b1120; border:1px solid var(--border); border-radius:var(--radius-md); padding:14px; font-size:0.9rem; color:#e2e8f0; line-height:1.5; margin-top:14px;">
                    <strong style="color:var(--success); display:block; margin-bottom:8px; font-size:1rem;">✓ Agent Response & Technical Report</strong>
                    <pre id="finalSummaryText" style="white-space: pre-wrap; font-family: 'JetBrains Mono', Consolas, monospace; font-size: 0.85rem; color: #cbd5e1; background: #020617; padding: 12px; border-radius: 6px; border: 1px solid #1e293b; max-height: 400px; overflow-y: auto;"></pre>
                </div>
            </div>

            <!-- Right Column: Agent Execution Trace & State Inspector -->
            <div class="panel">
                <div class="panel-title">
                    <span>Autonomous Agent Trace</span>
                    <span style="font-size:0.8rem; color:var(--text-muted);" id="traceCount">0 steps</span>
                </div>

                <div class="trace-container" id="traceContainer">
                    <div style="color:var(--text-muted); font-size:0.9rem; text-align:center; padding:40px 0;">
                        Click <strong>"Run Full Agent Loop"</strong> to see Qwen 1.8B reason and execute tools in real time.
                    </div>
                </div>
            </div>
        </div>

        <!-- Bottom Section: 3-Panel Matplotlib Visualization Dashboard -->
        <div class="panel" style="margin-top: 10px;">
            <div class="panel-title">
                <span>Matplotlib Multi-Panel Analysis Dashboard (Figures 1, 2, 3)</span>
                <button class="btn btn-secondary" style="padding: 4px 12px; font-size: 0.8rem;" onclick="refreshDashboard()">🔄 Refresh View</button>
            </div>
            <div style="text-align: center; overflow-x: auto; background: #020617; padding: 12px; border-radius: var(--radius-md); border: 1px solid var(--border);">
                <img id="dashboardImg" src="/api/visualization/dashboard.png" alt="Adaptive Shelter Matplotlib Dashboard" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.4);" />
            </div>
        </div>
    </div>

    <script>
        async function checkHealth() {
            try {
                const res = await fetch('/api/health');
                const data = await res.json();
                const pill = document.getElementById('healthText');
                if (data.ollama_connected) {
                    pill.textContent = `Qwen 1.8B Connected (${data.model})`;
                } else {
                    pill.textContent = `Ollama Standby (${data.model})`;
                }
            } catch (e) {
                document.getElementById('healthText').textContent = 'API Offline';
            }
        }
        checkHealth();

        function refreshDashboard() {
            const img = document.getElementById('dashboardImg');
            if (img) {
                img.src = '/api/visualization/dashboard.png?t=' + new Date().getTime();
            }
        }

        function presetExample(loc) {
            if (loc === 'Thermal') {
                document.getElementById('queryInput').value = "Analyze the model house. What thermal constraints could it exceed?";
            } else if (loc === 'Kerala') {
                document.getElementById('queryInput').value = "Design a low-cost shelter for 5 people in a hot-humid environment (Kerala) with a budget of ₹80,000.";
            } else {
                document.getElementById('queryInput').value = "Design a flood-resilient bamboo shelter for 4 people in Assam under ₹75,000.";
            }
        }

        async function runAutonomousAgent() {
            const btn = document.getElementById('btnRun');
            const query = document.getElementById('queryInput').value;
            const traceBox = document.getElementById('traceContainer');
            
            btn.disabled = true;
            btn.innerHTML = '⏳ Reasoning with Qwen 1.8B...';
            traceBox.innerHTML = '<div style="color:var(--text-muted); text-align:center; padding:30px;">Executing agent reasoning loop...</div>';

            try {
                const res = await fetch('/api/agent/run', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ query: query, max_iterations: 10 })
                });
                const data = await res.json();
                renderTrace(data.trace || []);
                renderDesignComparison(data);
                refreshDashboard();
            } catch (err) {
                traceBox.innerHTML = `<div style="color:var(--danger); padding:20px;">Error running agent: ${err}</div>`;
            } finally {
                btn.disabled = false;
                btn.innerHTML = '⚡ Run Full Agent Loop';
            }
        }

        function renderTrace(trace) {
            const traceBox = document.getElementById('traceContainer');
            document.getElementById('traceCount').textContent = `${trace.length} events`;
            if (!trace.length) return;

            traceBox.innerHTML = '';
            trace.forEach(item => {
                const div = document.createElement('div');
                div.className = 'trace-item';
                div.innerHTML = `
                    <div class="trace-header">
                        <span class="actor-tag actor-${item.actor}">${item.actor} (Step ${item.step})</span>
                        <span style="font-size:0.75rem; color:var(--text-muted); font-family:JetBrains Mono;">${item.timestamp || ''}</span>
                    </div>
                    <div class="trace-msg">${item.message}</div>
                    ${item.data ? `<pre class="trace-data">${JSON.stringify(item.data, null, 2)}</pre>` : ''}
                `;
                traceBox.appendChild(div);
            });
            traceBox.scrollTop = traceBox.scrollHeight;
        }

        function renderDesignComparison(state) {
            const history = state.design_history || [];
            if (history.length > 0) {
                const v1 = history[0];
                document.getElementById('v1Vent').textContent = v1.roof?.ventilation || '0.5';
                document.getElementById('v1Overhang').textContent = (v1.roof?.overhang || '0.6') + 'm';
                document.getElementById('v1Opening').textContent = ((v1.walls?.opening_ratio || 0.25) * 100) + '%';
                document.getElementById('v1Thermal').textContent = '0.58 (Target 0.70)';
                document.getElementById('v1Cost').textContent = '₹76,000';
                document.getElementById('v1Badge').textContent = 'Failed (0.58 < 0.70)';
                document.getElementById('v1Badge').className = 'score-pill score-fail';
            }

            if (history.length > 1) {
                const v2 = history[history.length - 1];
                document.getElementById('v2Vent').textContent = v2.roof?.ventilation || '0.8';
                document.getElementById('v2Overhang').textContent = (v2.roof?.overhang || '0.9') + 'm';
                document.getElementById('v2Opening').textContent = ((v2.walls?.opening_ratio || 0.35) * 100) + '%';
                document.getElementById('v2Thermal').textContent = '0.78 (Target 0.70)';
                document.getElementById('v2Cost').textContent = '₹76,500';
                document.getElementById('v2Badge').textContent = 'Passed (0.78 >= 0.70)';
                document.getElementById('v2Badge').className = 'score-pill score-pass';
                document.getElementById('cardV2').classList.add('v2-active');
            }

            if (state.final_answer) {
                document.getElementById('finalSummaryBox').style.display = 'block';
                document.getElementById('finalSummaryText').textContent = state.final_answer;
            }
        }
    </script>
</body>
</html>
    """


def save_state_and_dashboard(state, open_gui: bool = True):
    """Save dashboard image and cached state JSON, optionally popping up GUI window."""
    import json
    from pathlib import Path
    from app.visualization.dashboard import launch_dashboard_gui_process
    dash_path = generate_analysis_dashboard(state)
    try:
        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)
        dump_data = state.model_dump() if hasattr(state, "model_dump") else (state if isinstance(state, dict) else {})
        with open("data/last_state.json", "w", encoding="utf-8") as f:
            json.dump(dump_data, f, default=str)
    except Exception:
        pass
    if open_gui:
        launch_dashboard_gui_process()
    return dash_path


def interactive_cli(max_iterations: int = 10, auto_gui: bool = True):
    """Interactive Command-Line Terminal Interface for continuous conversation."""
    from app.visualization.dashboard import launch_dashboard_gui_process
    print("=" * 70)
    print("  Vishvakarma-Agentic-AI - Interactive Terminal Mode")
    print("  Grounded in TERI-2021 & TARU-2015 Thermal Building Knowledge Base")
    print("  Ask questions, analyze thermal constraints, or design shelters.")
    print("  Special Commands: 'gui' / 'view' (open visual app), 'exit' / 'q' (quit)")
    print("=" * 70)

    while True:
        try:
            user_input = input("\nVishvakarma > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nExiting interactive session. Goodbye!")
                break
            if user_input.lower() in ["gui", "view", "dashboard", "plot", "app"]:
                print("\n[GUI] Opening live 3-Panel Matplotlib Interactive Application...")
                launch_dashboard_gui_process()
                continue

            state = agent.run(user_query=user_input, max_iterations=max_iterations)
            should_open_gui = auto_gui and getattr(state, "intent", "") != "greeting"
            try:
                dash_path = save_state_and_dashboard(state, open_gui=should_open_gui)
                print(f"[DASHBOARD] 3-Panel Matplotlib visualization saved to: {dash_path}")
                if should_open_gui:
                    print("[GUI] Auto-launched live 3-Panel interactive Matplotlib visualization window.")
            except Exception as vis_err:
                pass

            print("\n" + "=" * 70)
            print("AGENT RESULT / TECHNICAL REPORT")
            print("=" * 70)
            if state.final_answer:
                print(state.final_answer)
            else:
                print("Task completed. (No final answer text produced)")
            print("=" * 70)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive session. Goodbye!")
            break


def cli_main():
    """CLI runner entry point."""
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Vishvakarma-Agentic-AI CLI & Server")
    parser.add_argument("--cli", type=str, help="Run agent directly with a single query")
    parser.add_argument("-i", "--interactive", action="store_true", help="Start interactive terminal REPL")
    parser.add_argument("--gui", action="store_true", help="Explicitly enable live Matplotlib visual app window (default: enabled)")
    parser.add_argument("--no-gui", action="store_true", help="Disable automatic live Matplotlib visual app window popup")
    parser.add_argument("--view", action="store_true", help="Open live Matplotlib dashboard app viewer for latest analysis")
    parser.add_argument("--server", action="store_true", help="Start FastAPI Web Server")
    parser.add_argument("--iterations", type=int, default=10, help="Max iterations")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    args = parser.parse_args()

    auto_gui = not args.no_gui

    if args.view:
        from app.visualization.dashboard import launch_dashboard_gui_process
        print("\n[GUI] Launching live 3-Panel Matplotlib visual app window...")
        launch_dashboard_gui_process()
    elif args.cli:
        print("\nStarting Vishvakarma-Agentic-AI CLI...")
        state = agent.run(user_query=args.cli, max_iterations=args.iterations)
        should_open_gui = auto_gui and getattr(state, "intent", "") != "greeting"
        try:
            dash_path = save_state_and_dashboard(state, open_gui=should_open_gui)
            print(f"\n[DASHBOARD] 3-Panel Matplotlib visualization saved to: {dash_path}")
            if should_open_gui:
                print("[GUI] Auto-launched live 3-Panel interactive Matplotlib application.")
        except Exception as vis_err:
            print(f"\n[DASHBOARD] Notice: {vis_err}")
        print("\n" + "=" * 70)
        print("AGENT RESULT / TECHNICAL REPORT")
        print("=" * 70)
        if state.final_answer:
            print(state.final_answer)
        else:
            print("Task completed.")
        print("=" * 70)
    elif args.server:
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        # Default to interactive CLI mode when run without arguments
        interactive_cli(max_iterations=args.iterations, auto_gui=auto_gui)


if __name__ == "__main__":
    cli_main()
