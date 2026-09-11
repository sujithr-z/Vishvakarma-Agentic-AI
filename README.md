# Vishvakarma-Agentic-AI

> **Agentic AI for Climate-Resilient Modular Shelter Design & Thermal Comfort Engineering**  
> Grounded in **NBC 2016**, **TERI-2021**, and **TARU-2015** Indian residential building standards. Powered by local **Qwen 1.8B** reasoning, deterministic thermal physics tools, persistent **PostgreSQL experience memory**, and an interactive **3-Panel Matplotlib visual application**.

---

## 🏛️ System Architecture

```text
       ┌────────────────────────────────────────────────────────┐
       │                   USER / CLI / WEB UI                  │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │             Qwen 1.8B (Local Ollama LLM)               │
       │              Reasoning & Decision Making               │
       └───────────────────────────┬────────────────────────────┘
                                   │
               ┌───────────────────┼───────────────────┐
               ▼                   ▼                   ▼
    ┌──────────────────────┐ ┌───────────┐ ┌──────────────────────┐
    │   RAG KNOWLEDGE      │ │   STATE   │ │ DETERMINISTIC TOOLS  │
    │  TERI-2021 / TARU-15 │ │ SITUATION │ │  Physics, NBC 2016,  │
    │ Engineering Standards│ │  TRACKER  │ │  Cost & Structure    │
    └──────────────────────┘ └─────┬─────┘ └──────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │               POSTGRESQL EXPERIENCE MEMORY             │
       │  11 Relational Tables | Design Lineage (V1 → V2 → V3)  │
       │     Episodic Memory | Constraint Audits | Runs & Steps │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │      MATPLOTLIB 3-PANEL INTERACTIVE VISUAL APP         │
       │  Fig 1: Thermal Graphs | Fig 2: Architecture Plan      │
       │          Fig 3: Live 3D Simulation Video Player        │
       └────────────────────────────────────────────────────────┘
```

---

## 🌟 Key Features

1. **Deterministic Physics & Strict Standard Grounding**:
   - **NBC 2016 Adaptive Thermal Comfort Equations**:
     - *Naturally Ventilated (NV)*: $T_n = 0.54 \cdot T_{rm} + 12.83$, 90% Acceptability Band: $T_n \pm 2.38\text{ °C}$
     - *Mixed-Mode (MM)*: $T_n = 0.28 \cdot T_{rm} + 17.87$, 90% Acceptability Band: $T_n \pm 3.48\text{ °C}$
   - **TERI-2021 Guidelines**: Air velocity comfort offsets ($0.5 - 1.5\text{ m/s}$ delivering $3.0 - 5.0\text{ °C}$ relief), window-to-floor ventilation ratio ($\ge 20\%$), roof insulation ($R \ge 3.5\text{ m}^2\cdot\text{K/W}$), and roof reflectance ($\ge 0.60$).
   - **TARU-2015 Cool-Roof Interventions**: Field-measured $\Delta T$ ranges ($2.0 - 4.0\text{ °C}$) and itemized 2014 INR material costs.

2. **Persistent PostgreSQL Experience Memory**:
   - Full 11-table schema tracking projects, modular buildings, immutable design version lineage ($V1 \rightarrow V2 \rightarrow V3$), agent reasoning runs, tool traces, environmental observations, thermal constraints, and episodic experience records.
   - Automatic retrieval injects relevant past design successes and failures into the LLM prompt context to prevent repeating past mistakes.

3. **Interactive 3-Panel Matplotlib Application Window**:
   - **Figure 1 (Bottom Row)**: Live thermal & climate graphs (Operative Temp vs NBC 2016 Adaptive Limit, Airflow with $0.5 - 1.5\text{ m/s}$ shaded band, Relative Humidity on twin-axis, and Comfort Score Evolution).
   - **Figure 2 (Top Left)**: High-resolution shelter architectural blueprint (`img_plot/arch_plan.jpeg`).
   - **Figure 3 (Top Right)**: **Live looping 3D CFD / walkthrough video player** (`video_plot/model_3d.mp4`) playing at ~25 FPS via OpenCV and Matplotlib animation!

4. **Zero-Hallucination Guardrails**:
   - Small LLM repetition pruning and stop-sequence protection.
   - Resilient regex action parser and state-based deterministic workflow recovery.
   - Intelligent greeting handling (`hi`, `hello`, `help`, `who are you`).

---

## 📂 Directory Structure

```text
adaptive-shelter-agent/
├── app/
│   ├── main.py                     # FastAPI server, CLI runner, and REPL entry point
│   ├── agent/
│   │   ├── agent.py                # Central agent controller & execution loop
│   │   ├── state.py                # AgentState & situation context Pydantic models
│   │   ├── planner.py              # System prompts, intent detection & prompt builder
│   │   ├── parser.py               # Robust JSON action parser with regex fallback
│   │   └── formatter.py            # Technical thermal analysis report formatter
│   ├── llm/
│   │   └── ollama_client.py        # Local Ollama HTTP client with anti-repetition options
│   ├── tools/
│   │   ├── registry.py             # Tool registry & metadata schemas
│   │   ├── shelter.py              # Modular shelter generator & modifier
│   │   ├── climate.py              # Environmental climate lookup
│   │   ├── thermal.py              # Heuristic thermal comfort evaluator
│   │   ├── cost.py                 # Deterministic cost & improvement calculator
│   │   ├── structure.py            # Structural span and load safety checks
│   │   └── constraints.py          # Grounded TC-001..TC-009 constraint engine
│   ├── memory/
│   │   ├── models.py               # SQLAlchemy 11-table relational models
│   │   ├── database.py             # Session scope & SQLite/PostgreSQL engine
│   │   ├── repository.py           # Experience repository & lineage tracking
│   │   ├── memory.py               # Experience memory wrapper interface
│   │   └── schema.sql              # Raw PostgreSQL DDL schema definition
│   ├── visualization/
│   │   ├── thermal_plots.py        # Figure 1: Multi-axis thermal & climate graphs
│   │   ├── architecture_plot.py    # Figure 2: Architecture plan blueprint renderer
│   │   ├── video_placeholder.py    # Figure 3: OpenCV live video frame animation
│   │   └── dashboard.py            # Combined 3-panel dashboard & GUI process launcher
│   └── knowledge/
│       ├── retriever.py            # Token overlap & semantic RAG search
│       └── knowledge.json          # Curated TERI-2021 & TARU-2015 knowledge base
├── img_plot/
│   └── arch_plan.jpeg              # Shelter architectural drawing / blueprint
├── video_plot/
│   └── model_3d.mp4                # 3D shelter simulation & walkthrough video
├── tests/
│   ├── test_agent.py               # 11 core agent capability & greeting tests
│   ├── test_tools.py               # Deterministic tools and constraint engine tests
│   ├── test_parser.py              # JSON parsing and error recovery tests
│   ├── test_memory_postgres.py     # PostgreSQL relational memory & lineage tests
│   └── test_visualization.py       # Matplotlib figures, animation, and export tests
├── data/
│   ├── analysis_dashboard.png      # Latest generated 3-panel dashboard image
│   ├── last_state.json             # Cached execution state for GUI viewer
│   └── shelter_examples.json       # Benchmark design requirement prompts
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
- **Python 3.10+** (Tested on Python 3.13 & 3.14 on Windows, Linux, macOS)
- **[Ollama](https://ollama.ai)** installed and running.
- **PostgreSQL** (Optional, automatic SQLite fallback is built-in for local development).

### 2. Pull the Local LLM Model
```bash
ollama pull qwen:1.8b
```

### 3. Install Python Dependencies
```bash
# From the repository root
python -m pip install -r requirements.txt
```

### 4. Configure Environment (Optional)
Create or edit `.env` in the project root:
```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen:1.8b
# Optional PostgreSQL connection (falls back to local SQLite if omitted):
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/shelter_memory
```

---

## 💻 How to Run

### Mode 1: Interactive Terminal REPL (Continuous Conversation)
Start the continuous interactive CLI:
```bash
python -m app.main
```
Inside the interactive session:
- Type **`hi`** to see the welcome screen, capabilities, and prompt templates.
- Enter any shelter query:
  ```text
  Vishvakarma > Analyze the model house. What thermal constraints could it exceed?
  ```
- Type **`gui`** or **`view`** at any prompt to immediately pop up the live 3-panel Matplotlib visual application window.
- Type **`exit`** or **`q`** to quit.

---

### Mode 2: Single CLI Query with Automatic Live GUI Window
Run a single query and automatically open the interactive visual window alongside the terminal report:
```bash
python -m app.main --cli "Analyze the model house. What thermal constraints could it exceed?" --gui
```

---

### Mode 3: Standalone GUI Viewer
Launch the interactive Matplotlib app to inspect the latest analysis without re-running the agent:
```bash
python -m app.main --view
```

---

### Mode 4: Interactive Web Application & FastAPI Server
Start the web dashboard server:
```bash
python -m app.main --server
```
*(or `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`)*

Open your browser at **[http://localhost:8000](http://localhost:8000)** to:
- Select preset benchmark prompts (*Thermal Constraints*, *Kerala 5P*, *Assam 4P*).
- Watch real-time streaming traces of Qwen 1.8B's thoughts and tool calls.
- Inspect $V1 \text{ Baseline} \rightarrow V2 \text{ Adaptive}$ metric cards.
- View the embedded responsive Matplotlib visualization dashboard with 1-click refresh.

---

## 🧪 Running the Test Suite

Run the full automated test suite covering all 41 test cases:

```bash
pytest tests/ -v
```

**Test Coverage Summary**:
- `tests/test_agent.py`: 15 core agent capability & critical reasoning audit tests (Observe -> Reason -> Act loop, unbiased prompt, tool failure recovery, invalid action recovery, conditional tool calling Tests A-E, labeled evidence, design optimization cycle, greetings).
- `tests/test_memory_postgres.py`: 7 PostgreSQL relational memory tests (11-table schema creation, immutable design lineage, run and step logging, episodic experience save/retrieve, prompt injection).
- `tests/test_tools.py`: 7 deterministic tool tests (climate, constraint engine TC-001..TC-009, cool-roof rates, RAG retrieval).
- `tests/test_parser.py`: 4 parser robustness tests (markdown wrappers, trailing commas, repetition sanitization, natural language fallback).
- `tests/test_visualization.py`: 8 Matplotlib visualization tests (Figure 1 metrics extraction & rendering, Figure 2 blueprint loading/fallback, Figure 3 OpenCV video frame animation, 3-panel dashboard PNG export).

---

## 📊 Thermal Constraint Index (TERI-2021 & TARU-2015)

| Code | Constraint Name | Mathematical / Engineering Threshold | Primary Source |
| :--- | :--- | :--- | :--- |
| **TC-001** | High Indoor Operative Temperature | $T_{in} \le T_n + 2.38\text{ °C}$ (Adaptive 90% Upper Limit) | TERI-2021 Table 2 |
| **TC-002** | Mixed-Mode Exceedance | $T_{in} \le T_n + 3.48\text{ °C}$ | TERI-2021 Table 2 |
| **TC-003** | Extreme Air Temperature Discomfort | $T_{air} \le 30.0\text{ °C}$ in hot-humid zones | TERI-2021 §4.4 |
| **TC-004** | Insufficient Air Speed for Cooling Offset | $v_{air} \ge 0.5 - 1.5\text{ m/s}$ (Provides $3 - 5\text{ °C}$ relief) | TERI-2021 Fig. 37 & Table 9 |
| **TC-005** | Inadequate Operable Opening Area | Opening-to-floor area ratio $\ge 20.0\%$ | TERI-2021 §4.4.5 |
| **TC-006** | Cool-Roof Qualification Failure | Solar Reflectance $\ge 0.60$, High SRI | TERI-2021 §4.9 |
| **TC-007** | Roof Insulation Below Code | Roof $R$-value $\ge 3.5\text{ m}^2\cdot\text{K/W}$ | TERI-2021 Table 5 |
| **TC-008** | Inadequate Roof Overhang Shading | Eaves projection $\ge 0.8\text{ m}$ (Solar & Monsoon shield) | TERI-2021 §4.5 |
| **TC-009** | High Relative Humidity Wet-Bulb Limit | $\text{RH} > 70\%$ requires air movement over evaporative coolers | TERI-2021 §4.6 |

---

## 👥 Authors & Contributors

- **Sujith R** ([@sujithr-z](https://github.com/sujithr-z))
- **Abhiyukth Krishna** ([@abhiyukth-krishna](https://github.com/abhiyukth-krishna)) — `krishnaabhiyukth@gmail.com`
- **Shamsuddin** ([@tricksterunknown](https://github.com/tricksterunknown)) — `shamsshamsuddin0585@gmail.com`
- **Kamalapriyan** ([@MadScientist-Shadow](https://github.com/MadScientist-Shadow)) — `kamalapriyan8@gmail.com`
- **Krishnapriya** ([@jewelnbule](https://github.com/jewelnbule)) — `krishnapriya2654@gmail.com`

---

## 📜 License

This project is released under the **MIT License**.
