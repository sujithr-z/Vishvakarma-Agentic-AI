-- PostgreSQL Schema for Adaptive Shelter Agent Long-Term Experience Memory
-- Relational Schema: Projects -> Buildings -> Design Versions -> Agent Runs -> Steps / Analyses / Experiences

-- 1. Projects Table
CREATE TABLE IF NOT EXISTS projects (
    id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. Buildings Table (Aid-Centric Modular Shelters, Disaster Relief Units, Prefab Housing)
CREATE TABLE IF NOT EXISTS buildings (
    id VARCHAR(64) PRIMARY KEY,
    project_id VARCHAR(64) REFERENCES projects(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    building_type VARCHAR(64) DEFAULT 'modular_shelter',
    climate_zone VARCHAR(64) NOT NULL,
    geometry JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. Design Versions Table (Tracks V1, V2, V3 lineage without overwriting)
CREATE TABLE IF NOT EXISTS design_versions (
    id VARCHAR(64) PRIMARY KEY,
    building_id VARCHAR(64) REFERENCES buildings(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    parent_version_id VARCHAR(64) REFERENCES design_versions(id) ON DELETE SET NULL,
    parameters JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_building_version UNIQUE(building_id, version_number)
);

-- 4. Agent Runs Table (Autonomous Execution Episodes)
CREATE TABLE IF NOT EXISTS agent_runs (
    id VARCHAR(64) PRIMARY KEY,
    project_id VARCHAR(64) REFERENCES projects(id) ON DELETE SET NULL,
    building_id VARCHAR(64) REFERENCES buildings(id) ON DELETE SET NULL,
    user_query TEXT NOT NULL,
    intent VARCHAR(64) NOT NULL,
    status VARCHAR(32) NOT NULL,
    total_steps INTEGER DEFAULT 0,
    final_response TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 5. Agent Steps Table (Granular Step-by-Step Execution History)
CREATE TABLE IF NOT EXISTS agent_steps (
    id VARCHAR(64) PRIMARY KEY,
    agent_run_id VARCHAR(64) REFERENCES agent_runs(id) ON DELETE CASCADE,
    step_number INTEGER NOT NULL,
    state_before JSONB,
    decision JSONB NOT NULL,
    tool_called VARCHAR(64),
    tool_arguments JSONB,
    tool_result JSONB,
    state_after JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_run_step UNIQUE(agent_run_id, step_number)
);

-- 6. Climate Observations Table
CREATE TABLE IF NOT EXISTS climate_observations (
    id VARCHAR(64) PRIMARY KEY,
    agent_run_id VARCHAR(64) REFERENCES agent_runs(id) ON DELETE CASCADE,
    location VARCHAR(128) NOT NULL,
    climate_zone VARCHAR(64) NOT NULL,
    temperature_c NUMERIC(5, 2),
    rh_percent NUMERIC(5, 2),
    wind_speed_ms NUMERIC(5, 2),
    solar_radiation_w_m2 NUMERIC(7, 2),
    trm_c NUMERIC(5, 2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 7. Thermal Analyses Table
CREATE TABLE IF NOT EXISTS thermal_analyses (
    id VARCHAR(64) PRIMARY KEY,
    agent_run_id VARCHAR(64) REFERENCES agent_runs(id) ON DELETE CASCADE,
    design_version_id VARCHAR(64) REFERENCES design_versions(id) ON DELETE SET NULL,
    indoor_temp_c NUMERIC(5, 2),
    neutral_temp_c NUMERIC(5, 2),
    upper_90_limit NUMERIC(5, 2),
    lower_90_limit NUMERIC(5, 2),
    comfort_score NUMERIC(4, 3),
    passed BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 8. Thermal Constraints Table (TC-001 through TC-009 Records)
CREATE TABLE IF NOT EXISTS thermal_constraints (
    id VARCHAR(64) PRIMARY KEY,
    thermal_analysis_id VARCHAR(64) REFERENCES thermal_analyses(id) ON DELETE CASCADE,
    constraint_id VARCHAR(32) NOT NULL,
    parameter VARCHAR(64) NOT NULL,
    actual_value TEXT,
    threshold_value TEXT,
    status VARCHAR(32) NOT NULL,
    reason TEXT,
    source VARCHAR(128),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 9. Improvements Table (Recorded Modifications & Expected Delta T)
CREATE TABLE IF NOT EXISTS improvements (
    id VARCHAR(64) PRIMARY KEY,
    agent_run_id VARCHAR(64) REFERENCES agent_runs(id) ON DELETE CASCADE,
    parameter VARCHAR(64) NOT NULL,
    current_value TEXT,
    recommended_value TEXT,
    expected_effect TEXT,
    reason TEXT,
    source VARCHAR(128),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 10. Cost Results Table (Itemized 2014 INR Historical & Projected Costs)
CREATE TABLE IF NOT EXISTS cost_results (
    id VARCHAR(64) PRIMARY KEY,
    agent_run_id VARCHAR(64) REFERENCES agent_runs(id) ON DELETE CASCADE,
    total_cost_inr NUMERIC(12, 2) NOT NULL,
    itemized JSONB NOT NULL,
    is_historical BOOLEAN DEFAULT TRUE,
    cost_year INTEGER DEFAULT 2014,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 11. Experiences Table (Complete Structured Episode for Long-Term Retrieval)
CREATE TABLE IF NOT EXISTS experiences (
    id VARCHAR(64) PRIMARY KEY,
    project_id VARCHAR(64) REFERENCES projects(id) ON DELETE SET NULL,
    building_id VARCHAR(64) REFERENCES buildings(id) ON DELETE SET NULL,
    agent_run_id VARCHAR(64) REFERENCES agent_runs(id) ON DELETE SET NULL,
    user_request TEXT NOT NULL,
    intent VARCHAR(64) NOT NULL,
    climate_context JSONB NOT NULL,
    initial_design JSONB,
    thermal_result JSONB,
    constraints_detected JSONB,
    actions_taken JSONB,
    improvements JSONB,
    final_design JSONB,
    final_evaluation JSONB,
    cost_impact JSONB,
    success BOOLEAN NOT NULL,
    failure_reason TEXT,
    key_learnings TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for Fast Experience and Run Retrieval
CREATE INDEX IF NOT EXISTS idx_experiences_intent ON experiences(intent);
CREATE INDEX IF NOT EXISTS idx_experiences_success ON experiences(success);
CREATE INDEX IF NOT EXISTS idx_buildings_climate ON buildings(climate_zone);
CREATE INDEX IF NOT EXISTS idx_agent_steps_run ON agent_steps(agent_run_id, step_number);
CREATE INDEX IF NOT EXISTS idx_design_versions_building ON design_versions(building_id, version_number);
