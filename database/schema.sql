-- Schema for Knee Implant Sizing Database (PostgreSQL / SQLite compatible)

CREATE TABLE IF NOT EXISTS implant_components (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    manufacturer VARCHAR(100) NOT NULL,
    system_name VARCHAR(100) NOT NULL,
    component_type VARCHAR(20) NOT NULL, -- 'femoral' or 'tibial'
    size_label VARCHAR(20) NOT NULL,
    ml_width_mm REAL NOT NULL,
    ap_dimension_mm REAL NOT NULL,
    tolerance_mm REAL NOT NULL DEFAULT 1.5,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS patient_measurements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_reference VARCHAR(100) NOT NULL,
    femur_ml_width_mm REAL NOT NULL,
    femur_ap_dimension_mm REAL NOT NULL,
    tibia_ml_width_mm REAL NOT NULL,
    tibia_ap_dimension_mm REAL NOT NULL,
    recommended_femoral_component_id INTEGER,
    recommended_tibial_component_id INTEGER,
    matching_score_femoral REAL,
    matching_score_tibial REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(recommended_femoral_component_id) REFERENCES implant_components(id),
    FOREIGN KEY(recommended_tibial_component_id) REFERENCES implant_components(id)
);

CREATE INDEX IF NOT EXISTS idx_implant_manuf_sys ON implant_components(manufacturer, system_name);
CREATE INDEX IF NOT EXISTS idx_patient_ref ON patient_measurements(patient_reference);
