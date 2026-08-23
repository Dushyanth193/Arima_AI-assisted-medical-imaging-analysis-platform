"""
Database schema (PostgreSQL via SQLAlchemy).

Two tables matter for this basic version:

    ImplantComponent : the "Implant Sizing Reference" flow-diagram box -
                        one row per (manufacturer, system, component type,
                        size) with its physical dimensions.

    PatientMeasurement : one row per processed patient CT, storing the
                          extracted anatomical measurements and the
                          resulting matching recommendation - lets the
                          "Reference Database" grow over time and lets
                          the API return past results without re-running
                          the pipeline.
"""
from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from src.utils.config import DATABASE_URL

Base = declarative_base()


class ComponentType(str, enum.Enum):
    FEMORAL = "femoral"
    TIBIAL = "tibial"


class ImplantComponent(Base):
    """
    One catalog row = one purchasable implant component size, for one
    manufacturer/system/component-type combination.

    Schema is deliberately vendor-agnostic (Gap 3 from the earlier
    research pass: don't hard-code one manufacturer) - `manufacturer`
    and `system_name` are just string columns, so a second implant
    family can be added by inserting more rows, not changing the schema.
    """
    __tablename__ = "implant_components"

    id = Column(Integer, primary_key=True)
    manufacturer = Column(String(100), nullable=False, index=True)
    system_name = Column(String(100), nullable=False, index=True)   # e.g. "Persona", "Triathlon"
    component_type = Column(Enum(ComponentType), nullable=False, index=True)
    size_label = Column(String(20), nullable=False)                 # e.g. "size 4", "F-63"

    ml_width_mm = Column(Float, nullable=False)
    ap_dimension_mm = Column(Float, nullable=False)

    # Manufacturer-published tolerance, used by the matching engine as
    # the acceptable mismatch band before flagging overhang/under-coverage.
    tolerance_mm = Column(Float, nullable=False, default=1.5)

    created_at = Column(DateTime, default=datetime.utcnow)


class PatientMeasurement(Base):
    __tablename__ = "patient_measurements"

    id = Column(Integer, primary_key=True)
    patient_reference = Column(String(100), nullable=False, index=True)  # de-identified case ID, not PII

    femur_ml_width_mm = Column(Float, nullable=False)
    femur_ap_dimension_mm = Column(Float, nullable=False)
    tibia_ml_width_mm = Column(Float, nullable=False)
    tibia_ap_dimension_mm = Column(Float, nullable=False)

    recommended_femoral_component_id = Column(Integer, ForeignKey("implant_components.id"), nullable=True)
    recommended_tibial_component_id = Column(Integer, ForeignKey("implant_components.id"), nullable=True)
    matching_score_femoral = Column(Float, nullable=True)   # lower = better fit
    matching_score_tibial = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    recommended_femoral_component = relationship("ImplantComponent", foreign_keys=[recommended_femoral_component_id])
    recommended_tibial_component = relationship("ImplantComponent", foreign_keys=[recommended_tibial_component_id])


engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """Create tables if they don't already exist."""
    Base.metadata.create_all(bind=engine)


def get_session():
    """FastAPI-style dependency: yields a session, closes it after use."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
