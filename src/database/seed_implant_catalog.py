"""
Seed the implant_components table with a small SAMPLE catalog.

IMPORTANT: the dimension values below are illustrative placeholders for
development/testing only (representative graduated sizing pattern, not
copied from any manufacturer's real published specification sheet).
Do NOT use these numbers for any actual clinical purpose. Before real
use, replace this with verified dimensions sourced directly from a
manufacturer's public surgical technique guide or a licensed dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.models import ComponentType, ImplantComponent, SessionLocal, init_db



def build_sample_catalog() -> list[ImplantComponent]:
    catalog = []

    # Illustrative femoral sizing ladder: 8 sizes, roughly linear step
    # in ML width, consistent in *shape* with published graduated implant
    # families (~4-5mm steps) but with placeholder starting values.
    femoral_sizes = [
        ("1", 55.6, 51.5), ("2", 58.3, 54.0), ("3", 60.8, 56.5),
        ("4", 63.8, 59.0), ("5", 66.4, 61.5), ("6", 69.3, 64.0),
        ("7", 72.2, 66.5), ("8", 75.5, 69.0),
    ]
    for size_label, ml, ap in femoral_sizes:
        catalog.append(
            ImplantComponent(
                manufacturer="SampleOrtho",
                system_name="Generic-PS-Sample",
                component_type=ComponentType.FEMORAL,
                size_label=size_label,
                ml_width_mm=ml,
                ap_dimension_mm=ap,
                tolerance_mm=1.5,
            )
        )

    tibial_sizes = [
        ("1", 60.0, 40.0), ("2", 63.0, 42.5), ("3", 66.0, 45.0),
        ("4", 69.0, 47.5), ("5", 72.0, 50.0), ("6", 75.5, 52.5),
        ("7", 79.0, 55.0), ("8", 82.5, 57.5),
    ]
    for size_label, ml, ap in tibial_sizes:
        catalog.append(
            ImplantComponent(
                manufacturer="SampleOrtho",
                system_name="Generic-PS-Sample",
                component_type=ComponentType.TIBIAL,
                size_label=size_label,
                ml_width_mm=ml,
                ap_dimension_mm=ap,
                tolerance_mm=1.5,
            )
        )

    return catalog


def seed() -> None:
    init_db()
    session = SessionLocal()
    try:
        existing = session.query(ImplantComponent).count()
        if existing > 0:
            print(f"implant_components already has {existing} rows - skipping seed.")
            return
        session.add_all(build_sample_catalog())
        session.commit()
        print("Seeded sample implant catalog.")
    finally:
        session.close()


if __name__ == "__main__":
    seed()
