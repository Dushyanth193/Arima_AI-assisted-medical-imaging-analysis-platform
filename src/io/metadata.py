"""
Image Metadata Extraction Utility
Extracts spatial and imaging metadata from SimpleITK Image objects.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import SimpleITK as sitk


@dataclass
class ImageMetadata:
    dimensions: tuple[int, ...]
    spacing: tuple[float, ...]
    origin: tuple[float, ...]
    direction: tuple[float, ...]
    modality: str = "CT/MRI"

    def to_dict(self) -> dict:
        return asdict(self)


def extract_image_metadata(image: sitk.Image, modality: str = "CT") -> ImageMetadata:
    """Extract spatial metadata from a SimpleITK Image."""
    return ImageMetadata(
        dimensions=image.GetSize(),
        spacing=image.GetSpacing(),
        origin=image.GetOrigin(),
        direction=image.GetDirection(),
        modality=modality,
    )
