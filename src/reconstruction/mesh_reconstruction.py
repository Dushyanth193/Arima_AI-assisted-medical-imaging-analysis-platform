"""
3D Reconstruction
==================
Converts a binary label mask (femur or tibia) into a 3D surface mesh
using marching cubes, via PyVista (which wraps VTK). Matches the
"3D reconstruction: VTK / PyVista" row of the tech stack.

Used by:
    - the "Visualization and Verification" flow-diagram stage (render
      the bone surface for the clinician to inspect)
    - optionally by the implant-matching engine for surface-based fit
      checks beyond simple width/AP scalar comparison (documented as a
      future improvement, not implemented in this basic version)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import SimpleITK as sitk

from src.utils.config import LABELS


def label_mask_to_mesh(label_image: sitk.Image, label_value: int, smoothing_iterations: int = 20) -> pv.PolyData:
    """
    Extract an isosurface for a single label from a multi-label mask.

    Parameters
    ----------
    label_image : sitk.Image
        Integer label mask (0=background, 1=femur, 2=tibia).
    label_value : int
        Which label to extract (see src.utils.config.LABELS).
    smoothing_iterations : int
        Laplacian smoothing passes to reduce voxel "staircase" artifacts
        on the mesh surface, purely for visualization/inspection quality
        - does not affect the scalar measurements computed elsewhere,
        which are taken from the raw voxel mask, not the smoothed mesh.

    Returns
    -------
    pv.PolyData
        Triangulated surface mesh in physical (mm) coordinates.
    """
    array = sitk.GetArrayFromImage(label_image)
    binary = (array == label_value).astype(np.uint8)

    spacing = label_image.GetSpacing()       # (x, y, z)
    origin = label_image.GetOrigin()          # (x, y, z)

    # PyVista's ImageData expects (x, y, z) axis order; SimpleITK arrays
    # are (z, y, x), so transpose before wrapping.
    binary_xyz = np.transpose(binary, (2, 1, 0))

    grid = pv.ImageData(dimensions=binary_xyz.shape)
    grid.spacing = spacing
    grid.origin = origin
    grid.point_data["mask"] = binary_xyz.flatten(order="F")

    surface = grid.contour(isosurfaces=[0.5], scalars="mask")
    surface = surface.smooth(n_iter=smoothing_iterations, relaxation_factor=0.1)
    surface = surface.clean()

    return surface


def reconstruct_bones(label_image: sitk.Image) -> dict:
    """Reconstruct both femur and tibia surfaces from one label mask."""
    meshes = {}
    for name in ("femur", "tibia"):
        meshes[name] = label_mask_to_mesh(label_image, LABELS[name])
    return meshes


def save_meshes(meshes: dict, output_dir: Path, file_format: str = "vtp") -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ext = file_format.lower().lstrip(".")
    for name, mesh in meshes.items():
        out_path = output_dir / f"{name}.{ext}"
        mesh.save(str(out_path))


def export_mesh_file(mesh: pv.PolyData, output_path: str | Path) -> Path:
    """Export a single mesh to STL, PLY, or VTK/VTP format."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(str(path))
    return path


def render_preview_png(meshes: dict, output_path: Path) -> None:
    """
    Off-screen render of femur + tibia meshes to a PNG, for quick visual
    QC without needing an interactive session (e.g. from a backend job).
    """
    plotter = pv.Plotter(off_screen=True)
    colors = {"femur": "ivory", "tibia": "lightblue"}
    for name, mesh in meshes.items():
        plotter.add_mesh(mesh, color=colors.get(name, "white"), opacity=0.9, smooth_shading=True)
    plotter.camera_position = "iso"
    plotter.screenshot(str(output_path))
    plotter.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reconstruct 3D femur/tibia meshes from a label mask.")
    parser.add_argument("label_mask_path", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

    img = sitk.ReadImage(args.label_mask_path)
    meshes = reconstruct_bones(img)
    save_meshes(meshes, Path(args.output_dir))
    render_preview_png(meshes, Path(args.output_dir) / "preview.png")
    print(f"Saved meshes and preview render to {args.output_dir}")
