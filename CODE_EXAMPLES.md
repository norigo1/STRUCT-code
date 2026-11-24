# Code examples

These examples are intended to be run from the repository root with the
project's Python dependencies installed. Replace the example paths and
coordinates with values from your data. The HELIOS examples only create scene
and survey XML files; they do not launch simulations.

## HELIOS scene and survey preparation

### Create a radius-limited scene around one scan position

This includes terrain and stems whose mesh centroids are within the requested
radius. A matching leaf file is included when its filename replaces
`_mesh.obj` with `_leaves.obj`.

```python
from pathlib import Path

from helios_simulations import generate_scene_for_radius

data_dir = Path("data/example_plot")
stem_dir = data_dir / "wood"
leaf_dir = data_dir / "leaves"
terrain_dir = data_dir / "terrain"
terrain_files = sorted(terrain_dir.glob("*.obj"))

if not terrain_files:
    raise FileNotFoundError(f"No terrain OBJ found in {terrain_dir}")

parts = generate_scene_for_radius(
    stem_files=sorted(stem_dir.glob("*_mesh.obj")),
    leaf_files=sorted(leaf_dir.glob("*_leaves.obj")),
    terrain_file=terrain_files[0],
    scan_position=(10.0, 20.0),
    radius=8.0,
    outpath="demo_output/scan_001_scene.xml",
    scene_name="scan_001",
)

print(f"Included {len(parts) - 1} vegetation meshes and the terrain.")
```

Coordinates and radius must use the same units as the OBJ meshes. The returned
`parts` list can be used to inspect exactly which mesh paths were included.

### Split a scan grid into small scenes and matching surveys

This creates one radius-limited scene per grid location and a survey with one
leg for each scene. The survey template must contain a `<survey>` element.

```python
from pathlib import Path

from helios_simulations import (
    generate_radius_scenes_from_grid,
    rebuild_survey_leg_from_point,
)

data_dir = Path("data/example_plot")
output_dir = Path("demo_output")
survey_template = Path("data/survey_template.xml")
scene_dir = output_dir / "scenes"
survey_dir = output_dir / "surveys"

scene_paths = generate_radius_scenes_from_grid(
    xmin=0.0,
    xmax=10.0,
    ymin=0.0,
    ymax=10.0,
    x_step=5.0,
    y_step=5.0,
    stem_dir=str(data_dir / "wood"),
    leaf_dir=str(data_dir / "leaves"),
    terrain_dir=str(data_dir / "terrain"),
    radius=4.0,
    outdir=str(scene_dir),
    prefix="plot",
)

survey_dir.mkdir(parents=True, exist_ok=True)
for scene_path_text in scene_paths:
    scene_path = Path(scene_path_text)
    _, x_text, y_text = scene_path.stem.rsplit("_", 2)
    survey_path = survey_dir / f"{scene_path.stem}_survey.xml"

    rebuild_survey_leg_from_point(
        path_survey=str(survey_template),
        outpath=str(survey_path),
        x=float(x_text),
        y=float(y_text),
        scene=str(scene_path),
        scene_name=scene_path.stem,
    )

print(f"Created {len(scene_paths)} scene/survey pairs.")
```

For a single survey that uses one scene at every grid point, use
`rebuild_survey_legs_from_grid` instead, setting its `scene` and `scene_name`
arguments if they are not already present in the template. For real scan
positions in a CSV, `rebuild_survey_legs_from_scanpos` can select scan IDs such
as `[1, 3]`; names in the CSV must follow the `ScanPos###` format.

## Occlusion and point-cloud LAI/PAI

### Read vertical profiles from normalized Occpy grids

The `.npy` files must already exist in `merged_voxgrids` under `out_dir`.
`plot_dim` and `vox_dim` should match the grid used to produce them.

```python
import json
import os
from pathlib import Path

from comparing_occlusion_LAI import OccpyAnalysis

output_dir = Path("data/occpy_output").resolve()
config = {
    "out_dir": str(output_dir) + os.sep,
    "plot_dim": [0.0, 0.0, 0.0, 20.0, 20.0, 15.0],
    "vox_dim": 0.5,
}

analysis = OccpyAnalysis(json.dumps(config))
profiles = {
    "occluded": analysis.get_occluded_vertical_profile(),
    "hit": analysis.get_hit_vertical_profile(),
    "empty": analysis.get_empty_vertical_profile(),
    "unobserved": analysis.get_unobserved_vertical_profile(),
}

for name, profile in profiles.items():
    print(f"{name}: first height/profile rows\n{profile[:5]}")

absolute_occluded = analysis.get_occluded_vertical_profile("absolute")
print(f"Absolute occluded-voxel counts:\n{absolute_occluded[:5]}")
```

Each profile contains voxel-center heights in column one and the selected
classification value in column two. The default is percentage; `"absolute"`
returns counts.

### Compare first-return and weighted profiles with a small ground scan

This synthetic scan contains three emitted pulses: one with two returns, one
with no return, and one with a single return. The no-return pulse stays in the
denominator for the gap estimate.

```python
from dataclasses import replace

import numpy as np

from comparing_occlusion_LAI import (
    CaldersProfileInput,
    CaldersScan,
    PointCloudVegetationAnalysis,
)

scan = CaldersScan(
    pulse_zenith=np.array([52.5, 52.5, 52.5]),
    pulse_azimuth=np.array([0.0, 0.0, 0.0]),
    pulse_return_count=np.array([2, 0, 1]),
    return_pulse_index=np.array([0, 0, 2]),
    return_height=np.array([0.5, 1.5, 1.5]),
    return_number=np.array([1, 2, 1]),
    return_classification=np.array([1, 2, 1]),
    min_zenith=50.0,
    max_zenith=55.0,
)
settings = CaldersProfileInput(
    scans=(scan,),
    max_height=3.0,
    height_bin_size=1.0,
    zenith_bin_size=5.0,
)
analysis = PointCloudVegetationAnalysis(
    "synthetic scan",
    source="real",
    platform="ground",
)

first_return = analysis.calculate_pavd_profile(settings)
weighted_settings = replace(settings, weighted=True)
weighted = analysis.calculate_pavd_profile(weighted_settings)
filtered_settings = replace(weighted_settings, excluded_classes=(2,))
filtered = analysis.calculate_pavd_profile(filtered_settings)

print("First-return gap probabilities:", first_return["pgap"])
print("Weighted gap probabilities:", weighted["pgap"])
print("Weighted, class-filtered probabilities:", filtered["pgap"])
print("Weighted hinge PAI:", analysis.calculate_pai(weighted)["hinge"]["pai"])
```

This small example uses one zenith bin, so it is intended to illustrate pulse
counting and return weighting—not to compare all PAI estimators. For a meaningful
hinge-versus-linear comparison, use a scan with sufficient zenith-angle coverage.

### Run the complete workflow on HELIOS++ LAS and pulse outputs

This expects an uncompressed LAS file with the `fullwaveIndex` extra dimension
and a matching HELIOS++ pulse file containing every emitted pulse, including
pulses with no detected return. The pulse file and LAS must come from the same
single-position ground scan.

```python
from pathlib import Path

from comparing_occlusion_LAI import (
    CaldersProfileInput,
    PointCloudVegetationAnalysis,
)

analysis = PointCloudVegetationAnalysis(
    point_cloud_path=Path("data/scan_001.las"),
    source="simulated",
    platform="ground",
    pulse_path=Path("data/scan_001_pulses.txt"),
    ground_elevation=0.0,
    calders_settings=CaldersProfileInput(
        scans=(),
        min_height=0.0,
        max_height=25.0,
        height_bin_size=0.5,
        zenith_bin_size=5.0,
    ),
)
pavd_profile, pai = analysis.run()

print("PAVD profile shape:", pavd_profile["hinge_pavd"].shape)
for method, result in pai.items():
    print(f"{method} PAI:", result["pai"])
```

Set `ground_elevation` to the ground reference in the levelled coordinate frame;
for uneven ground, supply one elevation per LAS return. A LAS file alone is not
enough for this workflow because it does not include emitted pulses with no
return. The linear estimator can also be unavailable (NaN) when the scan has
insufficient zenith-angle coverage.

## Canopy cover and scene area indices

### Compare canopy-cover ground-area definitions

This example assumes scene mesh names such as `tree_1_leaves.obj` and
`tree_1_mesh.obj`, stored in `leaves` and `wood` directories. Adapt the tree
pattern and IDs to match the scene.

```python
from pathlib import Path

from cc_lai_alteration import CanopyCover

scene_path = Path("data/example_plot/scene.xml")
tree_ids = range(1, 5)
mesh_directories = [
    str(scene_path.parent / "leaves"),
    str(scene_path.parent / "wood"),
]
analysis = CanopyCover(str(scene_path))

for ground_method in ("max_extent", "convexhull", "can_outline"):
    canopy_area, ground_area, cover, outlines, ground_outline = (
        analysis.calculate_canopy_cover(
            directories=mesh_directories,
            globstring="tree_{iter}_*",
            iterable=tree_ids,
            ga=ground_method,
        )
    )
    print(
        f"{ground_method}: cover={cover:.3f}, "
        f"canopy_area={canopy_area:.2f}, ground_area={ground_area:.2f}"
    )
```

`can_outline` uses the canopy outline as the ground area; for disconnected
canopy components, the method may fall back to a convex hull and issue a
warning.

### Compare LAI and PAI, clipping, and height profiles

The bounds below are illustrative. Choose XY bounds that match the plot or
sub-area you want to measure. By default, scene asset paths must contain
`"leaves"` and `"wood"` for the two area categories; pass custom identifiers to
`CalculateVegAreaIndex` if your paths use different names.

```python
from cc_lai_alteration import CalculateVegAreaIndex

scene_path = "data/example_plot/scene.xml"
bounds = (0.0, 20.0, 0.0, 10.0)
calculator = CalculateVegAreaIndex(scene_path)

for method in ("centroid", "clip"):
    leaf_area, ground_area, _ = calculator.calculate_leaf_area_index(
        *bounds,
        method=method,
    )
    plant_area, _, _ = calculator.calculate_plant_area_index(
        *bounds,
        method=method,
    )
    print(
        f"{method}: LAI={leaf_area / ground_area:.3f}, "
        f"PAI={plant_area / ground_area:.3f}"
    )

lai_profile = calculator.calculate_leaf_area_index_profile(
    *bounds,
    height_bins=10,
    height_range=(0.0, 20.0),
)
print("Height-center / LAI-contribution rows:\n", lai_profile)
```

`"clip"` clips triangle faces to box bounds; `"centroid"` assigns each whole
face by its centroid. The profile contains height-band centers and LAI
contributions; the explicit `height_range` must contain all included face
heights and should be the same when comparing scenes.

### Reduce LAI and verify the generated scene

Run this on a working copy of the scene and meshes: generated leaf PLY files are
written alongside the original leaf meshes. Set `target_lai` to a value below
the current LAI of the scene.

```python
from cc_lai_alteration import LeafAreaIndexNewScene

scene_path = "data/example_plot/scene.xml"
target_lai = 1.0

reducer = LeafAreaIndexNewScene(
    scene=scene_path,
    new_lai=target_lai,
    height_bins=10,
)
reducer.reduce_lai()
reduced_scene = reducer.create_new_scene()
measured_lai, measured_pai = reducer.calculate_new_indices()

print("Reduced scene:", reduced_scene)
print(
    f"Original LAI: {reducer.orig_lai:.3f}; "
    f"target LAI: {target_lai:.3f}; measured LAI: {measured_lai:.3f}"
)
print(f"Measured PAI: {measured_pai:.3f}")
```

The reduction retains whole connected leaf components while distributing the
thinning across height bands, so the resulting LAI is an approximation to the
requested target.
