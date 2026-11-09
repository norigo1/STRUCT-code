"""Unit tests for mesh reading and leaf-area reduction helpers."""

import importlib
import pathlib
import sys
import tempfile
import types
import unittest
import xml.etree.ElementTree as ET

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import scipy  # noqa: F401
    import shapely  # noqa: F401
    REAL_GEOMETRY_AVAILABLE = True
except ImportError:
    REAL_GEOMETRY_AVAILABLE = False


def _install_optional_dependency_stubs():
    """Allow unit tests for mesh/area code when plotting and geometry packages are absent."""
    if importlib.util.find_spec("matplotlib") is None:
        matplotlib = types.ModuleType("matplotlib")
        matplotlib.pyplot = types.ModuleType("matplotlib.pyplot")
        sys.modules["matplotlib"] = matplotlib
        sys.modules["matplotlib.pyplot"] = matplotlib.pyplot
    try:
        import scipy  # noqa: F401
    except ImportError:
        scipy = types.ModuleType("scipy")
        scipy.spatial = types.ModuleType("scipy.spatial")
        scipy.spatial.ConvexHull = scipy.spatial.Delaunay = object
        scipy.interpolate = types.ModuleType("scipy.interpolate")
        scipy.interpolate.interp1d = object
        sys.modules["scipy"] = scipy
        sys.modules["scipy.spatial"] = scipy.spatial
        sys.modules["scipy.interpolate"] = scipy.interpolate
    try:
        import shapely  # noqa: F401
    except ImportError:
        shapely = types.ModuleType("shapely")
        shapely.linearrings = shapely.union_all = shapely.polygons = object
        shapely.geometry = types.ModuleType("shapely.geometry")
        shapely.geometry.Polygon = shapely.geometry.MultiPoint = object
        shapely.ops = types.ModuleType("shapely.ops")
        shapely.ops.unary_union = object
        sys.modules["shapely"] = shapely
        sys.modules["shapely.geometry"] = shapely.geometry
        sys.modules["shapely.ops"] = shapely.ops


_install_optional_dependency_stubs()
import cc_lai_alteration as lai_module


def write_ascii_ply(path, vertices, faces):
    """Write a minimal triangular ASCII PLY mesh with zero-based face indices."""
    with open(path, "w", encoding="utf-8") as ply_file:
        ply_file.write("ply\nformat ascii 1.0\n")
        ply_file.write(f"element vertex {len(vertices)}\n")
        ply_file.write("property float x\nproperty float y\nproperty float z\n")
        ply_file.write(f"element face {len(faces)}\n")
        ply_file.write("property list uchar int vertex_indices\nend_header\n")
        for vertex in vertices:
            ply_file.write("{} {} {}\n".format(*vertex))
        for face in faces:
            ply_file.write("3 {} {} {}\n".format(*face))


def write_unit_leaf_mesh(path):
    """Write four separate unit-square leaves across two height bands."""
    vertices = []
    faces = []
    for z in (1, 2):
        for x in (0, 1):
            first = len(vertices)
            vertices.extend([
                (x, 0, z), (x + 1, 0, z), (x + 1, 1, z), (x, 1, z),
            ])
            faces.extend([
                (first, first + 1, first + 2),
                (first, first + 2, first + 3),
            ])
    write_ascii_ply(path, vertices, faces)


def write_unit_wood_mesh(path):
    """Write one small wood triangle inside the leaf mesh's ground bounds."""
    vertices = [(0.8, 0.4, 0), (1.2, 0.4, 0), (1, 0.6, 1)]
    write_ascii_ply(path, vertices, [(0, 1, 2)])


class SceneMeshTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = pathlib.Path(self.temp_dir.name)
        self.leaf_path = self.root / "tree_0_leaves.ply"
        self.wood_path = self.root / "tree_0_wood.ply"
        self.terrain_path = self.root / "terrain.ply"
        write_unit_leaf_mesh(self.leaf_path)
        write_unit_wood_mesh(self.wood_path)
        write_ascii_ply(self.terrain_path, [(0, 0, 0), (2, 0, 0), (0, 1, 0)], [(0, 1, 2)])
        self.scene_path = self.root / "scene.xml"
        scene = ET.Element("scene")
        for path in (self.leaf_path, self.wood_path, self.terrain_path):
            ET.SubElement(scene, "param", {"key": "filepath", "value": str(path)})
        ET.ElementTree(scene).write(self.scene_path)

    def test_scene_reader_reads_ply_and_converts_face_indices(self):
        vertices, faces = lai_module.SceneReader(str(self.scene_path)).read_ply(
            self.leaf_path
        )

        self.assertEqual(len(vertices), 16)
        np.testing.assert_array_equal(faces[0], [1, 2, 3])
        np.testing.assert_array_equal(faces[-1], [13, 15, 16])

    def test_calculate_lai_and_pai_from_scene(self):
        calculator = lai_module.CalculateVegAreaIndex(str(self.scene_path))

        leaf_area, ground_area, per_mesh_leaf_area = calculator.calculate_leaf_area_index(
            0, 2, 0, 1
        )
        plant_area, _, per_mesh_plant_area = calculator.calculate_plant_area_index(
            0, 2, 0, 1
        )

        expected_wood_area = np.sqrt(0.1664) / 2
        self.assertAlmostEqual(leaf_area / ground_area, 2.0)
        self.assertAlmostEqual(plant_area / ground_area, (4 + expected_wood_area) / 2)
        self.assertEqual(len(per_mesh_leaf_area), 1)
        self.assertEqual(len(per_mesh_plant_area), 2)

    def test_lai_reduction_writes_meshes_and_scene_and_reports_new_indices(self):
        reducer = lai_module.LeafAreaIndexNewScene(str(self.scene_path), new_lai=1.0)

        new_scene_path = pathlib.Path(reducer.create_new_scene())
        new_lai, new_pai = reducer.calculate_new_indices()
        new_scene_files = lai_module.SceneReader(str(new_scene_path)).read_scene()
        new_leaf_path = pathlib.Path(next(path for path in new_scene_files if "leaves" in path))
        reduced_vertices, reduced_faces = lai_module.SceneReader(str(new_scene_path)).read_ply(
            new_leaf_path
        )

        self.assertTrue(new_scene_path.exists())
        self.assertTrue(new_leaf_path.exists())
        self.assertEqual(len(reduced_faces), 4)
        self.assertEqual(len(reduced_vertices), 8)
        self.assertAlmostEqual(new_lai, 1.0)
        expected_wood_area = np.sqrt(0.1664) / 2
        self.assertAlmostEqual(new_pai, (2 + expected_wood_area) / 2)

    def test_lai_reduction_preserves_relative_area_in_each_height_band(self):
        reducer = lai_module.LeafAreaIndexNewScene.__new__(
            lai_module.LeafAreaIndexNewScene
        )
        reducer.retained_fraction = 0.5
        reducer.height_bins = 2

        vertices = []
        faces = []
        leaf_face_groups = []
        for height, radius in ((1.0, 1.0), (2.0, 1.4)):
            for index in range(10):
                x = index * 4.0
                first = len(vertices)
                vertices.extend([
                    (
                        x + radius * np.cos(angle * np.pi / 3),
                        radius * np.sin(angle * np.pi / 3),
                        height,
                    )
                    for angle in range(6)
                ])
                leaf_faces = [
                    (first + 1, first + index + 1, first + index + 2)
                    for index in range(1, 5)
                ]
                leaf_face_groups.append(np.arange(len(faces), len(faces) + 4))
                faces.extend(leaf_faces)
        vertices = np.asarray(vertices, dtype=float)
        faces = np.asarray(faces, dtype=int)

        kept_faces = reducer._select_faces(vertices, faces)
        all_areas = []
        kept_areas = []
        for height in (1.0, 2.0):
            band_ids = np.flatnonzero(
                np.mean(vertices[faces - 1, 2], axis=1) == height
            )
            kept_band_ids = np.intersect1d(band_ids, kept_faces)
            face_vertices = vertices[faces[band_ids] - 1]
            kept_vertices = vertices[faces[kept_band_ids] - 1]
            band_area = 0.5 * np.linalg.norm(
                np.cross(face_vertices[:, 1] - face_vertices[:, 0],
                         face_vertices[:, 2] - face_vertices[:, 0]),
                axis=1,
            ).sum()
            kept_area = 0.5 * np.linalg.norm(
                np.cross(kept_vertices[:, 1] - kept_vertices[:, 0],
                         kept_vertices[:, 2] - kept_vertices[:, 0]),
                axis=1,
            ).sum()
            all_areas.append(band_area)
            kept_areas.append(kept_area)

        for group in leaf_face_groups:
            retained_count = len(np.intersect1d(group, kept_faces))
            self.assertIn(retained_count, (0, 4))
        self.assertAlmostEqual(kept_areas[0] / all_areas[0], 0.5)
        self.assertAlmostEqual(kept_areas[1] / all_areas[1], 0.5)
        self.assertAlmostEqual(
            (kept_areas[1] / kept_areas[0]),
            (all_areas[1] / all_areas[0]),
        )


if __name__ == "__main__":
    unittest.main()
