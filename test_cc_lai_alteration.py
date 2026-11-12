"""Unit tests for mesh reading and leaf-area reduction helpers."""

import importlib
import pathlib
import sys
import tempfile
import types
import unittest
import xml.etree.ElementTree as ET
from unittest import mock

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
    """
    Write a minimal triangular ASCII PLY mesh.

    Parameters:
    -----------
    path : str or pathlib.Path
           Destination for the PLY file.
    vertices : iterable of tuple
               XYZ coordinates to write.
    faces : iterable of tuple
            Triangles with zero-based vertex indices.

    Returns:
    --------
    None
        Writes the PLY file at path.
    """
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
    """
    Write four unit-square leaves in two height bands.

    Parameters:
    -----------
    path : str or pathlib.Path
           Destination for the PLY leaf mesh.

    Returns:
    --------
    None
        Writes the leaf mesh at path.
    """
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
    """
    Write a small wood triangle inside the leaf ground bounds.

    Parameters:
    -----------
    path : str or pathlib.Path
           Destination for the PLY wood mesh.

    Returns:
    --------
    None
        Writes the wood mesh at path.
    """
    vertices = [(0.8, 0.4, 0), (1.2, 0.4, 0), (1, 0.6, 1)]
    write_ascii_ply(path, vertices, [(0, 1, 2)])


class SceneMeshTests(unittest.TestCase):
    """Unit tests for scene mesh reading and LAI/PAI calculations."""

    def setUp(self):
        """Create temporary PLY meshes and an XML scene for each test."""
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
        """Verify PLY vertices are read and face indices use the reader convention."""
        vertices, faces = lai_module.SceneReader(str(self.scene_path)).read_ply(
            self.leaf_path
        )

        self.assertEqual(len(vertices), 16)
        np.testing.assert_array_equal(faces[0], [1, 2, 3])
        np.testing.assert_array_equal(faces[-1], [13, 15, 16])

    def test_calculate_lai_and_pai_from_scene(self):
        """Verify scene-wide LAI and PAI against known mesh areas."""
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

    def test_area_index_height_limits(self):
        """Verify zmin and zmax select faces by height and are validated."""
        calculator = lai_module.CalculateVegAreaIndex(str(self.scene_path))

        upper, ground_area, _ = calculator.calculate_leaf_area_index(0, 2, 0, 1, zmin=1.5)
        lower, _, _ = calculator.calculate_leaf_area_index(0, 2, 0, 1, zmax=1.5)
        none, _, _ = calculator.calculate_leaf_area_index(0, 2, 0, 1, zmin=10)
        plant, _, _ = calculator.calculate_plant_area_index(0, 2, 0, 1, zmin=1.5)
        wood, _, _ = calculator.calculate_area_index(0, 2, 0, 1, zmin=1.5, type="wood")

        self.assertAlmostEqual(upper / ground_area, 1.0)
        self.assertAlmostEqual(lower / ground_area, 1.0)
        self.assertEqual(none, 0.0)
        self.assertAlmostEqual(plant, upper + wood)
        with self.assertRaisesRegex(ValueError, "zmin"):
            calculator.calculate_area_index(0, 2, 0, 1, zmin=2, zmax=1)

    def test_area_index_boxes_and_triangle_clipping(self):
        """Check paired boxes, partial trees, and clipping in all three dimensions."""
        path = self.root / "crossing_leaves.obj"
        path.write_text(
            "v -1 -1 0\nv 3 -1 0\nv -1 3 0\n"
            "v 0 0 0\nv 0 0 2\nv 2 0 0\n"
            "f 1 2 3\nf 4 5 6\n",
            encoding="utf-8",
        )
        scene = self.root / "crossing.xml"
        scene.write_text(f'<scene><param key="filepath" value="{path}"/></scene>')
        calculator = lai_module.CalculateVegAreaIndex(str(scene))
        reader = lai_module.SceneReader.read_obj
        with mock.patch.object(lai_module.SceneReader, "read_obj", autospec=True,
                               side_effect=reader) as read:
            area, ground, per_mesh = calculator.calculate_leaf_area_index(
                [0, 1], [1, 2], 0, 1, zmin=0, zmax=1, method="clip"
            )
        self.assertEqual(read.call_count, 1)
        np.testing.assert_allclose(area, [2.0, 1.0], atol=1e-14)
        np.testing.assert_allclose(ground, [1, 1])
        self.assertEqual(per_mesh.shape, (2, 1))
        np.testing.assert_allclose(per_mesh[:, 0], area)
        centroid, _, _ = calculator.calculate_leaf_area_index(
            [0, 1], [1, 2], 0, 1, zmin=0, zmax=1
        )
        np.testing.assert_allclose(centroid, [10, 0])
        scalar, scalar_ground, scalar_mesh = calculator.calculate_leaf_area_index(
            0, 1, 0, 1, zmin=0, zmax=1, method="clip"
        )
        self.assertAlmostEqual(scalar, area[0])
        self.assertEqual(scalar_ground, 1)
        self.assertEqual(scalar_mesh.shape, (1,))

    def test_area_index_boundary_planes_and_empty_boxes(self):
        """Faces on shared boundaries belong to only the upper-index box."""
        calculator = lai_module.CalculateVegAreaIndex(str(self.scene_path))
        for method in ("centroid", "clip"):
            leaf, ground, meshes = calculator.calculate_leaf_area_index(
                0, 2, 0, 1, zmin=[0, 1, 2, 3], zmax=[1, 2, 3, 4], method=method
            )
            np.testing.assert_allclose(leaf, [0, 2, 2, 0])
            np.testing.assert_allclose(ground, [2, 2, 2, 2])
            self.assertEqual(meshes.shape, (4, 1))
        area, ground, meshes = calculator.calculate_plant_area_index(
            [10, 20], [11, 21], 0, 1, method="clip"
        )
        np.testing.assert_array_equal(area, [0, 0])
        np.testing.assert_array_equal(ground, [1, 1])
        self.assertEqual(meshes.shape, (2, 0))

    def test_area_index_plant_arrays_and_input_validation(self):
        """Verify array broadcasting, plant aggregation, and invalid inputs."""
        calculator = lai_module.CalculateVegAreaIndex(str(self.scene_path))
        leaf, _, leaf_mesh = calculator.calculate_leaf_area_index(
            [0, 1], [1, 2], [0], 1, method="clip"
        )
        wood, _, wood_mesh = calculator.calculate_area_index(
            [0, 1], [1, 2], 0, 1, type="wood", method="clip"
        )
        plant, ground, meshes = calculator.calculate_plant_area_index(
            [0, 1], [1, 2], 0, 1, method="clip"
        )
        np.testing.assert_allclose(plant, leaf + wood)
        np.testing.assert_allclose(meshes, np.concatenate((leaf_mesh, wood_mesh), axis=1))
        np.testing.assert_array_equal(ground, [1, 1])
        for kwargs in (
            {"xmin": [0, 1, 2], "xmax": [1, 2]},
            {"xmin": [[0]], "xmax": 1},
            {"xmin": [], "xmax": []},
            {"xmin": 0, "xmax": 1, "zmin": np.nan},
            {"xmin": 0, "xmax": 1, "method": "unknown"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                calculator.calculate_area_index(ymin=0, ymax=1, **kwargs)

    def test_mesh_summary_skips_unrelated_meshes_and_reuses_total(self):
        """Create a small per-mesh summary and reopen only partially overlapping meshes."""
        other = self.root / "other_leaves.ply"
        write_ascii_ply(other, [(10, 0, 0), (11, 0, 0), (10, 1, 0)], [(0, 1, 2)])
        scene = ET.parse(self.scene_path)
        ET.SubElement(scene.getroot(), "param", {"key": "filepath", "value": str(other)})
        scene.write(self.scene_path)
        calculator = lai_module.CalculateVegAreaIndex(str(self.scene_path))
        read_ply = lai_module.SceneReader.read_ply
        with mock.patch.object(lai_module.SceneReader, "read_ply", autospec=True,
                               side_effect=read_ply) as read:
            first = calculator.calculate_leaf_area_index(0, 1, 0, 1, method="clip")
            self.assertEqual(read.call_count, 2)
            self.assertTrue(calculator.mesh_summary_path.exists())
            summary = np.load(calculator.mesh_summary_path, allow_pickle=False)
            self.assertEqual(len(summary), 2)
            self.assertEqual(set(summary["path"]), {str(self.leaf_path.resolve()), str(other.resolve())})
            record = summary[summary["path"] == str(self.leaf_path.resolve())][0]
            np.testing.assert_array_equal(record["minimum"], [0, 0, 1])
            np.testing.assert_array_equal(record["maximum"], [2, 1, 2])
            self.assertAlmostEqual(record["area"], 4)

            read.reset_mock()
            fresh = lai_module.CalculateVegAreaIndex(str(self.scene_path))
            second = fresh.calculate_leaf_area_index(0, 1, 0, 1, method="clip")
            self.assertEqual(read.call_count, 1)
            self.assertEqual(pathlib.Path(read.call_args.args[1]), self.leaf_path.resolve())
            np.testing.assert_allclose(first[0], second[0])
            np.testing.assert_allclose(first[2], second[2])

            read.reset_mock()
            whole = calculator.calculate_leaf_area_index(-1, 3, -1, 2)
            self.assertEqual(read.call_count, 0)
            self.assertEqual(whole[0], 4)
            read.reset_mock()
            calculator.calculate_plant_area_index(-1, 3, -1, 2)
            self.assertEqual(read.call_count, 1)
            self.assertEqual(pathlib.Path(read.call_args.args[1]), self.wood_path.resolve())
            self.assertEqual(len(np.load(calculator.mesh_summary_path, allow_pickle=False)), 3)

    def test_mesh_summary_refreshes_changed_mesh(self):
        """Rebuild stale entries before deciding which meshes intersect a box."""
        calculator = lai_module.CalculateVegAreaIndex(str(self.scene_path))
        initial = calculator.calculate_leaf_area_index(0, 2, 0, 1, method="clip")
        self.assertAlmostEqual(initial[0], 4)
        write_ascii_ply(self.leaf_path, [(10, 0, 0), (11, 0, 0), (10, 1, 0)], [(0, 1, 2)])
        reader = lai_module.SceneReader.read_ply
        with mock.patch.object(lai_module.SceneReader, "read_ply", autospec=True,
                               side_effect=reader) as read:
            result = calculator.calculate_leaf_area_index(0, 2, 0, 1, method="clip")
            self.assertEqual(read.call_count, 1)
            self.assertEqual(result[0], 0)
            result = calculator.calculate_leaf_area_index(9, 12, -1, 2)
            self.assertEqual(read.call_count, 1)
            self.assertAlmostEqual(result[0], 0.5)
        summary = np.load(calculator.mesh_summary_path, allow_pickle=False)
        self.assertAlmostEqual(summary["area"][0], 0.5)
        np.testing.assert_array_equal(summary["minimum"][0], [10, 0, 0])

    def test_mesh_summary_invalid_data_is_reported(self):
        """Do not silently treat corrupt summaries as valid cache misses."""
        calculator = lai_module.CalculateVegAreaIndex(str(self.scene_path))
        calculator.mesh_summary_path.write_bytes(b"not a numpy file")
        with self.assertRaisesRegex(ValueError, "Invalid mesh summary"):
            calculator.calculate_leaf_area_index(0, 2, 0, 1)

    def test_mesh_summary_not_replaced_if_read_fails(self):
        """A missing mesh must not replace an existing valid summary."""
        calculator = lai_module.CalculateVegAreaIndex(str(self.scene_path))
        calculator.calculate_leaf_area_index(0, 2, 0, 1)
        original = calculator.mesh_summary_path.read_bytes()
        self.wood_path.unlink()
        with self.assertRaises(FileNotFoundError):
            calculator.calculate_plant_area_index(0, 2, 0, 1)
        self.assertEqual(calculator.mesh_summary_path.read_bytes(), original)

    def test_clipping_conserves_area_over_adjacent_3d_boxes(self):
        """Check tilted random triangles conserve area across a box partition."""
        rng = np.random.default_rng(42)
        triangles = rng.uniform(0.01, 2.99, size=(100, 3, 3))
        edges = np.arange(4, dtype=float)
        for triangle in triangles:
            expected = 0.5 * np.linalg.norm(
                np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            )
            total = 0.0
            for x in range(3):
                for y in range(3):
                    for z in range(3):
                        total += lai_module.CalculateVegAreaIndex._clipped_triangle_area(
                            triangle, np.array([edges[x], edges[y], edges[z]]),
                            np.array([edges[x + 1], edges[y + 1], edges[z + 1]]),
                        )
            self.assertAlmostEqual(total, expected, places=12)

    def test_calculate_lai_and_pai_profiles_by_height(self):
        """Verify height-band profiles give the expected index contributions."""
        calculator = lai_module.CalculateVegAreaIndex(str(self.scene_path))

        leaf_profile = calculator.calculate_leaf_area_index_profile(
            0, 2, 0, 1, height_bins=2
        )
        plant_profile = calculator.calculate_plant_area_index_profile(
            0, 2, 0, 1, height_bins=2
        )
        total_lai, ground_area, _ = calculator.calculate_leaf_area_index(0, 2, 0, 1)
        total_pai, _, _ = calculator.calculate_plant_area_index(0, 2, 0, 1)

        np.testing.assert_allclose(leaf_profile[:, 0], [1.25, 1.75])
        np.testing.assert_allclose(leaf_profile[:, 1], [1.0, 1.0])
        self.assertAlmostEqual(np.sum(leaf_profile[:, 1]), total_lai / ground_area)
        self.assertAlmostEqual(np.sum(plant_profile[:, 1]), total_pai / ground_area)

    def test_area_index_profile_applies_min_height_and_validates_inputs(self):
        """Verify profile filtering and errors for unsupported profile requests."""
        calculator = lai_module.CalculateVegAreaIndex(str(self.scene_path))

        profile = calculator.calculate_leaf_area_index_profile(
            0, 2, 0, 1, min_height=1.5, height_bins=2
        )

        self.assertAlmostEqual(np.sum(profile[:, 1]), 1.0)
        with self.assertRaisesRegex(ValueError, "height_bins"):
            calculator.calculate_leaf_area_index_profile(0, 2, 0, 1, height_bins=0)
        with self.assertRaisesRegex(ValueError, "Invalid type"):
            calculator.calculate_area_index_profile(0, 2, 0, 1, type="wood")

    def test_area_index_profile_uses_and_validates_shared_height_range(self):
        """Verify explicit height ranges provide aligned profile bin centers."""
        calculator = lai_module.CalculateVegAreaIndex(str(self.scene_path))

        profile = calculator.calculate_leaf_area_index_profile(
            0, 2, 0, 1, height_bins=2, height_range=(1, 2)
        )

        np.testing.assert_allclose(profile[:, 0], [1.25, 1.75])
        np.testing.assert_allclose(profile[:, 1], [1.0, 1.0])
        with self.assertRaisesRegex(ValueError, "contain all included"):
            calculator.calculate_leaf_area_index_profile(
                0, 2, 0, 1, height_bins=2, height_range=(1.1, 2)
            )

    def test_lai_reduction_writes_meshes_and_scene_and_reports_new_indices(self):
        """Verify LAI reduction writes a scene and reports its calculated indices."""
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
        """Verify thinning approximately preserves leaf area across height bands."""
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
