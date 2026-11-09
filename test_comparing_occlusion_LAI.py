"""Unit tests for occlusion-profile and voxel LAI helpers."""

import contextlib
import io
import pathlib
import tempfile
import unittest

import numpy as np

import comparing_occlusion_LAI as occlusion_lai


class OcclusionAnalysisTests(unittest.TestCase):
    """Unit tests for scene-path reading, vertical profiles, and voxel LAI."""

    def test_scene_parameters_reads_mesh_paths_and_leaf_facets(self):
        """Read leaf and wood OBJ paths and return leaf face coordinates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            leaf_path = root / "leaves" / "leaf.obj"
            wood_path = root / "wood" / "wood.obj"
            leaf_path.parent.mkdir()
            wood_path.parent.mkdir()
            leaf_path.write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
                encoding="utf-8",
            )
            wood_path.write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
                encoding="utf-8",
            )
            scene_path = root / "scene.xml"
            scene_path.write_text(
                f'<shape filepath="{leaf_path}" />\n'
                f'<shape filepath="{wood_path}" />\n',
                encoding="utf-8",
            )

            scene = occlusion_lai.SceneParameters(scene_path)

            self.assertEqual(scene.get_leaf_files(), [str(leaf_path)])
            self.assertEqual(scene.get_wood_files(), [str(wood_path)])
            np.testing.assert_allclose(
                scene.get_all_leaves(),
                [[[0, 0, 0], [1, 0, 0], [0, 1, 0]]],
            )

    def test_occluded_vertical_profile_returns_percent_and_absolute_counts(self):
        """Check profile heights and both percentage and count outputs."""
        analysis = occlusion_lai.OccpyAnalysis.__new__(
            occlusion_lai.OccpyAnalysis
        )
        analysis.oconfig = {
            "plot_dim": [0, 0, 0, 2, 2, 2],
            "vox_dim": 1,
        }
        analysis.Classification_norm = np.array(
            [
                [[3, 1], [2, 3]],
                [[1, 3], [4, 1]],
            ]
        )

        percent = analysis.get_occluded_vertical_profile()
        absolute = analysis.get_occluded_vertical_profile("absolute")

        np.testing.assert_allclose(percent, [[0.5, 25.0], [1.5, 50.0]])
        np.testing.assert_allclose(absolute, [[0.5, 1], [1.5, 2]])

    def test_polygon_area_and_voxel_lai_for_contained_triangle(self):
        """Check polygon area and LAI for a triangle fully inside one voxel."""
        triangle = np.array(
            [[0.1, 0.1, 0.5], [0.8, 0.1, 0.5], [0.1, 0.8, 0.5]]
        )
        calculator = occlusion_lai.LAICalculations()

        self.assertAlmostEqual(calculator.polygon_area_3d(triangle), 0.245)
        with contextlib.redirect_stdout(io.StringIO()):
            lai = calculator.voxel_lai_from_facets(
                0, 1, 0, 1, 0, 1, triangle[np.newaxis, :, :]
            )
        self.assertAlmostEqual(lai, 0.245)

    def test_read_obj_expands_face_indices_to_vertex_coordinates(self):
        """Verify OBJ faces are expanded into their 3D vertex coordinates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            obj_path = pathlib.Path(temp_dir) / "triangle.obj"
            obj_path.write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
                encoding="utf-8",
            )

            facets = occlusion_lai.read_obj(obj_path)

        np.testing.assert_allclose(
            facets,
            [[[0, 0, 0], [1, 0, 0], [0, 1, 0]]],
        )


if __name__ == "__main__":
    unittest.main()
