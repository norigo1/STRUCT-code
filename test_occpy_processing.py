"""Unit tests for the lightweight mesh and platform readers."""

import csv
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np

import occpy_processing
from occpy_processing import ModelBoundaries, OccpyConfig, Reader, RunOccpy


class OccpyReaderTests(unittest.TestCase):
    """Unit tests for OBJ, PLY, and platform XML readers."""

    def test_obj_reader_returns_xyz_vertices(self):
        """Read XYZ vertex coordinates from an OBJ mesh."""
        with tempfile.TemporaryDirectory() as temp_dir:
            obj_path = pathlib.Path(temp_dir) / "mesh.obj"
            obj_path.write_text(
                "v 1 2 3\nv 4 5 6\nf 1 2 1\n",
                encoding="utf-8",
            )

            vertices = Reader.obj_reader(obj_path)

        np.testing.assert_allclose(vertices, [[1, 2, 3], [4, 5, 6]])

    def test_ply_reader_returns_xyz_vertices(self):
        """Read XYZ vertex coordinates from an ASCII PLY mesh."""
        with tempfile.TemporaryDirectory() as temp_dir:
            ply_path = pathlib.Path(temp_dir) / "mesh.ply"
            ply_path.write_text(
                "ply\n"
                "format ascii 1.0\n"
                "element vertex 2\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "element face 0\n"
                "property list uchar int vertex_indices\n"
                "end_header\n"
                "1 2 3\n"
                "4 5 6\n",
                encoding="utf-8",
            )

            vertices = Reader.ply_reader(ply_path)

        np.testing.assert_allclose(vertices, [[1, 2, 3], [4, 5, 6]])

    def test_platform_xml_returns_scan_positions_without_writing(self):
        """Return scan positions from platform XML without creating an output file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            xml_path = pathlib.Path(temp_dir) / "platform.xml"
            xml_path.write_text(
                "<document>\n"
                '  <platformSettings x="1.5" y="-2" z="3.25" />\n'
                '  <platformSettings x="4" y="5" z="6" />\n'
                "</document>\n",
                encoding="utf-8",
            )

            positions = Reader.platform_xml(xml_path, write_to_scanpos=False)

        np.testing.assert_allclose(positions, [[1.5, -2, 3.25], [4, 5, 6]])

    def test_platform_xml_handles_no_platform_settings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "platform.xml"
            path.write_text("<document/>\n", encoding="utf-8")
            positions = Reader.platform_xml(path, write_to_scanpos=False)
            self.assertEqual(positions.size, 0)

    def test_model_boundaries_find_requested_terrain_and_mesh_types(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            for name in ("terrain", "leaves", "wood", "snags"):
                (root / name).mkdir()
            (root / "terrain" / "terrain.obj").touch()
            (root / "terrain" / "chosen.obj").touch()
            (root / "leaves" / "one.obj").touch()
            (root / "wood" / "one.ply").touch()
            boundaries = ModelBoundaries(str(root), terrain_fname="chosen.obj")
            self.assertEqual(pathlib.Path(boundaries.terrain).name, "chosen.obj")
            self.assertEqual(
                [pathlib.Path(p).name for p in boundaries.leaves], ["one.obj"]
            )
            self.assertEqual(boundaries.wood, [])
            self.assertEqual(boundaries.snags, [])

    def test_run_occpy_passes_paths_to_normalization(self):
        config = {
            "ScanPos": "scan.csv",
            "root_folder": "/project",
            "out_dir": "output",
            "plot_dim": [0, 0, 0, 1, 1, 1],
            "vox_dim": 1,
            "tif_in": {"DTM": "dtm.tif", "DSM": "dsm.tif"},
            "lower_threshold": 0,
            "output_voxels": True,
        }
        results = tuple(np.array([i]) for i in range(5))
        with (
            mock.patch.object(occpy_processing.occpy, "OccPy") as occpy,
            mock.patch.object(
                occpy_processing.occpy.util,
                "normalize_occlusion_output",
                return_value=results,
            ) as normalize,
        ):
            runner = RunOccpy(config)
        occpy.assert_called_once_with(config=config)
        occpy.return_value.define_sensor_pos.assert_called_once_with(
            path2file="scan.csv",
            delimiter=",",
            hdr_scanpos_id="#Name",
            hdr_x="X",
            hdr_y="Y",
            hdr_z="Z",
            sens_pos_id_offset=0,
        )
        occpy.return_value.do_raytracing.assert_called_once_with()
        normalize.assert_called_once_with(
            input_folder="/project/output",
            PlotDim=[0, 0, 0, 1, 1, 1],
            vox_dim=1,
            dtm_file="/project/dtm.tif",
            dsm_file="/project/dsm.tif",
            lower_threshold=0,
            output_voxels=True,
        )
        for name, expected in zip(
            ("Nhit_norm", "Nmiss_norm", "Nocc_norm", "Classification_norm", "chm"),
            results,
        ):
            np.testing.assert_array_equal(getattr(runner, name), expected)

    def test_classified_voxels_return_bounds_for_each_category(self):
        runner = RunOccpy.__new__(RunOccpy)
        runner.config = {"plot_dim": [10, -2, 4, 11, -0.5, 6], "vox_dim": 0.5}
        runner.Classification_norm = np.zeros((2, 3, 4), dtype=int)
        runner.Classification_norm[0, 1, 3] = 1
        runner.Classification_norm[1, 2, 0] = 1
        runner.Classification_norm[1, 0, 2] = 2
        runner.Classification_norm[0, 2, 1] = 3
        runner.Classification_norm[1, 1, 3] = 4
        cases = (
            (runner.get_hit_voxels, [[10, -1.5, 5.5], [10.5, -1, 4]]),
            (runner.get_empty_voxels, [[10.5, -2, 5]]),
            (runner.get_occluded_voxels, [[10, -1, 4.5]]),
            (runner.get_unobserved_voxels, [[10.5, -1.5, 5.5]]),
        )
        for get_voxels, expected_min in cases:
            with self.subTest(category=get_voxels.__name__):
                mins, maxs = get_voxels()
                np.testing.assert_array_equal(mins, expected_min)
                np.testing.assert_array_equal(maxs, np.asarray(expected_min) + 0.5)

    def test_classified_voxels_return_empty_coordinate_arrays(self):
        runner = RunOccpy.__new__(RunOccpy)
        runner.config = {"plot_dim": [0, 0, 0, 1, 1, 1], "vox_dim": 1}
        runner.Classification_norm = np.zeros((1, 1, 1), dtype=int)

        mins, maxs = runner.get_hit_voxels()

        self.assertEqual(mins.shape, (0, 3))
        self.assertEqual(maxs.shape, (0, 3))

    def test_platform_xml_writes_scan_positions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            path = root / "platform.xml"
            path.write_text(
                "<document>\n"
                '<platformSettings x="1" y="2" z="3" />\n'
                '<platformSettings x="4.5" y="-2" z="6" />\n'
                "</document>\n",
                encoding="utf-8",
            )
            Reader.platform_xml(str(path), write_to_scanpos=True)
            with (root / "ScanPositions.txt").open(
                encoding="utf-8", newline=""
            ) as file:
                reader = csv.DictReader(file)
                self.assertEqual(
                    reader.fieldnames, ["#Name", "X", "Y", "Z", "ID"]
                )
                rows = list(reader)
            self.assertEqual([row["#Name"] for row in rows], ["ScanPos00", "ScanPos01"])
            self.assertEqual([int(row["ID"]) for row in rows], [0, 1])
            np.testing.assert_allclose(
                [[float(row[axis]) for axis in ("X", "Y", "Z")] for row in rows],
                [[1, 2, 3], [4.5, -2, 6]],
            )

    def test_config_uses_supplied_laz_folder(self):
        config = OccpyConfig(
            None,
            root_folder="/project",
            laz_folder="laz",
            out_folder="output",
            DTM_in="dtm.tif",
            DSM_in="dsm.tif",
            voxel_dim=1,
            plot_dim=[0, 0, 0, 1, 1, 1],
            scanpos="scan.csv",
        )
        self.assertEqual(config["laz_in"], "laz")


if __name__ == "__main__":
    unittest.main()
