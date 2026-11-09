"""Unit tests for the lightweight mesh and platform readers."""

import pathlib
import tempfile
import unittest

import numpy as np

from occpy_processing import Reader


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


if __name__ == "__main__":
    unittest.main()
