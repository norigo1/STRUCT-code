"""Unit tests for scene and mesh input boundaries."""

import pathlib
import tempfile
import unittest

import numpy as np

from scene_io import SceneReader, read_obj_facets, read_obj_mesh


class SceneIOTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = pathlib.Path(self.temp_dir.name)
        self.reader = SceneReader(str(self.root / "scene.xml"))

    def test_relative_paths_are_resolved_against_scene_directory(self):
        mesh = self.root / "tree.obj"
        mesh.write_text("v 0 0 0\n", encoding="utf-8")
        self.assertEqual(self.reader.resolve_filepath("tree.obj"), str(mesh))
        missing = "not-a-real-mesh.obj"
        self.assertEqual(self.reader.resolve_filepath(missing), missing)
        self.assertEqual(self.reader.resolve_filepath(mesh), str(mesh))

    def test_obj_faces_reuse_and_reorder_one_based_vertex_indices(self):
        mesh = self.root / "mesh.obj"
        mesh.write_text(
            "# comment\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 3 2 1\nf 1 3 2\n",
            encoding="utf-8",
        )
        vertices, faces = read_obj_mesh(mesh)
        self.assertEqual(faces, [[3, 2, 1], [1, 3, 2]])
        np.testing.assert_array_equal(
            read_obj_facets(mesh),
            np.asarray(vertices)[np.asarray(faces) - 1],
        )

    def test_ascii_ply_accepts_empty_mesh_and_extra_vertex_properties(self):
        mesh = self.root / "mesh.ply"
        mesh.write_text(
            "ply\nformat ascii 1.0\nelement vertex 1\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property float intensity\nelement face 0\n"
            "property list uchar int vertex_indices\nend_header\n"
            "1 2 3 99\n",
            encoding="utf-8",
        )
        vertices, faces = self.reader.read_ply(mesh)
        self.assertEqual(vertices, [[1.0, 2.0, 3.0]])
        self.assertEqual(faces, [])

    def test_ascii_ply_reports_invalid_headers_and_records(self):
        cases = [
            ("", "Not a PLY"),
            ("ply\nformat binary_little_endian 1.0\nend_header\n", "Only ASCII"),
            ("ply\nformat ascii 1.0\n", "no end_header"),
            (
                "ply\nformat ascii 1.0\nelement vertex 2\nend_header\n0 0 0\n",
                "incomplete",
            ),
            ("ply\nformat ascii 1.0\nelement vertex 1\nend_header\n0 0\n", "fewer"),
            (
                "ply\nformat ascii 1.0\nelement face 1\nend_header\n4 0 1 2 3\n",
                "triangular",
            ),
        ]
        path = self.root / "bad.ply"
        for contents, message in cases:
            with self.subTest(message=message):
                path.write_text(contents, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    self.reader.read_ply(path)

    def test_scene_reader_preserves_reference_order_and_skips_other_keys(self):
        scene = self.root / "scene.xml"
        scene.write_text(
            '<scene><param key="filepath" value="first.obj"/>'
            '<param key="name" value="ignored"/>'
            '<param key="filepath" value="second.ply"/></scene>',
            encoding="utf-8",
        )
        self.assertEqual(self.reader.read_scene(), ["first.obj", "second.ply"])

    def test_scene_reader_accepts_path_objects(self):
        scene = self.root / "scene.xml"
        scene.write_text("<scene/>", encoding="utf-8")
        self.assertEqual(SceneReader(scene).read_scene(), [])

    def test_read_tree_accepts_one_matching_mesh(self):
        mesh = self.root / "tree.obj"
        mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
        vertices, faces = self.reader.read_tree([str(self.root)], "*.obj")
        self.assertEqual(vertices.shape, (3, 3))
        np.testing.assert_array_equal(faces, [[1, 2, 3]])


if __name__ == "__main__":
    unittest.main()
