"""Unit tests for HELIOS scene and survey generation helpers."""

import pathlib
import tempfile
import unittest
import xml.etree.ElementTree as ET

import helios_simulations as helios


def _write_obj(path, x, y, z=0):
    path.write_text(
        f"v {x} {y} {z}\nv {x + 1} {y} {z}\nv {x} {y + 1} {z}\n",
        encoding="utf-8",
    )


class HeliosSimulationTests(unittest.TestCase):
    """Unit tests for HELIOS scene generation and survey rebuilding."""

    def test_generate_scene_includes_nearby_stem_leaf_and_terrain(self):
        """Include nearby stem/leaf meshes and terrain in the generated XML."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            terrain = root / "terrain.obj"
            nearby_stem = root / "tree_1_mesh.obj"
            nearby_leaf = root / "tree_1_leaves.obj"
            distant_stem = root / "tree_2_mesh.obj"
            _write_obj(terrain, 0, 0)
            _write_obj(nearby_stem, 1, 0)
            _write_obj(nearby_leaf, 1, 0, 2)
            _write_obj(distant_stem, 20, 20)
            output = root / "generated" / "scene.xml"

            parts = helios.generate_scene_for_radius(
                [nearby_stem, distant_stem],
                [nearby_leaf],
                terrain,
                scan_position=(0, 0),
                radius=3,
                outpath=output,
                scene_name="test_scene",
            )

            self.assertEqual(
                parts,
                [(0, str(terrain)), (1, str(nearby_stem)), (1, str(nearby_leaf))],
            )
            root_xml = ET.parse(output).getroot()
            scene = root_xml.find("scene")
            self.assertEqual(scene.get("name"), "test_scene")
            filepaths = [
                param.get("value")
                for param in scene.findall(".//param[@key='filepath']")
            ]
            self.assertEqual(
                filepaths, [str(terrain), str(nearby_stem), str(nearby_leaf)]
            )

    def test_rebuild_survey_legs_from_grid_replaces_old_legs(self):
        """Replace existing survey legs with grid locations and settings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            survey_path = root / "survey.xml"
            output_path = root / "output" / "survey.xml"
            survey_path.write_text(
                "<document><survey><leg id='old'/></survey></document>",
                encoding="utf-8",
            )

            count = helios.rebuild_survey_legs_from_grid(
                str(survey_path),
                str(output_path),
                xmin=0,
                xmax=1,
                ymin=2,
                ymax=2,
                x_step=1,
                y_step=1,
                scanner_template="test-scanner",
            )

            survey = ET.parse(output_path).getroot().find("survey")
            legs = survey.findall("leg")
            self.assertEqual(count, 2)
            self.assertEqual(len(legs), 2)
            self.assertEqual(
                [leg.find("platformSettings").get("x") for leg in legs],
                ["0.000000", "1.000000"],
            )
            self.assertTrue(
                all(
                    leg.find("scannerSettings").get("template") == "test-scanner"
                    for leg in legs
                )
            )

    def test_scan_id_parser_rejects_non_scanpos_names(self):
        """Accept scan-position names and reject malformed identifiers."""
        self.assertEqual(helios._scan_id_from_name("ScanPos007"), 7)
        self.assertEqual(helios._scan_id_from_name(" ScanPos0 "), 0)
        with self.assertRaises(ValueError):
            helios._scan_id_from_name("position-7")

    def test_radius_includes_boundary_and_rejects_negative_radius(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            stem = root / "tree_mesh.obj"
            stem.write_text("v 1 0 0\n", encoding="utf-8")
            output = root / "scene.xml"
            parts = helios.generate_scene_for_radius(
                [stem], [], root / "terrain.obj", (0, 0), 1, output, "boundary"
            )
            self.assertEqual(parts, [(0, str(root / "terrain.obj")), (1, str(stem))])
            with self.assertRaisesRegex(ValueError, "radius"):
                helios.generate_scene_for_radius(
                    [stem], [], root / "terrain.obj", (0, 0), -0.01, output, "bad"
                )

    def test_obj_centroid_rejects_mesh_without_vertices(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "empty.obj"
            path.write_text("# no vertices\nf 1 2 3\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No vertices"):
                helios._obj_centroid(path)

    def test_grid_scene_generation_single_point_and_invalid_bounds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            for name in ("wood", "leaves", "terrain"):
                (root / name).mkdir()
            _write_obj(root / "terrain" / "terrain.obj", 0, 0)
            options = {
                "xmin": 2,
                "xmax": 2,
                "ymin": -1,
                "ymax": -1,
                "x_step": 1,
                "y_step": 1,
                "stem_dir": str(root / "wood"),
                "leaf_dir": str(root / "leaves"),
                "terrain_dir": str(root / "terrain"),
                "radius": 0,
                "outdir": str(root / "output"),
            }
            scenes = helios.generate_radius_scenes_from_grid(**options)
            self.assertEqual(len(scenes), 1)
            self.assertEqual(pathlib.Path(scenes[0]).name, "scene_2.000_-1.000.xml")
            for changes in (
                {"x_step": 0},
                {"y_step": -1},
                {"xmin": 3},
                {"ymin": 0},
            ):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    helios.generate_radius_scenes_from_grid(**(options | changes))

    def test_scan_position_survey_filters_ids_and_preserves_coordinates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            template = root / "template.xml"
            template.write_text(
                "<document><survey><leg id='old'/></survey></document>",
                encoding="utf-8",
            )
            positions = root / "positions.csv"
            positions.write_text(
                "#Name,X,Y,Z,Roll,Pitch,Yaw\n"
                "ScanPos001,1,2,3,0,0,0\n"
                "ScanPos002,-4,5,6,0,0,0\n",
                encoding="utf-8",
            )
            output = root / "survey.xml"
            count = helios.rebuild_survey_legs_from_scanpos(
                str(template), str(positions), str(output), [2], "scanner"
            )
            self.assertEqual(count, 1)
            survey = ET.parse(output).getroot().find("survey")
            legs = survey.findall("leg")
            self.assertEqual(len(legs), 1)
            self.assertEqual(
                legs[0].find("platformSettings").attrib,
                {"x": "-4.000000", "y": "5.000000", "z": "6.000000"},
            )
            self.assertEqual(legs[0].find("scannerSettings").get("template"), "scanner")
            with self.assertRaisesRegex(ValueError, "No scan positions"):
                helios.rebuild_survey_legs_from_scanpos(
                    str(template), str(positions), str(output), [99]
                )

    def test_scan_position_scenes_require_terrain_and_all_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            for name in ("wood", "leaves", "terrain"):
                (root / name).mkdir()
            positions = root / "positions.csv"
            positions.write_text("#Name,X\nScanPos001,0\n", encoding="utf-8")
            args = (
                str(positions),
                str(root / "wood"),
                str(root / "leaves"),
                str(root / "terrain"),
                1,
                str(root / "output"),
            )
            with self.assertRaisesRegex(FileNotFoundError, "No terrain OBJ"):
                helios.generate_radius_scenes_from_csv(*args)
            _write_obj(root / "terrain" / "terrain.obj", 0, 0)
            with self.assertRaisesRegex(ValueError, "Missing required columns"):
                helios.generate_radius_scenes_from_csv(*args)

    def test_survey_rejects_missing_columns_and_missing_survey(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            positions = root / "positions.csv"
            positions.write_text("#Name,X\nScanPos001,0\n", encoding="utf-8")
            template = root / "survey.xml"
            template.write_text("<document><survey/></document>", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing required columns"):
                helios.rebuild_survey_legs_from_scanpos(
                    str(template), str(positions), str(root / "out.xml")
                )
            template.write_text("<document/>", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No <survey>"):
                helios.rebuild_survey_leg_from_point(
                    str(template), str(root / "out.xml"), 0, 0
                )

    def test_single_point_survey_replaces_legs_and_updates_references(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            template = root / "template.xml"
            output = root / "output.xml"
            template.write_text(
                "<document><survey><leg/><leg/></survey></document>", encoding="utf-8"
            )
            helios.rebuild_survey_leg_from_point(
                str(template),
                str(output),
                -2,
                3,
                name="new",
                scene="scene.xml",
                scene_name="plot",
                platform_file="platform.xml",
                platform="p",
                scanner_file="scanner.xml",
                scanner="s",
            )
            survey = ET.parse(output).getroot().find("survey")
            self.assertEqual(survey.get("scene"), "scene.xml#plot")
            self.assertEqual(survey.get("platform"), "platform.xml#p")
            self.assertEqual(survey.get("scanner"), "scanner.xml#s")
            self.assertEqual(len(survey.findall("leg")), 1)
            self.assertEqual(survey.find("leg/platformSettings").get("x"), "-2.000000")

    def test_radius_scenes_from_csv_handles_name_column(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            for name in ("wood", "leaves", "terrain"):
                (root / name).mkdir()
            _write_obj(root / "terrain" / "terrain.obj", 0, 0)
            _write_obj(root / "wood" / "near.obj", 1, 2)
            positions = root / "positions.csv"
            positions.write_text(
                "Y,#Name,Z,X,Roll,Pitch,Yaw\n"
                "2,ScanPos001,0,1,0,0,0\n"
                "-7,ScanPos002,0,11,0,0,0\n",
                encoding="utf-8",
            )
            scenes = helios.generate_radius_scenes_from_csv(
                str(positions),
                str(root / "wood"),
                str(root / "leaves"),
                str(root / "terrain"),
                0.5,
                str(root / "output"),
            )
            self.assertEqual(
                [pathlib.Path(scene).name for scene in scenes],
                ["scene_ScanPos001.xml", "scene_ScanPos002.xml"],
            )
            filepaths = [
                [
                    param.get("value")
                    for param in ET.parse(scene).getroot().findall(
                        ".//param[@key='filepath']"
                    )
                ]
                for scene in scenes
            ]
            self.assertIn(str(root / "wood" / "near.obj"), filepaths[0])
            self.assertNotIn(str(root / "wood" / "near.obj"), filepaths[1])


if __name__ == "__main__":
    unittest.main()
