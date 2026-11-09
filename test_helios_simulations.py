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
            self.assertEqual(filepaths, [str(terrain), str(nearby_stem), str(nearby_leaf)])

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
        with self.assertRaises(ValueError):
            helios._scan_id_from_name("position-7")


if __name__ == "__main__":
    unittest.main()
