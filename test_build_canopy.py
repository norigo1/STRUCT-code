"""Unit tests for canopy scene and Eradiate output helpers."""

import os
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr

with patch.dict(os.environ, {"ERADIATE_PROJECT_DIR": tempfile.gettempdir()}):
    import build_canopy as bc


def _band_dataset(wavelengths, offset=0):
    wavelengths = np.asarray(wavelengths, dtype=float)
    nw = len(wavelengths)
    return xr.Dataset(
        data_vars={
            "irradiance": (("sza", "saa", "w"), [[wavelengths + offset]]),
            "radiance": (
                ("w", "y_index", "x_index", "saa", "sza"),
                (wavelengths + offset).reshape(nw, 1, 1, 1, 1),
            ),
            "radiance_srf": (
                ("y_index", "x_index", "saa", "sza"),
                np.full((1, 1, 1, 1), offset, dtype=float),
            ),
        },
        coords={
            "w": wavelengths,
            "bin_wmin": ("w", wavelengths - 1),
            "bin_wmax": ("w", wavelengths + 1),
            "w_srf": float(wavelengths.mean()),
            "sza": [30.0],
            "saa": [90.0],
            "y_index": [0],
            "x_index": [0],
            "y": ("y_index", [2.0]),
            "x": ("x_index", [3.0]),
            "vza": (("y_index", "x_index"), [[0.0]]),
            "vaa": (("y_index", "x_index"), [[180.0]]),
        },
    )


class CanopyTests(unittest.TestCase):
    def setUp(self):
        self.canopy = bc.Canopy()
        self.wavelengths = np.array([400, 500])
        self.reflectance = np.array([0.1, 0.2])

    def test_spectra_and_materials(self):
        reflectance = self.canopy.spectra(self.wavelengths, self.reflectance)
        transmittance = self.canopy.spectra(
            self.wavelengths, np.array([0.3, 0.4]), "transmittance"
        )
        self.assertEqual(reflectance["quantity"], "reflectance")
        self.assertEqual(reflectance["wavelengths_units"], "nanometers")
        np.testing.assert_array_equal(reflectance["values"], self.reflectance)
        self.assertEqual(transmittance["quantity"], "transmittance")
        self.assertEqual(self.canopy.spectra_wrapper(reflectance)["type"], "lambertian")
        translucent = self.canopy.spectra_wrapper(reflectance, transmittance)
        self.assertEqual(translucent["type"], "translucent")
        self.assertIs(translucent["transmittance"], transmittance)
        self.assertIsNone(self.canopy.spectra(self.wavelengths, None))

    def test_spectra_rejects_invalid_inputs(self):
        cases = [
            ([400, 500], self.reflectance, "reflectance"),
            (self.wavelengths, [0.1, 0.2], "reflectance"),
            (self.wavelengths.reshape(1, 2), self.reflectance, "reflectance"),
            (self.wavelengths, np.array([1, 2]), "reflectance"),
            (self.wavelengths, self.reflectance, "absorption"),
        ]
        for wavelengths, values, kind in cases:
            with self.subTest(kind=kind, wavelengths=wavelengths, values=values):
                with self.assertRaises(IOError):
                    self.canopy.spectra(wavelengths, values, kind)

    def test_mesh_elements_use_cross_referenced_spectra(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            (root / "leaf_mapping.csv").write_text(
                "first.obj,second.txt\nsecond.obj,first.txt\n", encoding="utf-8"
            )
            wavelengths = np.array([[400, 410], [500, 510]])
            reflectances = np.array([[0.1, 0.3], [0.2, 0.4]])
            transmittances = np.array([[0.01, 0.03], [0.02, 0.04]])
            elements = self.canopy.make_mesh_tree_elements(
                [str(root / "first.obj"), str(root / "second.obj")],
                "leaf",
                wavelengths,
                reflectances,
                ["first.txt", "second.txt"],
                transmittances,
                str(root),
            )

        self.assertEqual([element["id"] for element in elements], ["leaf0", "leaf1"])
        self.assertEqual(elements[0]["mesh_filename"], root / "first.obj")
        np.testing.assert_array_equal(elements[0]["reflectance"]["values"], [0.3, 0.4])
        np.testing.assert_array_equal(
            elements[0]["transmittance"]["values"], [0.03, 0.04]
        )
        np.testing.assert_array_equal(
            elements[1]["reflectance"]["wavelengths"], [400, 500]
        )
        np.testing.assert_array_equal(elements[1]["reflectance"]["values"], [0.1, 0.2])

    def test_crossref_rejects_missing_or_duplicate_mesh_entries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping = pathlib.Path(temp_dir) / "leaf_mapping.csv"
            mapping.write_text("a.obj,a.txt\na.obj,b.txt\n", encoding="utf-8")
            lookup = self.canopy.crossref(temp_dir, "leaf")
            with self.assertRaisesRegex(IOError, "No entries"):
                lookup("missing.obj", 0, ["a.txt", "b.txt"])
            with self.assertRaisesRegex(IOError, "More than one entry"):
                lookup("a.obj", 0, ["a.txt", "b.txt"])

    def test_instance_offsets_and_canopy_structure(self):
        elements = [{"id": "tree0"}, {"id": "tree1"}]
        positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        instances = self.canopy.make_instanced_canopy_element(
            elements, positions, np.array([10.0, 20.0, 30.0])
        )
        np.testing.assert_array_equal(
            instances[0]["instance_positions"], [[11, 22, 33], [14, 25, 36]]
        )
        self.assertEqual(instances[0]["canopy_element"]["id"], "tree")
        scene = self.canopy.create_canopy((20, 30, 40), 1, instances)
        self.assertEqual(scene["size"], (20, 30, 40))
        self.assertEqual(scene["padding"], 1)
        self.assertIs(scene["instanced_canopy_elements"], instances)


class OutputTests(unittest.TestCase):
    def test_json_round_trip_numpy_values_and_quantities(self):
        data = {
            "array": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "integer": np.int64(7),
            "float": np.float32(0.5),
            "angle": 30 * bc.ureg.degrees,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "scene.json"
            json_io = bc.JsonManipulation()
            json_io.write_json(data, path)
            restored = json_io.read_json(path)
        np.testing.assert_array_equal(restored["array"], data["array"])
        self.assertEqual(restored["integer"], 7)
        self.assertEqual(restored["float"], 0.5)
        self.assertEqual(restored["angle"], data["angle"])

    def test_json_filename_suffix_and_override(self):
        json_io = bc.JsonManipulation()
        self.assertEqual(
            json_io.change_json_fname("scene.json", False, "direct", [30, 90]),
            "scene_direct_30_90.json",
        )
        self.assertEqual(
            json_io.change_json_fname("scene.json", True, "direct", [30, 90]),
            "scene.json",
        )
        with self.assertRaisesRegex(IOError, "str_prefix"):
            json_io.change_json_fname("scene.json", False)

    def test_padding_preserves_data_and_rejects_short_target(self):
        array = np.array([[1, 2], [3, 4]])
        padded = bc.EradiateXarray.pad_along_axis(array, 3, axis=1)
        np.testing.assert_array_equal(padded[:, :2], array)
        self.assertTrue(np.isnan(padded[:, 2]).all())
        np.testing.assert_array_equal(
            bc.EradiateXarray.pad_along_axis(array, 2, axis=1), array
        )
        with self.assertRaisesRegex(ValueError, "exceeds target length"):
            bc.EradiateXarray.pad_along_axis(array, 1, axis=1)

    def test_merge_bands_pads_spectra_and_writes_netcdf(self):
        bands = [_band_dataset([400, 500]), _band_dataset([600], offset=10)]
        metadata = {
            key: "test"
            for key in (
                "atmosphere",
                "canopy",
                "background_surface",
                "illumination",
                "measure",
                "usr",
                "time",
                "eradiate_version",
                "branch",
                "commit",
                "git_url",
            )
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output = pathlib.Path(temp_dir) / "bands.nc"
            merged = bc.EradiateXarray(**metadata).merge_eradiate_bands_with_padding(
                bands, output, band_names=["blue", "red"]
            )
            with xr.open_dataset(output) as saved:
                np.testing.assert_array_equal(saved["band"].values, ["blue", "red"])
                np.testing.assert_allclose(
                    saved["w"].values, [[400, 500], [600, np.nan]]
                )
                np.testing.assert_allclose(
                    saved["irradiance"].values[:, 0, 0, :],
                    [[400, 500], [610, np.nan]],
                )
                self.assertEqual(saved.attrs["band_count"], 2)
            self.assertEqual(merged.sizes["w_dim"], 2)
            np.testing.assert_allclose(
                merged["radiance"].values[:, :, 0, 0, 0, 0],
                [[400, 500], [610, np.nan]],
            )

    def test_merge_rejects_incompatible_bands(self):
        first = _band_dataset([400, 500])
        second = _band_dataset([600]).assign_coords(sza=[45.0])
        merger = bc.EradiateXarray()
        with self.assertRaisesRegex(ValueError, "non-matching shared coordinate 'sza'"):
            merger.merge_eradiate_bands_with_padding([first, second], "unused.nc")
        with self.assertRaisesRegex(ValueError, "band_names length"):
            merger.merge_eradiate_bands_with_padding(
                [first, second], "unused.nc", band_names=["only-one"]
            )


if __name__ == "__main__":
    unittest.main()
