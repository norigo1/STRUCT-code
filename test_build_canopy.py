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

    def test_crossref_defaults_to_matching_order_without_a_mapping_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            lookup = self.canopy.crossref(temp_dir, "leaf")
            self.assertEqual(lookup("a.obj", 1, ["a.txt", "b.txt"]), 1)

    def test_crossref_accepts_a_single_mapping_row(self):
        """A one-row mapping should work like a multi-row mapping."""
        with tempfile.TemporaryDirectory() as temp_dir:
            (pathlib.Path(temp_dir) / "leaf_mapping.csv").write_text(
                "a.obj,b.txt\n", encoding="utf-8"
            )
            lookup = self.canopy.crossref(temp_dir, "leaf")
            self.assertEqual(lookup("a.obj", 0, ["a.txt", "b.txt"]), 1)

    def test_shared_spectrum_is_used_for_multiple_meshes(self):
        elements = self.canopy.make_mesh_tree_elements(
            ["a.obj", "b.obj"],
            "wood",
            self.wavelengths,
            self.reflectance,
            ["wood.txt"],
        )
        self.assertEqual(len(elements), 2)
        for element in elements:
            np.testing.assert_array_equal(
                element["reflectance"]["values"], self.reflectance
            )
            self.assertNotIn("transmittance", element)

    def test_automatic_canopy_creation_writes_meshes_and_spectra(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            mesh_dir = root / "meshes" / "leaves"
            spectra_dir = root / "spectra" / "leaves"
            mesh_dir.mkdir(parents=True)
            spectra_dir.mkdir(parents=True)
            (mesh_dir / "leaf.obj").write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8"
            )
            np.savetxt(spectra_dir / "leaf.txt", [[400, 0.1, 0.2], [500, 0.3, 0.4]])
            output = root / "canopy.json"
            dem = {
                "type": "instanced",
                "instance_positions": np.array([[1.0, 2.0, 3.0]]),
                "centre_offsets": np.array([1.0, 2.0, 3.0]),
            }
            options = {
                "directory": str(root),
                "meshes_dir": "meshes",
                "mesh_names": ["leaves"],
                "spectra_dir": "spectra",
                "canopy_pos": np.array([[0.0, 0.0, 0.0]]),
                "dem": dem,
                "size": (10, 10, 10),
                "padding": 0,
                "out_json": str(output),
            }
            with patch.dict(os.environ, {"ERADIATE_PROJECT_DIR": str(root) + os.sep}):
                bc.Canopy(**options)
                scene = bc.JsonManipulation().read_json(output)
            tree = scene["instanced_canopy_elements"][0]
            np.testing.assert_array_equal(tree["instance_positions"], [[1, 2, 3]])
            element = tree["canopy_element"]["mesh_tree_elements"][0]
            self.assertEqual(element["mesh_filename"], mesh_dir / "leaf.obj")
            np.testing.assert_allclose(element["transmittance"]["values"], [0.2, 0.4])

    def test_automatic_canopy_rejects_bad_position_shape(self):
        with self.assertRaisesRegex(IOError, "canopy_pos"):
            bc.Canopy(
                directory="project",
                meshes_dir="meshes",
                mesh_names=["leaves"],
                spectra_dir="spectra",
                canopy_pos=np.array([0.0, 0.0, 0.0]),
                dem={},
                size=(1, 1, 1),
                padding=0,
                out_json="unused.json",
            )

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


class SceneConfigurationTests(unittest.TestCase):
    def test_illumination_units_and_boundary_angles(self):
        illumination = bc.Illumination(angle_units=bc.ureg.degrees)
        light = illumination.directional(0, 360, irradiance={"value": 2})
        self.assertEqual(light["zenith"], 0 * bc.ureg.degrees)
        self.assertEqual(light["azimuth"], 360 * bc.ureg.degrees)
        self.assertEqual(light["irradiance"], {"value": 2})
        self.assertEqual(illumination.isotropic()["type"], "constant")
        with self.assertRaisesRegex(IOError, "angle_units"):
            bc.Illumination(angle_units=bc.ureg.m)

    def test_illumination_rejects_out_of_range_zenith(self):
        with self.assertRaises(IOError):
            bc.Illumination().directional(-1, 0)

    def test_sun_angular_diameter_is_converted_to_radians(self):
        light = bc.Illumination(angle_units=bc.ureg.radians).sun(0, 0)
        self.assertAlmostEqual(light["angular_diameter"].magnitude, np.deg2rad(0.5385))

    def test_perspective_camera_validates_input_and_preserves_srf(self):
        camera = bc.Camera()
        options = {
            "spp": 1,
            "npix": (1, 1),
            "fov": 45.0,
            "camloc": (0.0, -1.0, 1.0),
            "tarloc": (0.0, 0.0, 0.0),
            "srf": {"type": "multi_delta", "wavelengths": [550.0]},
            "idd": "camera.1",
        }
        measure = camera.perspective_camera(**options)
        self.assertEqual(measure["id"], "camera,1")
        self.assertEqual(measure["film_resolution"], (1, 1))
        self.assertEqual(measure["srf"], options["srf"])
        for changes in ({"spp": 1.5}, {"npix": (1,)}, {"camloc": (0, 0, 1)}):
            with self.subTest(changes=changes), self.assertRaises(IOError):
                camera.perspective_camera(**(options | changes))

    def test_camera_distant_and_hemispherical_geometry(self):
        camera = bc.Camera()
        position = (0.0, -3.0, 4.0)
        target = (0.0, 0.0, 0.0)
        distant = camera.mpdistant_camera(
            1, (1, 2), position, target, srf="response", idd="test"
        )
        self.assertEqual(distant["direction"], (0.0, -3.0, 4.0))
        self.assertEqual(distant["ray_offset"], 5.0)
        self.assertEqual(distant["srf"]["path"], pathlib.Path("response.nc"))
        hemispherical = camera.hemispherical_camera(
            1, (1, 2), position, target, srf="response", idd="test"
        )
        self.assertEqual(hemispherical["fov"], 180 * bc.ureg.degrees)

    def test_sentinel_all_bands_and_invalid_inputs_without_simulation(self):
        camera = bc.Camera()
        options = {
            "spp": 1,
            "npix": (1, 1),
            "camlocx": 0.0,
            "camlocy": 0.0,
            "tarloc": (0.0, 0.0, 0.0),
            "version": "a",
        }
        with (
            patch.object(camera, "srf_path_solving", return_value="/srf/"),
            patch.object(
                camera,
                "mpdistant_camera",
                side_effect=lambda *args, **kwargs: kwargs,
            ) as make_camera,
        ):
            measures = camera.sentinel2_msi(**options)
            self.assertEqual(len(measures), 13)
            self.assertEqual(make_camera.call_count, 13)
            self.assertTrue(measures[0]["srf"].endswith("-1"))
            self.assertTrue(measures[-1]["srf"].endswith("-12"))
            for changes in ({"version": "c"}, {"res": "invalid"}, {"bands": ["99"]}):
                with self.subTest(changes=changes), self.assertRaises(IOError):
                    camera.sentinel2_msi(**(options | changes))

    def test_sentinel_supports_documented_20m_resolution(self):
        camera = bc.Camera()
        with (
            patch.object(camera, "srf_path_solving", return_value="/srf/"),
            patch.object(camera, "mpdistant_camera", return_value={}),
        ):
            measures = camera.sentinel2_msi(
                1, (1, 1), 0.0, 0.0, (0.0, 0.0, 0.0), "a", res="20m"
            )
        self.assertEqual(len(measures), 13)

    def test_sentinel_supports_selecting_individual_bands(self):
        camera = bc.Camera()
        with (
            patch.object(camera, "srf_path_solving", return_value="/srf/"),
            patch.object(camera, "mpdistant_camera", return_value={}),
        ):
            measures = camera.sentinel2_msi(
                1, (1, 1), 0.0, 0.0, (0.0, 0.0, 0.0), "a", bands=["2"]
            )
        self.assertEqual(len(measures), 1)

    def test_spectra_loads_two_and_three_column_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            path = root / "leaf.txt"
            np.savetxt(path, [[400, 0.1, 0.2], [500, 0.3, 0.4]])
            leaf = bc.Spectra(str(root) + os.sep)
            np.testing.assert_allclose(leaf.wvs, [400, 500])
            np.testing.assert_allclose(leaf.trans, [0.2, 0.4])
            np.savetxt(path, [[400, 0.1], [500, 0.3]])
            wood = bc.Spectra(str(root) + os.sep)
            self.assertIsNone(wood.trans)
            np.testing.assert_allclose(wood.refl, [0.1, 0.3])

    def test_multiple_spectra_are_matched_to_filenames(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            np.savetxt(root / "a.txt", [[400, 0.1], [500, 0.2]])
            np.savetxt(root / "b.txt", [[400, 0.3], [500, 0.4]])
            spectra = bc.Spectra(str(root) + os.sep)
            self.assertIsNone(spectra.trans)
            for name, expected in (("a.txt", [0.1, 0.2]), ("b.txt", [0.3, 0.4])):
                index = spectra.fnames.index(name)
                np.testing.assert_allclose(spectra.refl[:, index], expected)
                np.testing.assert_allclose(spectra.wvs[:, index], [400, 500])

    def test_dem_mesh_points_use_centred_pixel_coordinates(self):
        dem = bc.DEMFromTiff.__new__(bc.DEMFromTiff)
        dem.resolution = 2
        dem.dem_arr = np.array([[[1, 2], [3, 4]]])
        points = dem.prepare_mesh()
        np.testing.assert_allclose(
            points, [[-1, -1, 1], [1, -1, 2], [-1, 1, 3], [1, 1, 4]]
        )
        mesh = dem.create_mesh(points)
        np.testing.assert_allclose(np.asarray(mesh.vertices), points)
        self.assertEqual(len(mesh.triangles), 2)


class SimulationTests(unittest.TestCase):
    def test_all_and_sequential_orders_and_invalid_configuration(self):
        components = (
            ["c1.json", "c2.json"],
            ["b1.json", "b2.json"],
            ["a1.json", "a2.json"],
            ["i1.json", "i2.json"],
            ["m1.json", "m2.json"],
            ["/p1", "/p2"],
        )
        all_runs = bc.RunSimulations(*components)
        sequential = bc.RunSimulations(*components, order="sequential")
        self.assertEqual(all_runs.len_sims, 32)
        self.assertEqual(sequential.len_sims, 2)
        self.assertEqual(
            sequential.simulations[1][:5],
            ("a2.json", "c2.json", "b2.json", "i2.json", "m2.json"),
        )
        with self.assertRaisesRegex(IOError, "canopies.*project_dirs"):
            bc.RunSimulations(*components[:-1], ["/one"])
        with self.assertRaisesRegex(IOError, "same length"):
            bc.RunSimulations(
                components[0], ["b1.json"], *components[2:], order="sequential"
            )
        with self.assertRaisesRegex(IOError, "order"):
            bc.RunSimulations(*components, order="unknown")

    def test_single_result_is_saved_with_metadata_without_running_eradiate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = bc.RunSimulations(
                ["canopy.json"],
                ["background.json"],
                ["atmosphere.json"],
                ["illumination.json"],
                ["measure.json"],
                [temp_dir],
                results_dir=temp_dir + os.sep,
            )
            dataset = xr.Dataset({"radiance": ("w", [1.5, 2.5])})
            header = ("user", "time", "version", "branch", "commit", "url")
            with (
                patch.object(runner.jm, "read_json", side_effect=[{}, {}, {}, {}, {}]),
                patch.object(
                    bc.eradiate.experiments,
                    "CanopyAtmosphereExperiment",
                    return_value=object(),
                ),
                patch.object(bc.eradiate, "run", return_value=dataset) as run,
                patch.object(runner, "get_logfile_header", return_value=header),
            ):
                runner.run_simulations()
            run.assert_called_once()
            with xr.open_dataset(runner.result_fnames[0]) as saved:
                np.testing.assert_allclose(saved["radiance"], [1.5, 2.5])
                self.assertEqual(saved.attrs["canopy"], "canopy.json")
                self.assertEqual(saved.attrs["atmosphere"], "atmosphere.json")
                self.assertEqual(saved.attrs["user"], "user")
                self.assertEqual(saved.attrs["git_url"], "url")

    def test_multiple_band_results_are_merged_without_running_eradiate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = bc.RunSimulations(
                ["canopy.json"],
                ["background.json"],
                ["atmosphere.json"],
                ["illumination.json"],
                ["measure.json"],
                [temp_dir],
                results_dir=temp_dir + os.sep,
            )
            measures = [
                {"srf": xr.DataArray(0, attrs={"band": "blue"})},
                {"srf": xr.DataArray(0, attrs={"band": "red"})},
            ]
            header = ("user", "time", "version", "branch", "commit", "url")
            with (
                patch.object(
                    runner.jm, "read_json", side_effect=[{}, {}, {}, {}, measures]
                ),
                patch.object(
                    bc.eradiate.experiments,
                    "CanopyAtmosphereExperiment",
                    return_value=object(),
                ),
                patch.object(
                    bc.eradiate,
                    "run",
                    return_value={
                        "blue": _band_dataset([500, 510]),
                        "red": _band_dataset([650]),
                    },
                ) as run,
                patch.object(runner, "get_logfile_header", return_value=header),
            ):
                runner.run_simulations()
            run.assert_called_once()
            with xr.open_dataset(runner.result_fnames[0]) as saved:
                np.testing.assert_array_equal(saved["band"], ["blue", "red"])
                np.testing.assert_allclose(saved["w"], [[500, 510], [650, np.nan]])
                self.assertEqual(saved.attrs["canopy"], "canopy.json")


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

    def test_merge_rejects_empty_or_incomplete_datasets(self):
        merger = bc.EradiateXarray()
        with self.assertRaisesRegex(ValueError, "empty"):
            merger.merge_eradiate_bands_with_padding([], "unused.nc")
        with self.assertRaisesRegex(ValueError, "missing data variable 'radiance'"):
            merger.merge_eradiate_bands_with_padding(
                [_band_dataset([500]).drop_vars("radiance")], "unused.nc"
            )
        with self.assertRaisesRegex(ValueError, "missing coordinate 'w_srf'"):
            merger.merge_eradiate_bands_with_padding(
                [_band_dataset([500]).drop_vars("w_srf")], "unused.nc"
            )

    def test_padding_custom_fill_preserves_integer_dtype(self):
        array = np.array([[1, 2], [3, 4]], dtype=np.int64)
        padded = bc.EradiateXarray.pad_along_axis(array, 3, axis=0, fill_value=-1)
        self.assertEqual(padded.dtype, array.dtype)
        np.testing.assert_array_equal(padded, [[1, 2], [3, 4], [-1, -1]])


if __name__ == "__main__":
    unittest.main()
