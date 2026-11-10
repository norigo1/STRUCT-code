"""Unit tests for occlusion-profile and voxel LAI helpers."""

import contextlib
import io
import pathlib
import struct
import tempfile
import unittest

import laspy
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


class PointCloudVegetationAnalysisTests(unittest.TestCase):
    """Verify the point-cloud template routes and sequences its extension hooks."""

    def test_routes_both_sources_and_all_platforms(self):
        """Check each source/platform combination follows the same workflow."""

        class ExampleMethod(occlusion_lai.PointCloudVegetationAnalysis):
            def load_simulated_point_cloud(self):
                return ("simulated", self.point_cloud_path)

            def load_real_point_cloud(self):
                return ("real", self.point_cloud_path)

            def prepare_ground_point_cloud(self, point_cloud):
                return (point_cloud, "ground")

            def prepare_uav_point_cloud(self, point_cloud):
                return (point_cloud, "uav")

            def prepare_airborne_point_cloud(self, point_cloud):
                return (point_cloud, "airborne")

            def calculate_pavd_profile(self, prepared_cloud):
                return ("profile", prepared_cloud)

            def calculate_pai(self, pavd_profile):
                return ("pai", pavd_profile)

        for source in ("simulated", "real"):
            for platform in ("ground", "uav", "airborne"):
                with self.subTest(source=source, platform=platform):
                    analysis = ExampleMethod("scan", source, platform)
                    profile, pai = analysis.run()
                    prepared = ((source, "scan"), platform)
                    self.assertEqual(profile, ("profile", prepared))
                    self.assertEqual(pai, ("pai", profile))

    def test_invalid_source_and_platform_are_rejected(self):
        """Check unsupported choices are reported before processing."""
        with self.assertRaisesRegex(ValueError, "source"):
            occlusion_lai.PointCloudVegetationAnalysis("scan", "other", "ground")
        with self.assertRaisesRegex(ValueError, "platform"):
            occlusion_lai.PointCloudVegetationAnalysis("scan", "real", "other")

    def test_unimplemented_hooks_raise_explicit_errors(self):
        """Verify the base template cannot return placeholder results."""
        analysis = occlusion_lai.PointCloudVegetationAnalysis(
            "scan", "simulated", "ground"
        )
        for method, args in (
            (analysis.load_real_point_cloud, ()),
            (analysis.prepare_uav_point_cloud, (object(),)),
            (analysis.prepare_airborne_point_cloud, (object(),)),
        ):
            with self.subTest(method=method.__name__):
                with self.assertRaises(NotImplementedError):
                    method(*args)
        with self.assertRaisesRegex(TypeError, "HeliosSimulationData"):
            analysis.prepare_ground_point_cloud(object())

    def test_load_helios_las_pulses_and_waveform_text(self):
        """Read LAS returns and preserve separately provided pulse and waveform data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            path = root / "leg000_points.las"
            header = laspy.LasHeader(point_format=6, version="1.4")
            header.add_extra_dim(laspy.ExtraBytesParams(name="fullwaveIndex", type=np.uint64))
            las = laspy.LasData(header)
            las.x, las.y, las.z = [1, 2], [3, 4], [5, 6]
            las.return_number, las.number_of_returns = [1, 2], [2, 2]
            las.classification = [1, 2]
            las.fullwaveIndex = [7, 7]
            las.write(path)
            pulses = root / "leg000_pulse.txt"
            pulses.write_text(
                "0 0 1 0 0 1 1234567890123456789012345678901234567890123456789 7 0\n"
                "0 0 1 1 0 0 1234567890123456790 8 0\n",
                encoding="utf-8",
            )
            waveform = root / "leg000_fullwave.txt"
            waveform.write_text(
                "7 0 0 1 0 0 1 10 12 1234567890123456789 1.5 2.5 0\n",
                encoding="utf-8",
            )
            analysis = occlusion_lai.PointCloudVegetationAnalysis(
                path, "simulated", "ground", pulse_path=pulses, waveform_path=waveform
            )

            data = analysis.load_point_cloud()

            np.testing.assert_array_equal(data.las.fullwaveIndex, [7, 7])
            np.testing.assert_array_equal(data.las.return_number, [1, 2])
            np.testing.assert_allclose(data.las.x, [1, 2])
            self.assertEqual(len(data.pulses), 2)
            self.assertEqual(
                data.pulses["gps_time"][0], "1234567890123456789012345678901234567890123456789"
            )
            np.testing.assert_array_equal(data.pulses["pulse_index"], [7, 8])
            wave = list(data.iter_waveforms())
            self.assertEqual(wave[0]["fullwave_index"], 7)
            self.assertEqual(wave[0]["gps_time"], "1234567890123456789")
            np.testing.assert_allclose(wave[0]["intensities"], [1.5, 2.5, 0])
            self.assertEqual(len(wave), 1)

    def test_load_las_internal_and_external_waveform_packets(self):
        """Read binary waveform payloads relative to internal or external headers."""
        from laspy.vlrs.known import WaveformPacketStruct, WaveformPacketVlr

        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            for point_format, internal in (
                (fmt, flag) for fmt in (4, 5, 9, 10) for flag in (True, False)
            ):
                with self.subTest(internal=internal, point_format=point_format):
                    path = root / f"{point_format}_{'internal' if internal else 'external'}.las"
                    version = "1.3" if point_format in (4, 5) else "1.4"
                    header = laspy.LasHeader(point_format=point_format, version=version)
                    header.global_encoding.waveform_data_packets_internal = internal
                    header.global_encoding.waveform_data_packets_external = not internal
                    descriptor = WaveformPacketVlr(100)
                    descriptor.parsed_record = WaveformPacketStruct()
                    descriptor.parsed_record.bits_per_sample = 8
                    descriptor.parsed_record.number_of_samples = 3
                    descriptor.parsed_record.digitizer_gain = 2
                    header.vlrs.append(descriptor)
                    las = laspy.LasData(header)
                    las.x, las.y, las.z = [1], [2], [3]
                    las.wavepacket_index = [1]
                    las.wavepacket_offset = [60]
                    las.wavepacket_size = [3]
                    las.write(path)
                    record = struct.pack("<H16sHQ32s", 0, b"LASF_Spec", 65535, 3, b"")
                    if internal:
                        with path.open("ab") as file:
                            start = file.tell()
                            file.write(record + b"\x01\x02\x03")
                        with path.open("r+b") as file:
                            file.seek(227)
                            file.write(struct.pack("<Q", start))
                    else:
                        path.with_suffix(".wdp").write_bytes(record + b"\x01\x02\x03")

                    analysis = occlusion_lai.PointCloudVegetationAnalysis(
                        path, "simulated", "ground"
                    )
                    data = analysis.load_point_cloud()
                    packet, metadata = data.read_waveform_packet(0)
                    self.assertEqual(packet, b"\x01\x02\x03")
                    self.assertEqual(metadata.number_of_samples, 3)
                    self.assertEqual(metadata.digitizer_gain, 2)
                    if not internal:
                        path.with_suffix(".wdp").write_bytes(record + b"\x01")
                        with self.assertRaisesRegex(ValueError, "Truncated waveform packet"):
                            data.read_waveform_packet(0)
                        path.with_suffix(".wdp").unlink()
                        with self.assertRaises(FileNotFoundError):
                            data.read_waveform_packet(0)

    def test_load_helios_reports_missing_and_malformed_inputs(self):
        """Report missing, unsupported, or malformed input rather than inventing pulses."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            path = root / "scan.las"
            laspy.LasData(laspy.LasHeader(point_format=6, version="1.4")).write(path)
            analysis = occlusion_lai.PointCloudVegetationAnalysis(path, "simulated", "ground")
            data = analysis.load_point_cloud()
            self.assertIsNone(data.pulses)
            with self.assertRaisesRegex(ValueError, "No HELIOS"):
                list(data.iter_waveforms())
            with self.assertRaisesRegex(ValueError, "point_index"):
                data.read_waveform_packet(0)
            with self.assertRaisesRegex(ValueError, "LAS or LAZ"):
                occlusion_lai.PointCloudVegetationAnalysis(
                    root / "scan.xyz", "simulated", "ground"
                ).load_point_cloud()
            with self.assertRaises(FileNotFoundError):
                occlusion_lai.PointCloudVegetationAnalysis(
                    path, "simulated", "ground", waveform_path=root / "missing.txt"
                ).load_point_cloud()
            waveform = root / "waveform.txt"
            waveform.write_text("invalid waveform\n", encoding="utf-8")
            data = occlusion_lai.PointCloudVegetationAnalysis(
                path, "simulated", "ground", waveform_path=waveform
            ).load_point_cloud()
            with self.assertRaisesRegex(ValueError, "Malformed waveform"):
                list(data.iter_waveforms())
            pulses = root / "pulses.txt"
            pulses.write_text("0 0 1 0 nan 1 123 0 0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be finite"):
                occlusion_lai.PointCloudVegetationAnalysis(
                    path, "simulated", "ground", pulse_path=pulses
                ).load_point_cloud()
            pulses.write_text("0 0 1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                occlusion_lai.PointCloudVegetationAnalysis(
                    path, "simulated", "ground", pulse_path=pulses
                ).load_point_cloud()
            with self.assertRaisesRegex(ValueError, "pulse file"):
                analysis.run()

    def test_prepare_helios_ground_scan_and_run(self):
        """Associate shuffled returns with emitted pulses and ground-reference heights."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            path = root / "scan.las"
            header = laspy.LasHeader(point_format=6, version="1.4")
            header.add_extra_dim(laspy.ExtraBytesParams(name="fullwaveIndex", type=np.int32))
            las = laspy.LasData(header)
            las.x, las.y, las.z = [1, 2, 3], [0, 0, 0], [101, 102, 103]
            las.fullwaveIndex = [10, 30, 10]
            las.return_number = [2, 1, 1]
            las.number_of_returns = [2, 1, 2]
            las.classification = [2, 3, 4]
            las.write(path)
            pulses = root / "pulses.txt"
            pulses.write_text(
                "0 0 100 0.7934 0 0.6088 1 30 0\n"
                "0 0 100 0 0.7934 0.6088 2 20 0\n"
                "0 0 100 0.7934 0 0.6088 3 10 0\n"
                "0 0 100 -0.7934 0 0.6088 4 40 0\n",
                encoding="utf-8",
            )
            settings = occlusion_lai.CaldersProfileInput(
                scans=(), max_height=4, height_bin_size=1
            )
            analysis = occlusion_lai.PointCloudVegetationAnalysis(
                path, "simulated", "ground", pulse_path=pulses,
                ground_elevation=100, calders_settings=settings, zenith_range=(50, 55),
            )
            data = analysis.load_point_cloud()
            prepared = analysis.prepare_ground_point_cloud(data)
            scan = prepared.scans[0]
            np.testing.assert_array_equal(scan.return_pulse_index, [2, 0, 2])
            np.testing.assert_array_equal(scan.pulse_return_count, [1, 0, 2, 0])
            np.testing.assert_allclose(scan.return_height, [1, 2, 3])
            np.testing.assert_allclose(scan.pulse_azimuth, [0, 90, 0, 180])
            np.testing.assert_allclose(scan.pulse_zenith, 52.5, atol=0.01)
            np.testing.assert_array_equal(scan.return_classification, [2, 3, 4])
            np.testing.assert_array_equal(scan.return_number, [2, 1, 1])
            profile, pai = analysis.run()
            np.testing.assert_allclose(profile["pgap"], [[1, 1, 0.75, 0.5]])
            self.assertEqual(set(pai), {"hinge", "linear", "weighted"})

            analysis.calders_settings = occlusion_lai.CaldersProfileInput(
                scans=(), max_height=4, height_bin_size=1, weighted=True
            )
            weighted = analysis.calculate_pavd_profile(analysis.prepare_ground_point_cloud(data))
            np.testing.assert_allclose(weighted["pgap"], [[1, 0.875, 0.625, 0.5]])
            analysis.ground_elevation = [100, 101, 102]
            scan = analysis.prepare_ground_point_cloud(data).scans[0]
            np.testing.assert_allclose(scan.return_height, [1, 1, 1])
            self.assertEqual(settings.scans, ())

            analysis.ground_elevation = [100, 101]
            with self.assertRaisesRegex(ValueError, "ground_elevation"):
                analysis.prepare_ground_point_cloud(data)
            analysis.ground_elevation = None
            with self.assertRaisesRegex(ValueError, "ground_elevation"):
                analysis.prepare_ground_point_cloud(data)

            analysis.ground_elevation = 100
            analysis.azimuth_range = (350, 10)
            with self.assertRaisesRegex(ValueError, "angular ranges"):
                analysis.prepare_ground_point_cloud(data)
            analysis.azimuth_range = (0, 360)
            data.pulses["pulse_index"][0] = 10
            with self.assertRaisesRegex(ValueError, "unique"):
                analysis.prepare_ground_point_cloud(data)
            data.pulses["pulse_index"][0] = 31
            with self.assertRaisesRegex(ValueError, "no matching"):
                analysis.prepare_ground_point_cloud(data)
            data.pulses["pulse_index"][0] = 30
            data.pulses["device_index"][0] = 1
            with self.assertRaisesRegex(ValueError, "Multiple scanner devices"):
                analysis.prepare_ground_point_cloud(data)
            data.pulses["device_index"][0] = 0
            data.pulses["origin_x"][0] = 0.1
            with self.assertRaisesRegex(ValueError, "single scanner position"):
                analysis.prepare_ground_point_cloud(data)
            data.pulses["origin_x"][0] = 0
            data.pulses["direction_x"][0] = 0
            data.pulses["direction_z"][0] = 0
            with self.assertRaisesRegex(ValueError, "non-zero"):
                analysis.prepare_ground_point_cloud(data)
            data.pulses["direction_x"][0] = 0.7934
            data.pulses["direction_z"][0] = 0.6088
            data.las.number_of_returns = [2, 2, 2]
            with self.assertRaisesRegex(ValueError, "return counts"):
                analysis.prepare_ground_point_cloud(data)
            data.las.number_of_returns = [2, 1, 2]
            data.las.fullwaveIndex = [-1, 30, 10]
            with self.assertRaisesRegex(ValueError, "fullwaveIndex"):
                analysis.prepare_ground_point_cloud(data)

    def test_pai_simpson_integration_and_sampled_range(self):
        """Integrate three quadratic profiles and check exact reported bounds."""
        analysis = occlusion_lai.PointCloudVegetationAnalysis("scan", "real", "ground")
        heights = np.arange(5, dtype=float)
        profile = {
            "height": heights,
            "hinge_pavd": heights**2,
            "linear_pavd": 2 * heights**2,
            "weighted_pavd": 3 * heights**2,
        }

        full = analysis.calculate_pai(profile)
        partial = analysis.calculate_pai(profile, height_range=(1, 3))

        for estimator, multiplier in (("hinge", 1), ("linear", 2), ("weighted", 3)):
            self.assertAlmostEqual(full[estimator]["pai"], multiplier * 64 / 3)
            self.assertAlmostEqual(partial[estimator]["pai"], multiplier * 26 / 3)
            self.assertEqual(full[estimator]["height_min"], 0)
            self.assertEqual(full[estimator]["height_max"], 4)
            self.assertEqual(partial[estimator]["height_min"], 1)
            self.assertEqual(partial[estimator]["height_max"], 3)
            self.assertEqual(len(full[estimator]["nan_bins"]), 0)
            self.assertEqual(len(full[estimator]["infinite_bins"]), 0)

    def test_pai_reports_nonfinite_bins_without_skipping_them(self):
        """Mark PAI unavailable and report original indices for selected non-finite bins."""
        analysis = occlusion_lai.PointCloudVegetationAnalysis("scan", "real", "ground")
        profile = {
            "height": np.arange(5.0),
            "hinge_pavd": [0, np.nan, 2, np.inf, 4],
            "linear_pavd": [np.nan, 1, 2, 3, np.inf],
            "weighted_pavd": [0, -np.inf, np.nan, np.inf, 4],
        }

        result = analysis.calculate_pai(profile, height_range=(1, 3))

        self.assertTrue(np.isnan(result["hinge"]["pai"]))
        np.testing.assert_array_equal(result["hinge"]["nan_bins"], [1])
        np.testing.assert_array_equal(result["hinge"]["infinite_bins"], [3])
        self.assertAlmostEqual(result["linear"]["pai"], 4.0)
        np.testing.assert_array_equal(result["linear"]["nan_bins"], [])
        np.testing.assert_array_equal(result["linear"]["infinite_bins"], [])
        self.assertTrue(np.isnan(result["weighted"]["pai"]))
        np.testing.assert_array_equal(result["weighted"]["nan_bins"], [2])
        np.testing.assert_array_equal(result["weighted"]["infinite_bins"], [1, 3])

    def test_pai_validates_profile_and_height_range(self):
        """Reject invalid heights, ranges, and profile shapes."""
        analysis = occlusion_lai.PointCloudVegetationAnalysis("scan", "real", "ground")
        profile = {
            "height": [0, 1, 2],
            "hinge_pavd": [1, 1, 1],
            "linear_pavd": [1, 1, 1],
            "weighted_pavd": [1, 1, 1],
        }
        for bounds in ((0.1, 2), (0, 3), (2, 1), (1, 1), (0, np.nan)):
            with self.subTest(bounds=bounds):
                with self.assertRaises(ValueError):
                    analysis.calculate_pai(profile, height_range=bounds)
        for heights in ([0], [0, 0, 1], [0, np.nan, 2]):
            with self.subTest(heights=heights):
                with self.assertRaises(ValueError):
                    analysis.calculate_pai({**profile, "height": heights})
        with self.assertRaisesRegex(ValueError, "linear_pavd"):
            analysis.calculate_pai({**profile, "linear_pavd": [1, 2]})
        with self.assertRaises(KeyError):
            analysis.calculate_pai({"height": [0, 1]})

    def test_pai_two_samples_and_roundoff_in_range(self):
        """Verify two samples use a trapezoid and round-off does not reject bounds."""
        analysis = occlusion_lai.PointCloudVegetationAnalysis("scan", "real", "ground")
        profile = {
            "height": [0.1 + 0.2, 1.3, 2.3],
            "hinge_pavd": [1, 3, 5],
            "linear_pavd": [1, 3, 5],
            "weighted_pavd": [1, 3, 5],
        }
        result = analysis.calculate_pai(profile, height_range=(0.3, 1.3))
        self.assertAlmostEqual(result["hinge"]["pai"], 2.0)
        self.assertEqual(result["hinge"]["height_min"], profile["height"][0])

    def test_run_integrates_real_calders_profiles(self):
        """Check the workflow passes calculated PAVD profiles to PAI integration."""
        scan = occlusion_lai.CaldersScan(
            pulse_zenith=np.repeat([42.5, 47.5, 52.5], 4),
            pulse_azimuth=np.zeros(12),
            pulse_return_count=np.tile([1, 1, 1, 0], 3),
            return_pulse_index=[0, 1, 2, 4, 5, 6, 8, 9, 10],
            return_height=np.tile([0.5, 1.5, 2.5], 3),
            return_number=np.ones(9, dtype=int),
            return_classification=np.ones(9, dtype=int),
            min_zenith=40,
            max_zenith=55,
        )
        inputs = occlusion_lai.CaldersProfileInput(
            scans=(scan,), max_height=3, height_bin_size=1
        )

        class PreparedScanAnalysis(occlusion_lai.PointCloudVegetationAnalysis):
            def load_real_point_cloud(self):
                return inputs

            def prepare_ground_point_cloud(self, point_cloud):
                return point_cloud

        analysis = PreparedScanAnalysis("scan", "real", "ground")
        profiles, result = analysis.run()

        for estimator in ("hinge", "linear", "weighted"):
            values = profiles[f"{estimator}_pavd"]
            expected = (values[0] + 4 * values[1] + values[2]) / 3
            self.assertAlmostEqual(result[estimator]["pai"], expected)
            self.assertEqual(result[estimator]["height_max"], 2.0)

    def test_calders_profiles_from_known_pulse_counts(self):
        """Check all three estimators and no-return pulse denominators."""
        angles = np.repeat([42.5, 47.5, 52.5], 4)
        scan = occlusion_lai.CaldersScan(
            pulse_zenith=angles,
            pulse_azimuth=np.zeros(12),
            pulse_return_count=np.tile([1, 1, 1, 0], 3),
            return_pulse_index=[0, 1, 2, 4, 5, 6, 8, 9, 10],
            return_height=np.tile([0.5, 1.5, 2.5], 3),
            return_number=np.ones(9, dtype=int),
            return_classification=np.ones(9, dtype=int),
            min_zenith=40,
            max_zenith=55,
        )
        inputs = occlusion_lai.CaldersProfileInput(
            scans=(scan,), max_height=3, height_bin_size=1
        )
        analysis = occlusion_lai.PointCloudVegetationAnalysis("scan", "real", "ground")

        result = analysis.calculate_pavd_profile(inputs)

        gap = np.array([0.75, 0.5, 0.25])
        depth = -np.log(gap)
        np.testing.assert_array_equal(result["height"], [0, 1, 2])
        np.testing.assert_array_equal(result["zenith"], [42.5, 47.5, 52.5])
        np.testing.assert_allclose(result["pgap"], np.tile(gap, (3, 1)))
        np.testing.assert_allclose(result["hinge_pai"], 1.1 * depth)
        np.testing.assert_allclose(result["linear_pai"], depth, atol=1e-14)
        np.testing.assert_allclose(result["weighted_pai"], 1.1 * depth)
        for estimator, factor in (("hinge", 1.1), ("linear", 1.0), ("weighted", 1.1)):
            np.testing.assert_allclose(
                result[estimator + "_pavd"], np.gradient(factor * depth), atol=1e-14
            )

    def test_calders_weighted_returns_and_class_filter(self):
        """Check multi-return weighting uses original pulse counts after exclusion."""
        scan = occlusion_lai.CaldersScan(
            pulse_zenith=[52.5, 52.5],
            pulse_azimuth=[0, 0],
            pulse_return_count=[2, 0],
            return_pulse_index=[0, 0],
            return_height=[0.5, 1.5],
            return_number=[1, 2],
            return_classification=[1, 2],
            min_zenith=50,
            max_zenith=55,
        )
        analysis = occlusion_lai.PointCloudVegetationAnalysis("scan", "real", "ground")
        for weighted, excluded, expected in (
            (False, (), [0.5, 0.5]),
            (True, (), [0.75, 0.5]),
            (True, (2,), [0.75, 0.75]),
        ):
            with self.subTest(weighted=weighted, excluded=excluded):
                config = occlusion_lai.CaldersProfileInput(
                    scans=(scan,), max_height=2, height_bin_size=1,
                    weighted=weighted, excluded_classes=excluded,
                )
                result = analysis.calculate_pavd_profile(config)
                np.testing.assert_allclose(result["pgap"][0], expected)
                self.assertTrue(np.isnan(result["linear_pai"]).all())

    def test_calders_rejects_bad_input_and_other_platforms(self):
        """Check incomplete pulse records and unsupported platforms fail explicitly."""
        scan = occlusion_lai.CaldersScan(
            pulse_zenith=[52.5], pulse_azimuth=[0], pulse_return_count=[2],
            return_pulse_index=[0], return_height=[1], return_number=[1],
            return_classification=[1], min_zenith=50, max_zenith=55,
        )
        config = occlusion_lai.CaldersProfileInput(scans=(scan,))
        ground = occlusion_lai.PointCloudVegetationAnalysis("scan", "real", "ground")
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            ground.calculate_pavd_profile(config)
        with self.assertRaisesRegex(TypeError, "CaldersProfileInput"):
            ground.calculate_pavd_profile(object())
        for platform in ("uav", "airborne"):
            with self.subTest(platform=platform):
                analysis = occlusion_lai.PointCloudVegetationAnalysis("scan", "real", platform)
                with self.assertRaisesRegex(ValueError, "ground scan"):
                    analysis.calculate_pavd_profile(config)

    def test_calders_scans_angles_and_external_pai(self):
        """Check scans with separate angular limits share the selected total PAI."""
        first = occlusion_lai.CaldersScan(
            pulse_zenith=[40, 45, 45, 45], pulse_azimuth=[0, 0, 180, 0],
            pulse_return_count=[0, 1, 0, 0],
            return_pulse_index=[1], return_height=[0.5], return_number=[1],
            return_classification=[1], min_zenith=40, max_zenith=45,
            min_azimuth=0, max_azimuth=90,
        )
        second = occlusion_lai.CaldersScan(
            pulse_zenith=[50, 50], pulse_azimuth=[0, 0],
            pulse_return_count=[1, 0], return_pulse_index=[0],
            return_height=[1.5], return_number=[1], return_classification=[1],
            min_zenith=45, max_zenith=50,
        )
        config = occlusion_lai.CaldersProfileInput(
            scans=(first, second), max_height=2, height_bin_size=1,
            total_pai_method="external", external_pai=2,
        )
        analysis = occlusion_lai.PointCloudVegetationAnalysis("scan", "real", "ground")
        result = analysis.calculate_pavd_profile(config)

        np.testing.assert_array_equal(result["zenith"], [42.5, 47.5])
        np.testing.assert_allclose(result["pgap"], [[0.5, 0.5], [1, 0.5]])
        weights = np.sin(np.radians([42.5, 47.5]))
        np.testing.assert_allclose(result["weighted_pai"], [2 * weights[0] / sum(weights), 2])

    def test_calders_empty_returns_and_missing_data(self):
        """Check no-return pulses and unobserved bins do not become false vegetation."""
        scan = occlusion_lai.CaldersScan(
            pulse_zenith=[42.5], pulse_azimuth=[0], pulse_return_count=[0],
            return_pulse_index=[], return_height=[], return_number=[],
            return_classification=[], min_zenith=40, max_zenith=50,
        )
        config = occlusion_lai.CaldersProfileInput(
            scans=(scan,), max_height=2, height_bin_size=1,
        )
        analysis = occlusion_lai.PointCloudVegetationAnalysis("scan", "simulated", "ground")
        result = analysis.calculate_pavd_profile(config)
        np.testing.assert_array_equal(result["pgap"][0], [1, 1])
        self.assertTrue(np.isnan(result["pgap"][1]).all())
        self.assertTrue(np.isnan(result["weighted_pai"]).all())
        self.assertTrue(np.isnan(result["linear_pai"]).all())

    def test_calders_closed_canopy_remains_explicitly_undefined(self):
        """Check zero gap produces infinite or undefined values rather than zeros."""
        scan = occlusion_lai.CaldersScan(
            pulse_zenith=[52.5], pulse_azimuth=[0], pulse_return_count=[1],
            return_pulse_index=[0], return_height=[0.5], return_number=[1],
            return_classification=[1], min_zenith=50, max_zenith=55,
        )
        config = occlusion_lai.CaldersProfileInput(scans=(scan,), max_height=2,
                                                  height_bin_size=1)
        analysis = occlusion_lai.PointCloudVegetationAnalysis("scan", "real", "ground")
        result = analysis.calculate_pavd_profile(config)
        np.testing.assert_array_equal(result["pgap"], [[0, 0]])
        self.assertTrue(np.isinf(result["hinge_pai"]).all())
        self.assertTrue(np.isnan(result["weighted_pai"]).all())


if __name__ == "__main__":
    unittest.main()
