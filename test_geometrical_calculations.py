"""Unit tests for 3D geometry and voxel LAI calculations."""

import contextlib
import io
import unittest
from unittest import mock

import numpy as np
from scipy.spatial import cKDTree

from geometrical_calculations import LAICalculations, PreparedFacets, polygon_area_3d


class GeometricalCalculationsTests(unittest.TestCase):
    """Check polygon areas, voxel LAI, and segment intersections."""

    def test_polygon_area_and_voxel_lai_for_contained_triangle(self):
        """Check polygon area and LAI for a triangle fully inside one voxel."""
        triangle = np.array([[0.1, 0.1, 0.5], [0.8, 0.1, 0.5], [0.1, 0.8, 0.5]])
        calculator = LAICalculations()

        self.assertAlmostEqual(calculator.polygon_area_3d(triangle), 0.245)
        with contextlib.redirect_stdout(io.StringIO()):
            lai = calculator.voxel_lai_from_facets(
                0, 1, 0, 1, 0, 1, triangle[np.newaxis, :, :]
            )
        self.assertAlmostEqual(lai, 0.245)

    def test_polygon_area_is_independent_of_position_and_vertex_order(self):
        """Measure a tilted rectangle after translation and reversed winding."""
        vertices = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 1], [0, 1, 1]], dtype=float)
        expected = 2 * np.sqrt(2)
        self.assertAlmostEqual(polygon_area_3d(vertices), expected)
        self.assertAlmostEqual(polygon_area_3d(vertices[::-1]), expected)
        self.assertAlmostEqual(polygon_area_3d(vertices + [10, -5, 3]), expected)

    def test_polygon_area_is_zero_for_collinear_vertices(self):
        """A degenerate polygon has no area."""
        vertices = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=float)
        self.assertEqual(polygon_area_3d(vertices), 0.0)

    def test_voxel_lai_adds_contained_facets_and_ignores_outside_facets(self):
        """Sum areas of contained triangles and divide by voxel ground area."""
        facets = np.array(
            [
                [[0.1, 0.1, 0.5], [1.1, 0.1, 0.5], [0.1, 1.1, 0.5]],
                [[0.2, 0.2, 0.5], [1.2, 0.2, 0.5], [0.2, 1.2, 0.5]],
                [[3.0, 3.0, 0.5], [4.0, 3.0, 0.5], [3.0, 4.0, 0.5]],
            ]
        )
        with contextlib.redirect_stdout(io.StringIO()):
            lai = LAICalculations().voxel_lai_from_facets(0, 2, 0, 2, 0, 1, facets)
        self.assertAlmostEqual(lai, 0.25)

    def test_voxel_lai_for_empty_facets(self):
        """An empty facet array contributes no leaf area."""
        with contextlib.redirect_stdout(io.StringIO()):
            lai = LAICalculations().voxel_lai_from_facets(
                0, 1, 0, 1, 0, 1, np.empty((0, 3, 3))
            )
        self.assertEqual(lai, 0.0)

    def test_voxel_normals_cover_all_six_faces(self):
        """Return a unit normal perpendicular to each face of a non-cubic voxel."""
        normals, points = LAICalculations.get_voxel_normals(0, 2, 0, 3, 0, 4)
        self.assertEqual(normals.shape, (6, 3))
        self.assertEqual(points.shape, (6, 4, 3))
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)
        np.testing.assert_allclose(
            np.abs(normals),
            [[0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]],
        )
        np.testing.assert_allclose(
            np.sum(normals * (points[:, 1] - points[:, 0]), axis=1), 0.0
        )
        np.testing.assert_allclose(
            np.sum(normals * (points[:, 2] - points[:, 0]), axis=1), 0.0
        )

    def test_voxel_lai_clips_facets_crossing_each_face(self):
        """A facet crossing any face should retain the same clipped area."""
        triangle = np.array([[0.25, 0.25, 0.5], [0.75, 0.25, 0.5], [0.25, 1.25, 0.5]])
        for axis in range(3):
            for upper in (False, True):
                with self.subTest(axis=axis, upper=upper):
                    if axis == 0:
                        facet = triangle[:, [1, 0, 2]]
                    elif axis == 2:
                        facet = triangle[:, [0, 2, 1]]
                    else:
                        facet = triangle.copy()
                    if not upper:
                        facet[:, axis] = 1 - facet[:, axis]
                    with contextlib.redirect_stdout(io.StringIO()):
                        lai = LAICalculations().voxel_lai_from_facets(
                            0, 1, 0, 1, 0, 1, facet[np.newaxis]
                        )
                    self.assertAlmostEqual(lai, 0.234375)

    def test_voxel_lai_includes_intersection_without_inside_vertices(self):
        """A triangle can cross the voxel even if all vertices lie outside it."""
        facet = np.array([[[-1.0, 0.5, 0.5], [2.0, 0.5, 0.5], [0.5, 2.0, 0.5]]])
        with contextlib.redirect_stdout(io.StringIO()):
            lai = LAICalculations().voxel_lai_from_facets(0, 1, 0, 1, 0, 1, facet)
        self.assertAlmostEqual(lai, 0.5)

    def test_voxel_clipping_handles_enclosing_and_nonintersecting_triangles(self):
        facets = np.array(
            [
                [[-10, -10, 0.5], [10, -10, 0.5], [0, 10, 0.5]],
                [[0.8, 1.3, 0.5], [1.3, 0.8, 0.5], [1.3, 1.3, 0.5]],
            ]
        )
        calculator = LAICalculations()
        for geometry in (
            facets,
            PreparedFacets(facets, method="scan"),
            PreparedFacets(facets, method="index"),
        ):
            with self.subTest(geometry=type(geometry)), contextlib.redirect_stdout(
                io.StringIO()
            ):
                lai = calculator.voxel_lai_from_facets(0, 1, 0, 1, 0, 1, geometry)
                self.assertAlmostEqual(lai, 1.0)

    def test_prepared_methods_agree_for_multiple_voxels_and_snapshot_input(self):
        facets = np.array(
            [
                [[-1, 0.5, 0.5], [2, 0.5, 0.5], [0.5, 2, 0.5]],
                [[3, 0.2, 0.5], [4, 0.2, 0.5], [3, 0.8, 0.5]],
                [[-100, -100, 0.5], [100, -100, 0.5], [0, 100, 0.5]],
            ]
        )
        scan = PreparedFacets(facets, method="scan")
        indexed = PreparedFacets(facets, method="index")
        facets[:] = 1000
        calculator = LAICalculations()
        for x in (-5, 0, 3, 50, 200):
            with self.subTest(x=x):
                lower = np.array([x, 0, 0])
                upper = lower + 1
                np.testing.assert_array_equal(
                    indexed.candidate_indices(lower, upper),
                    scan.candidate_indices(lower, upper),
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    expected = calculator.voxel_lai_from_facets(
                        x, x + 1, 0, 1, 0, 1, scan
                    )
                    actual = calculator.voxel_lai_from_facets(
                        x, x + 1, 0, 1, 0, 1, indexed
                    )
                self.assertAlmostEqual(actual, expected)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertAlmostEqual(
                calculator.voxel_lai_from_facets(0, 1, 0, 1, 0, 1, scan), 1.5
            )

    def test_default_index_is_built_once_and_reused_for_multiple_voxels(self):
        facets = np.array([[[0.1, 0.1, 0.5], [0.8, 0.1, 0.5], [0.1, 0.8, 0.5]]])
        calculator = LAICalculations()
        with mock.patch("scipy.spatial.cKDTree", wraps=cKDTree) as build_index:
            prepared = PreparedFacets(facets)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertAlmostEqual(
                    calculator.voxel_lai_from_facets(0, 1, 0, 1, 0, 1, prepared),
                    0.245,
                )
                self.assertEqual(
                    calculator.voxel_lai_from_facets(1, 2, 0, 1, 0, 1, prepared), 0
                )
            build_index.assert_called_once()

    def test_unprepared_facets_use_default_index(self):
        facets = np.array([[[0.1, 0.1, 0.5], [0.8, 0.1, 0.5], [0.1, 0.8, 0.5]]])
        with (
            mock.patch("scipy.spatial.cKDTree", wraps=cKDTree) as build_index,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            lai = LAICalculations().voxel_lai_from_facets(0, 1, 0, 1, 0, 1, facets)
        self.assertAlmostEqual(lai, 0.245)
        build_index.assert_called_once()

    def test_clips_enclosing_triangle_on_a_tilted_plane(self):
        facet = np.array([[[-10, -10, -10], [10, -10, 10], [0, 10, 0]]])
        calculator = LAICalculations()
        for method in ("scan", "index"):
            for vertices in (facet, facet[:, ::-1]):
                with self.subTest(method=method), contextlib.redirect_stdout(
                    io.StringIO()
                ):
                    lai = calculator.voxel_lai_from_facets(
                        0, 1, 0, 1, 0, 1, PreparedFacets(vertices, method)
                    )
                    self.assertAlmostEqual(lai, np.sqrt(2))

    def test_voxel_face_facets_belong_to_only_the_lower_inclusive_voxel(self):
        triangle = np.array([[[0.1, 0.1, 1], [0.8, 0.1, 1], [0.1, 0.8, 1]]])
        calculator = LAICalculations()
        for method in ("scan", "index"):
            geometry = PreparedFacets(triangle, method=method)
            with self.subTest(method=method), contextlib.redirect_stdout(io.StringIO()):
                below = calculator.voxel_lai_from_facets(0, 1, 0, 1, 0, 1, geometry)
                above = calculator.voxel_lai_from_facets(0, 1, 0, 1, 1, 2, geometry)
                self.assertEqual(below, 0)
                self.assertAlmostEqual(above, 0.245)

    def test_prepared_empty_facets_and_invalid_inputs(self):
        calculator = LAICalculations()
        for method in ("scan", "index"):
            geometry = PreparedFacets(np.empty((0, 3, 3)), method=method)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    calculator.voxel_lai_from_facets(0, 1, 0, 1, 0, 1, geometry), 0
                )
        with self.assertRaisesRegex(ValueError, "method"):
            PreparedFacets(np.empty((0, 3, 3)), method="invalid")
        with self.assertRaisesRegex(ValueError, "shape"):
            PreparedFacets(np.empty((0, 3)))
        with self.assertRaisesRegex(ValueError, "finite"):
            PreparedFacets(np.full((1, 3, 3), np.nan))
        with self.assertRaisesRegex(ValueError, "voxel maximum"):
            calculator.voxel_lai_from_facets(0, 0, 0, 1, 0, 1, np.empty((0, 3, 3)))

    def test_parallel_segment_returns_no_intersection_without_division_by_zero(self):
        with np.errstate(divide="raise", invalid="raise"):
            result = LAICalculations.intersect_segment_voxel_plane(
                np.array([0.0, 0.0, 0.0]),
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 0.5, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                0,
                1,
                0,
                1,
                0,
                1,
            )
        self.assertIsNone(result)

    def test_intersection_requires_segment_and_voxel_bounds(self):
        """Return the plane intersection only when both constraints are met."""
        calculator = LAICalculations()
        plane_point = np.array([0.5, 0.0, 0.0])
        normal = np.array([1.0, 0.0, 0.0])
        cases = [
            ([0.0, 0.25, 0.5], [1.0, 0.25, 0.5], [0.5, 0.25, 0.5]),
            ([0.0, 0.25, 0.5], [0.25, 0.25, 0.5], None),
            ([0.0, 1.5, 0.5], [1.0, 1.5, 0.5], None),
        ]
        for start, end, expected in cases:
            with self.subTest(start=start, end=end):
                result = calculator.intersect_segment_voxel_plane(
                    np.array(start),
                    np.array(end),
                    plane_point,
                    normal,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                )
                if expected is None:
                    self.assertIsNone(result)
                else:
                    np.testing.assert_allclose(result, expected)


if __name__ == "__main__":
    unittest.main()
