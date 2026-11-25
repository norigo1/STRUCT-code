"""Geometric calculations used by voxel-based leaf-area analysis."""

import time

import numpy as np


def polygon_area_3d(vertices):
    """
    Calculate the area of a polygon from its 3D vertex coordinates.

    Parameters:
    -----------
    vertices : np.ndarray, float
               n x 3 array of polygon vertex coordinates in boundary order.

    Returns:
    --------
    area : float
           Polygon area in squared coordinate units.
    """
    v_next = np.roll(vertices, -1, axis=0)
    cross_products = np.cross(vertices, v_next)
    normal_vector = np.sum(cross_products, axis=0)
    return 0.5 * np.linalg.norm(normal_vector)


class PreparedFacets:
    """Cache triangle bounds, optionally with a spatial index, for voxel queries."""

    def __init__(self, facets, method="index"):
        """
        Prepare a snapshot of triangular facets for repeated voxel calculations.

        Parameters:
        -----------
        facets : np.ndarray
                 Triangle coordinates with shape (n, 3, 3).
        method : str, default="index"
                 "scan" checks all cached bounds; "index" builds a SciPy
                 spatial index, which can be reused across many voxels.
        """
        if method not in ("scan", "index"):
            raise ValueError("method must be 'scan' or 'index'")
        self.facets = np.array(facets, dtype=float, copy=True)
        if self.facets.ndim != 3 or self.facets.shape[1:] != (3, 3):
            raise ValueError("facets must have shape (n, 3, 3)")
        if not np.all(np.isfinite(self.facets)):
            raise ValueError("facets must contain finite coordinates")
        self.facets.setflags(write=False)
        self.minimum = self.facets.min(axis=1)
        self.maximum = self.facets.max(axis=1)
        self.minimum.setflags(write=False)
        self.maximum.setflags(write=False)
        self.tree = None
        if method == "index":
            from scipy.spatial import cKDTree

            centers = self.minimum + (self.maximum - self.minimum) / 2
            half_sizes = (self.maximum - self.minimum) / 2
            self.max_radius = np.max(np.linalg.norm(half_sizes, axis=1), initial=0.0)
            self.tree = cKDTree(centers)

    def candidate_indices(self, lower, upper):
        """Return triangles whose bounding boxes overlap the half-open voxel."""
        if self.tree is None:
            indices = np.arange(len(self.facets))
        else:
            center = lower + (upper - lower) / 2
            radius = np.linalg.norm((upper - lower) / 2) + self.max_radius
            indices = np.asarray(
                self.tree.query_ball_point(center, np.nextafter(radius, np.inf)),
                dtype=int,
            )
            indices.sort()
        overlap = np.all(
            (self.maximum[indices] >= lower) & (self.minimum[indices] < upper), axis=1
        )
        return indices[overlap]


class LAICalculations:
    """Calculate polygon areas and voxel-clipped leaf area index."""

    def voxel_lai_from_facets(self, xmin, xmax, ymin, ymax, zmin, zmax, facets):
        """
        Calculate the LAI contributed by facets intersecting a voxel.

        Parameters:
        -----------
        xmin, xmax : float
                     Voxel bounds along the x axis.
        ymin, ymax : float
                     Voxel bounds along the y axis.
        zmin, zmax : float
                     Voxel bounds along the z axis.
        facets : np.ndarray or PreparedFacets
                 Triangle coordinates with shape (n, 3, 3), or a prepared
                 snapshot reused across voxels. Passing an array builds a
                 new spatial index on each call.
                 Voxel minima are inclusive and maxima exclusive.

        Returns:
        --------
        voxel_lai : float
                    Leaf area inside the voxel divided by its horizontal ground area.
        """
        lower = np.array([xmin, ymin, zmin], dtype=float)
        upper = np.array([xmax, ymax, zmax], dtype=float)
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("voxel bounds must be finite")
        if np.any(lower >= upper):
            raise ValueError("each voxel maximum must exceed its minimum")
        prepared = (
            facets if isinstance(facets, PreparedFacets) else PreparedFacets(facets)
        )
        indices = prepared.candidate_indices(lower, upper)
        rel_facets = prepared.facets[indices]
        in_voxel = np.all((rel_facets >= lower) & (rel_facets < upper), axis=2)
        sum_inn = np.sum(in_voxel, axis=1)

        a = time.time()
        areas_allin = np.sum(
            [self.polygon_area_3d(x) for x in rel_facets[sum_inn == 3]]
        )
        b = time.time()
        print(float(b - a) / 60.0, np.sum(sum_inn == 3))

        c = time.time()
        new_areas = []
        for facet in rel_facets[sum_inn < 3]:
            clipped = self._clip_facet_to_voxel(facet, lower, upper)
            if len(clipped) >= 3:
                new_areas.append(self.polygon_area_3d(clipped))

        d = time.time()
        print(float(d - c) / 60.0, np.sum(sum_inn < 3))
        total_la = np.sum(areas_allin) + np.sum(new_areas)
        ground_a = (xmax - xmin) * (ymax - ymin)
        return total_la / ground_a

    @staticmethod
    def _clip_facet_to_voxel(facet, lower, upper):
        """Clip an ordered polygon against the six axis-aligned voxel planes."""
        vertices = list(facet)
        for axis in range(3):
            for bound, keep_above in ((lower[axis], True), (upper[axis], False)):
                if not vertices:
                    return np.empty((0, 3))
                clipped = []
                previous = vertices[-1]
                previous_inside = (
                    previous[axis] >= bound if keep_above else previous[axis] <= bound
                )
                for current in vertices:
                    current_inside = (
                        current[axis] >= bound
                        if keep_above
                        else current[axis] <= bound
                    )
                    if previous_inside != current_inside:
                        fraction = (bound - previous[axis]) / (
                            current[axis] - previous[axis]
                        )
                        clipped.append(previous + fraction * (current - previous))
                    if current_inside:
                        clipped.append(current)
                    previous = current
                    previous_inside = current_inside
                vertices = clipped
        return np.asarray(vertices).reshape(-1, 3)

    @staticmethod
    def get_voxel_normals(xmin, xmax, ymin, ymax, zmin, zmax):
        """
        Return normals and corner coordinates for voxel faces.

        Parameters:
        -----------
        xmin, xmax : float
                     Voxel bounds along the x axis.
        ymin, ymax : float
                     Voxel bounds along the y axis.
        zmin, zmax : float
                     Voxel bounds along the z axis.

        Returns:
        --------
        normals : np.ndarray, float
                  Unit normal for each face, shape (6, 3). The direction follows
                  the vertex ordering and is not necessarily outward.
        points : np.ndarray, float
                 Corner coordinates for the six voxel faces.
        """
        points = np.array(
            [
                [
                    [xmin, ymin, zmin],
                    [xmax, ymin, zmin],
                    [xmax, ymax, zmin],
                    [xmin, ymax, zmin],
                ],
                [
                    [xmin, ymin, zmax],
                    [xmax, ymin, zmax],
                    [xmax, ymax, zmax],
                    [xmin, ymax, zmax],
                ],
                [
                    [xmin, ymin, zmin],
                    [xmin, ymin, zmax],
                    [xmin, ymax, zmax],
                    [xmin, ymax, zmin],
                ],
                [
                    [xmax, ymin, zmin],
                    [xmax, ymin, zmax],
                    [xmax, ymax, zmax],
                    [xmax, ymax, zmin],
                ],
                [
                    [xmin, ymin, zmin],
                    [xmax, ymin, zmin],
                    [xmax, ymin, zmax],
                    [xmin, ymin, zmax],
                ],
                [
                    [xmin, ymax, zmin],
                    [xmax, ymax, zmin],
                    [xmax, ymax, zmax],
                    [xmin, ymax, zmax],
                ],
            ]
        )

        v1 = points[:, 1] - points[:, 0]
        v2 = points[:, 2] - points[:, 0]
        cp = np.cross(v1, v2)
        ncp = cp / np.linalg.norm(cp, axis=1, keepdims=True)
        return ncp, points

    @staticmethod
    def intersect_segment_voxel_plane(A, B, P0, N, xmin, xmax, ymin, ymax, zmin, zmax):
        """
        Find the intersection of segment AB with a voxel face plane.

        Parameters:
        -----------
        A, B : np.ndarray, float
               Endpoints of the segment.
        P0 : np.ndarray, float
             A point on the plane.
        N : np.ndarray, float
            Normal vector of the plane.
        xmin, xmax, ymin, ymax, zmin, zmax : float
            Bounds of the voxel.

        Returns:
        --------
        point : np.ndarray, float or None
                Intersection point if it lies on the segment and within the
                voxel bounds; otherwise None.
        """
        V = B - A
        denominator = np.dot(V, N)
        if denominator == 0:
            return None
        t = np.dot(P0 - A, N) / denominator
        point = A + t * V
        if (
            (point[0] >= xmin)
            & (point[0] <= xmax)
            & (point[1] >= ymin)
            & (point[1] <= ymax)
            & (point[2] >= zmin)
            & (point[2] <= zmax)
            & np.all(0.0 <= t)
            & np.all(t <= 1.0)
        ):
            return point
        return None

    @staticmethod
    def polygon_area_3d(vertices):
        """
        Calculate the area of a polygon from its 3D vertex coordinates.

        Parameters:
        -----------
        vertices : np.ndarray, float
                   n x 3 array of polygon vertex coordinates in boundary order.

        Returns:
        --------
        area : float
               Polygon area in squared coordinate units.
        """
        return polygon_area_3d(vertices)
