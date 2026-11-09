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
        facets : np.ndarray, float
                 n x m x 3 array of polygon vertices; this implementation assumes
                 triangular facets.

        Returns:
        --------
        voxel_lai : float
                    Leaf area inside the voxel divided by its horizontal ground area.
        """
        in_voxel = (
            (facets[:, :, 0] >= xmin)
            & (facets[:, :, 0] < xmax)
            & (facets[:, :, 1] >= ymin)
            & (facets[:, :, 1] < ymax)
            & (facets[:, :, 2] >= zmin)
            & (facets[:, :, 2] < zmax)
        )

        sum_in = np.sum(in_voxel, axis=1)
        rel_facets = facets[sum_in > 0, :, :]
        sum_inn = sum_in[sum_in > 0]

        a = time.time()
        areas_allin = np.sum(
            [self.polygon_area_3d(x) for x in rel_facets[sum_inn == 3]]
        )
        b = time.time()
        print(float(b - a) / 60.0, np.sum(sum_inn == 3))

        norms, pts = self.get_voxel_normals(
            xmin, xmax, ymin, ymax, zmin, zmax
        )

        c = time.time()
        new_areas = []
        for j, facet in enumerate(rel_facets[sum_inn < 3]):
            facetmin1 = np.roll(facet, -1, axis=0)
            ivj = in_voxel[sum_in > 0][sum_inn < 3][j]

            new_verts = []
            for k, vertex in enumerate(ivj):
                if vertex and np.roll(ivj, -1)[k]:
                    new_verts.append(facet[k])
                else:
                    if vertex:
                        new_verts.append(facet[k])
                    for i, norm in enumerate(norms):
                        ip = self.intersect_segment_voxel_plane(
                            facet[k],
                            facetmin1[k],
                            pts[i, :-1],
                            norm,
                            xmin,
                            xmax,
                            ymin,
                            ymax,
                            zmin,
                            zmax,
                        )
                        if ip is not None:
                            new_verts.append(ip)

            new_vert = np.array(new_verts)
            new_areas.append(self.polygon_area_3d(new_vert))

        d = time.time()
        print(float(d - c) / 60.0, np.sum(sum_inn < 3))
        total_la = np.sum(areas_allin) + np.sum(new_areas)
        ground_a = (xmax - xmin) * (ymax - ymin)
        return total_la / ground_a

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
                  Face normal vectors used by the clipping calculation.
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

        v1 = points[:-1, 1] - points[:-1, 0]
        v2 = points[:-1, 2] - points[:-1, 0]
        cp = np.cross(v1, v2)
        ncp = cp / np.linalg.norm(cp, axis=0)
        return ncp, points

    @staticmethod
    def intersect_segment_voxel_plane(
        A, B, P0, N, xmin, xmax, ymin, ymax, zmin, zmax
    ):
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
