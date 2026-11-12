#######################################################################
#
# Code to alter the canopy cover and leaf area index of a scene, 
# by changing the scene's vegetation parameters.
#
# Written by Niall Origo, 02/10/2026
#
######################################################################

"""Measure and modify canopy cover and leaf/plant area indices in scenes."""

import matplotlib.pyplot as plt
import numpy as np
import shapely
from xml.etree import ElementTree as ET
from scipy.spatial import ConvexHull, Delaunay
from scipy import interpolate as interp
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union
import pathlib
import pdb
import os
import tempfile
import time

from geometrical_calculations import polygon_area_3d
from scene_io import SceneReader


class GapFraction:
    """Calculate projected vegetation gap fractions for a scene."""

    def __init__(self, scene):
        """
        Initialize gap-fraction calculations for a scene.

        Parameters:
        -----------
        scene : str or pathlib.Path
                Path to the scene XML file.
        """
        self.scene = scene
        sr = SceneReader(scene)
        self.filepaths = sr.read_scene()

    def reproject_vertices(self, vertices, point, normal):
        """
        Reproject vertices onto a plane defined by a point and normal vector.

        Parameters:
        -----------
        vertices : np.ndarray, float
                   n x 3 array of vertex coordinates.
        point : np.ndarray, float
                A point on the projection plane.
        normal : np.ndarray, float
                 Normal vector of the projection plane.

        Returns:
        --------
        projected_vertices : np.ndarray, float
                             Projected coordinates with the same shape as vertices.
        """
        normal = normal / np.linalg.norm(normal)  # normalize the normal vector
        d = -np.dot(normal, point)
        distances = np.dot(vertices, normal) + d
        projected_vertices = vertices - np.outer(distances, normal)
        return projected_vertices

    def calculate_gap_fraction(self, vertices, faces, zenith, azimuth, alpha_radius=0.2):
        """
        Calculate the gap fraction of a vegetation object given its vertices and faces.
        The gap fraction is calculated by projecting the vertices onto a plane defined
        by the zenith and azimuth angles, calculating the crown area using the alpha shape,
        along with the crown area considering only the reprojected triangles. The latter is 
        divided by the former to retrieve the gap fraction.

        Parameters:
        -----------
        vertices : array-like, float
                   Mesh vertex coordinates.
        faces : array-like, int
                Triangular faces with one-based vertex indices.
        zenith : float
                 Viewing zenith angle in degrees.
        azimuth : float
                  Viewing azimuth angle in degrees.
        alpha_radius : float, default=0.2
                       Maximum circumradius used by the alpha-shape outline.

        Returns:
        --------
        gap_fraction : float
                       Fraction of the filled crown area not covered by projected
                       vegetation faces.
        vegetation_area : float
                          Area covered by the projected vegetation faces.
        crown_area : float
                     Crown area enclosed by the filled alpha shape.
        """
        # Convert angles to radians
        zenith_rad = np.radians(zenith)
        azimuth_rad = np.radians(azimuth)

        # Calculate normal vector from zenith and azimuth angles
        normal = np.array([
            np.sin(zenith_rad) * np.cos(azimuth_rad),
            np.sin(zenith_rad) * np.sin(azimuth_rad),
            np.cos(zenith_rad)
        ])

        # Reproject vertices onto plane defined by normal vector
        projected_vertices = self.reproject_vertices(vertices, np.mean(vertices, axis=0), normal)

        # Calculate alpha shape of projected points
        cc = CanopyCover(self.scene)

        # Calculate area of alpha shape and bounding box
        fg_area = cc.calculate_crown_area_fill_gaps(projected_vertices, 
                                                    faces, 
                                                    alpha_radius=alpha_radius)[0]
        wg_area = cc.calculate_crown_area_with_gaps(projected_vertices, faces)[0]

        # Calculate gap fraction
        gap_fraction = 1 - (wg_area / fg_area) if fg_area > 0 else 0.0

        return gap_fraction, wg_area, fg_area


# class to calculate canopy cover
class CanopyCover:
    """Calculate crown areas and canopy cover from scene vegetation meshes."""

    def __init__(self, scene):
        """
        Initialize canopy-cover calculations for a scene.

        Parameters:
        -----------
        scene : str or pathlib.Path
                Path to the scene XML file.
        """
        self.scene = scene
        self.sr = SceneReader(scene)
        self.filepaths = self.sr.read_scene()

    def calculate_canopy_cover(self, directories, globstring, iterable, ga="max_extent"):
        """
        Calculate the canopy cover of the scene by reading in the vertices and faces 
        of the scene's vegetation objects. Works for wood files where the stem is made
        of facets. Will need a new method if the stem is made of cylinders.

        Parameters:
        -----------
        directories : iterable of str
                      Directories containing tree mesh files.
        globstring : str
                     Mesh glob pattern containing the ``{iter}`` placeholder.
        iterable : iterable
                   Values substituted into ``globstring`` to select trees.
        ga : str, default='max_extent'
             Ground-area method: bounding extent, convex hull, or canopy outline.

        Returns:
        --------
        canopy_area : float
                      Total union area of the crowns.
        ground_area : float
                      Area used as the ground-area denominator.
        canopy_cover : float
                       Canopy area divided by ground area.
        crown_outlines : list of np.ndarray
                         XY outline coordinates for each selected crown.
        ground_outline : np.ndarray
                         XY coordinates of the ground-area boundary.
        """
        crown_outlines = []
        for i in iterable:
            print(f"Processing tree {i/len(iterable)*100:.2f}%...")#, end='\r')
            for directory in directories:
                if directory[-1] != '/':
                    directory += '/'

            vertices, faces = self.sr.read_tree(directories, globstring.format(iter=i))
            ca, co = self.calculate_crown_area_fill_gaps(vertices, faces)
            crown_outlines.append(co)

        multipolygon = False
        if len(crown_outlines) > 0:
            canopy_outline = shapely.union_all([Polygon(c) for c in crown_outlines])
            if canopy_outline.geom_type == 'Polygon':
                # area accounting for the holes in the interiror of the canopy
                ca_total = canopy_outline.area
                # outline of the whole canopy, ignoring the holes in the interior
                ol = np.array(canopy_outline.exterior.coords)
            elif canopy_outline.geom_type == 'MultiPolygon':
                ol = []
                clump_areas = []
                for x in canopy_outline.geoms:
                    olx = np.array(x.exterior.coords)
                    ol.append(olx)
                    clump_areas.append(x.area)
                ca_total = np.sum(clump_areas)
                ol = np.vstack(ol)
                multipolygon = True
            else:
                ca_total = 0.0
        else:
            ca_total = 0.0

        if ga == "max_extent":
            minx, miny, maxx, maxy = canopy_outline.bounds
            canopy_ga = (maxx - minx) * (maxy - miny)
            ol = np.array([[minx, miny], 
                           [minx, maxy], 
                           [maxx, maxy], 
                           [maxx, miny], 
                           [minx, miny], 
                           [minx, miny]])
        elif multipolygon and ga == "can_outline":
            print("Warning: Canopy outline is a MultiPolygon. Using convexhull instead.")
            olpg = Polygon(ol).convex_hull
            canopy_ga = olpg.area
            ol = np.array(olpg.exterior.coords)
        elif ga == "convexhull":
            olpg = Polygon(ol).convex_hull
            canopy_ga = olpg.area
            ol = np.array(olpg.exterior.coords)
        elif ga == "can_outline":
            canopy_ga = Polygon(ol).area
        else:
            raise ValueError(f"Invalid ground area calculation method: {ga}. "
                              "Choose from 'max_extent', 'convexhull', or 'can_outline'.")
        
        return (ca_total, 
                canopy_ga, 
                ca_total / canopy_ga, 
                crown_outlines, 
                ol)

    def calculate_canopy_cover_with_gaps(self, 
                                         directories, 
                                         globstring, 
                                         iterable, 
                                         ga="max_extent", 
                                         alpha_radius=0.2,
                                         canopy_cover_outline=None):
        """
        Calculate the canopy cover of the scene by reading in the vertices and faces 
        of the scene's vegetation objects. Works for wood files where the stem is made
        of facets. Will need a new method if the stem is made of cylinders.

        Parameters:
        -----------
        directories : iterable of str
                      Directories containing tree mesh files.
        globstring : str
                     Mesh glob pattern containing the ``{iter}`` placeholder.
        iterable : iterable
                   Values substituted into ``globstring`` to select trees.
        ga : str, default='max_extent'
             Ground-area method reserved for the planned implementation.
        alpha_radius : float, default=0.2
                       Alpha-shape radius reserved for the planned implementation.
        canopy_cover_outline : iterable of tuple or None, default=None
                               Optional outline reserved for the planned
                               implementation.

        Returns:
        --------
        None
            This method is currently a placeholder and does not calculate a value.
        """
        something = 0

    def calculate_alpha_shape(self, points, alpha_radius=0.2):
        """
        Calculate the alpha shape (concave hull) of a set of XY points.

        Parameters:
        -----------
        points : array-like, float
                 Point coordinates; only the first two columns are used.
        alpha_radius : float, default=0.2
                       Maximum circumradius of Delaunay triangles retained in
                       the alpha shape.

        Returns:
        --------
        shape : shapely geometry
                Union of retained triangles, or the convex hull for fewer than
                four input points.
        """
        points = np.asarray(points)[:, :2]
        if len(points) < 4:
            return Polygon(points).convex_hull

        tri = Delaunay(points)
        a = points[tri.simplices[:, 0]]
        b = points[tri.simplices[:, 1]]
        c = points[tri.simplices[:, 2]]
        la = np.linalg.norm(b - c, axis=1)
        lb = np.linalg.norm(a - c, axis=1)
        lc = np.linalg.norm(a - b, axis=1)
        twice_area = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                            - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
        with np.errstate(divide='ignore', invalid='ignore'):
            circumradius = la * lb * lc / (2.0 * twice_area)
        keep = np.isfinite(circumradius) & (circumradius <= alpha_radius)
        triangle_coords = points[tri.simplices[keep]]
        triangle_rings = shapely.linearrings(triangle_coords)
        shape = shapely.union_all(shapely.polygons(triangle_rings))
        return shape if shape.is_valid else shape.buffer(0)

    def calculate_crown_area_fill_gaps(self, vertices, faces, alpha_radius=None):
        """
        Calculate the crown area including gaps using an alpha shape.

        The leaf vertices are projected onto the XY plane and Delaunay triangulated.
        Triangles whose circumradius exceeds alpha_radius (i.e. alpha = 1 / alpha_radius)
        are discarded, the remaining triangles are merged and interior holes are filled,
        so gaps between leaves are closed while concavities in the outline are kept.

        Parameters:
        -----------
        vertices : array-like, float
                   Mesh vertex coordinates.
        faces : array-like, int
                Triangular faces with one-based vertex indices.
        alpha_radius : float or None, default=None
                       Maximum circumradius of retained Delaunay triangles.
                       The default is based on the mesh's XY extent.

        Returns:
        --------
        crown_area : float
                     Area enclosed by the alpha shape with holes filled.
        contour : np.ndarray, float
                  XY coordinates of the largest component's exterior.
        """
        vertices = np.asarray(vertices)[:, :2]
        faces = np.asarray(faces, dtype=int)
        if len(faces) == 0:
            return 0.0, np.empty((0, 2))

        points = np.unique(vertices[np.unique(faces[:, :3]) - 1], axis=0)
        if len(points) < 3:
            return 0.0, np.empty((0, 2))

        extent = np.max(points.max(axis=0) - points.min(axis=0))
        if extent == 0:
            return 0.0, np.empty((0, 2))
        if alpha_radius is None:
            alpha_radius = 0.2 * extent

        try:
            shape = self.calculate_alpha_shape(points, alpha_radius)
        except Exception:
            return 0.0, np.empty((0, 2))

        polygons = list(shape.geoms) if hasattr(shape, 'geoms') else [shape]
        polygons = [p for p in polygons if p.geom_type == 'Polygon' and not p.is_empty]
        if not polygons:
            return 0.0, np.empty((0, 2))

        # fill holes: area of the exterior rings only
        crown_area = sum(Polygon(p.exterior).area for p in polygons)
        largest = max(polygons, key=lambda p: Polygon(p.exterior).area)
        contour = np.asarray(largest.exterior.coords)#[:-1]
        return crown_area, contour

    def calculate_crown_area_with_gaps(self, vertices, faces):
        """
        Calculate the crown area of a vegetation object given its vertices and faces.

        Parameters:
        -----------
        vertices : array-like, float
                   Mesh vertex coordinates.
        faces : array-like, int
                Triangular faces with one-based vertex indices.

        Returns:
        --------
        crown_area : float
                     Area of the union of projected face polygons.
        union_polygon : shapely geometry
                        Union of the projected vegetation faces.
        """
        leaf_polygons = []
        vertices = np.array(vertices)[:,:2]
        for face in faces:
            v1 = vertices[face[0] - 1]
            v2 = vertices[face[1] - 1]
            v3 = vertices[face[2] - 1]
            leaf_polygons.append(Polygon([v1, v2, v3, v1]))
        union_polygon = unary_union(leaf_polygons)
        crown_area = union_polygon.area
        return crown_area, union_polygon

    def line_segment_intersection(self, line1, seg1, seg2):
        """
        Calculate the intersection of an infinite line and a line segment.

        Parameters:
        -----------
        line1 : array-like, float
                Coefficients ``(a, b, c)`` of the line ``a*x + b*y = c``.
        seg1, seg2 : array-like, float
                     XY coordinates of the segment endpoints.

        Returns:
        --------
        intersection : tuple of float or None
                       XY intersection if it lies on the segment; otherwise None.
        """
        a, b, c = line1[:3]
        x1, y1 = seg1
        x2, y2 = seg2
        dx = x2 - x1
        dy = y2 - y1
        denom = a * dx + b * dy

        if np.isclose(denom, 0.0):
            if np.isclose(a * x1 + b * y1, c) and np.isclose(a * x2 + b * y2, c):
                return (x1, y1)
            return None

        t = (c - a * x1 - b * y1) / denom
        if t < 0.0 or t > 1.0:
            return None
        return (x1 + t * dx, y1 + t * dy)

    def rotational_sort(self, points):
        """
        Order points greedily by angular progression and distance.

        Parameters:
        -----------
        points : array-like, float
                 XY point coordinates to order.

        Returns:
        --------
        ordered_points : np.ndarray, float
                         Input points reordered by the greedy traversal.
        """
        points = np.asarray(points)
        if len(points) < 2:
            return points

        centroid = np.mean(points, axis=0)
        offsets = points - centroid
        angles = np.arctan2(offsets[:, 1], offsets[:, 0])
        order = [int(np.argmin(angles))]
        remaining = set(range(len(points)))
        remaining.remove(order[0])

        while remaining:
            current = order[-1]
            candidates = np.array(sorted(remaining), dtype=int)
            distances = np.linalg.norm(points[candidates] - points[current], axis=1)
            nearest = np.argsort(distances)[:10]
            candidates = candidates[nearest]
            distances = distances[nearest]
            angular_gaps = (angles[candidates] - angles[current]) % (2.0 * np.pi)
            max_distance = np.max(distances)
            normalized_distances = distances / max_distance if max_distance > 0 else distances
            scores = 0.25 * angular_gaps / (2.0 * np.pi) + 0.75 * normalized_distances
            next_point = int(candidates[np.argmin(scores)])
            order.append(next_point)
            remaining.remove(next_point)
        return points[order]


# class to calculate leaf area index
class CalculateVegAreaIndex:
    """Calculate scene-wide leaf and plant area from vegetation meshes."""

    def __init__(self, scene, leaf_id="leaves", terrain_id="terrain", wood_id="wood"):
        """
        Initialize area-index calculations for a scene.

        Parameters:
        -----------
        scene : str or pathlib.Path
                Path to the scene XML file.
        leaf_id : str, default='leaves'
                  Identifier used to recognize leaf mesh paths.
        terrain_id : str, default='terrain'
                     Identifier used to recognize terrain paths.
        wood_id : str, default='wood'
                  Identifier used to recognize wood mesh paths.
        """
        self.scene = scene
        sr = SceneReader(scene)
        self.filepaths = sr.read_scene()
        self.leaf_id = leaf_id
        self.terrain_id = terrain_id
        self.wood_id = wood_id
        self.mesh_summary_path = pathlib.Path(scene).resolve().with_name(
            pathlib.Path(scene).name + ".mesh_summary.npy"
        )

    def _read_mesh_summary(self):
        """Read the scene's per-mesh summary without loading Python objects."""
        try:
            data = np.load(self.mesh_summary_path, allow_pickle=False)
        except FileNotFoundError:
            return {}
        except (ValueError, EOFError) as exc:
            raise ValueError(f"Invalid mesh summary: {self.mesh_summary_path}") from exc
        if not isinstance(data, np.ndarray):
            data.close()
            raise ValueError(f"Invalid mesh summary format: {self.mesh_summary_path}")
        fields = ("path", "size", "mtime_ns", "ctime_ns", "has_faces",
                  "minimum", "maximum", "area")
        if (data.ndim != 1
            or data.dtype.names != fields
            or data.dtype["path"].kind != "U"
            or any(data.dtype[name] != np.dtype("int64")
                   for name in ("size", "mtime_ns", "ctime_ns"))
            or data.dtype["has_faces"] != np.dtype("bool")
            or any(data.dtype[name].subdtype != (np.dtype("float64"), (3,))
                   for name in ("minimum", "maximum"))
            or data.dtype["area"] != np.dtype("float64")):
            raise ValueError(f"Invalid mesh summary format: {self.mesh_summary_path}")
        if (len(np.unique(data["path"])) != len(data)
            or np.any(data["path"] == "")
            or np.any(data["size"] < 0)
            or not np.isfinite(data["minimum"]).all()
            or not np.isfinite(data["maximum"]).all()
            or np.any(data["minimum"] > data["maximum"])
            or not np.isfinite(data["area"]).all()
            or np.any(data["area"] < 0)):
            raise ValueError(f"Invalid mesh summary values: {self.mesh_summary_path}")
        return {
            str(record["path"]): (
                int(record["size"]), int(record["mtime_ns"]), int(record["ctime_ns"]),
                bool(record["has_faces"]), record["minimum"].copy(),
                record["maximum"].copy(), float(record["area"]),
            )
            for record in data
        }

    def _write_mesh_summary(self, summary):
        """Atomically save per-mesh extents and total areas beside the scene."""
        width = max((len(path) for path in summary), default=1)
        dtype = np.dtype([
            ("path", f"U{width}"), ("size", "i8"), ("mtime_ns", "i8"),
            ("ctime_ns", "i8"), ("has_faces", "?"), ("minimum", "f8", (3,)),
            ("maximum", "f8", (3,)), ("area", "f8"),
        ])
        records = np.empty(len(summary), dtype=dtype)
        for index, (path, values) in enumerate(summary.items()):
            records[index] = (path, *values)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{self.mesh_summary_path.name}.",
                suffix=".tmp", dir=self.mesh_summary_path.parent, delete=False,
            ) as file:
                temporary_path = pathlib.Path(file.name)
                np.save(file, records, allow_pickle=False)
            os.replace(temporary_path, self.mesh_summary_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _read_mesh_geometry(reader, path, suffix, fingerprint):
        """Read one mesh and verify that it did not change during the read."""
        if suffix == ".obj":
            vertices, faces = reader.read_obj(path)
        else:
            vertices, faces = reader.read_ply(path)
        stat = path.stat()
        if fingerprint != (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns):
            raise RuntimeError(f"Mesh changed while reading: {path}")
        vertices = np.asarray(vertices, dtype=float)
        if not len(vertices) or not len(faces):
            return vertices, np.empty((0, 3, 3)), np.empty(0)
        triangles = vertices[np.array([face[:3] for face in faces], dtype=int) - 1]
        areas = 0.5 * np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0],
                     triangles[:, 2] - triangles[:, 0]), axis=1,
        )
        return vertices, triangles, areas

    def calculate_area_index(self, xmin, xmax, ymin, ymax, zmin=-np.inf, zmax=np.inf,
                             type="leaf", *, method="centroid"):
        """
        Calculate vegetation surface area in one or more boxes.

        A small summary named <scene filename>.mesh_summary.npy is maintained
        beside the scene. It stores each processed mesh's path, file metadata,
        extent, and total face area, but no per-face data. Missing or changed
        meshes are read to update it; subsequent calls can skip meshes outside
        all boxes or use the total area when a box fully contains a mesh.
        Partially overlapping meshes are read once and evaluated for all boxes.
        Leaf and wood summaries are populated when each type is requested.
        For LAI or PAI divide the returned area by the ground area.

        Parameters:
        -----------
        xmin, xmax, ymin, ymax : float or one-dimensional array
                                 Paired XY bounds for each box. Scalars and
                                 length-one arrays broadcast to all boxes.
        zmin, zmax : float or one-dimensional array, default=-inf, +inf
                     Paired height bounds. Set only zmin to specify a minimum
                     height without an upper limit.
        type : str, default='leaf'
               Mesh category to measure: 'leaf' or 'wood'.
        method : {'centroid', 'clip'}, default='centroid'
                 Assign whole faces by centroid or clip faces to the box.
                 Boxes are lower-inclusive and upper-exclusive. Clipping is
                 exact apart from floating-point error; faces lying exactly
                 on an upper boundary are assigned to the adjacent box.

        Raises:
        -------
        ValueError
            If bounds, method, type, a mesh format, or the summary are invalid.
        OSError
            If a mesh or the summary cannot be read or written.
        RuntimeError
            If a mesh changes while it is being read.

        Returns:
        --------
        total_area : float or np.ndarray
                     Surface area for each box, not divided by ground area.
        ground_area : float or np.ndarray
                      XY ground area for each box.
        area_by_mesh : np.ndarray, float
                       Per-mesh contributions in scene order, shape (meshes,)
                       for scalar bounds or (boxes, meshes) for array bounds.
                       Only meshes whose bounds overlap at least one box are
                       included; zero means that mesh contributes no area.
        """
        if method not in ("centroid", "clip"):
            raise ValueError("method must be 'centroid' or 'clip'.")
        inputs = [np.asarray(value, dtype=float)
                  for value in (xmin, xmax, ymin, ymax, zmin, zmax)]
        if any(value.ndim > 1 for value in inputs):
            raise ValueError("Bounds must be scalars or one-dimensional arrays.")
        scalar = all(value.ndim == 0 for value in inputs)
        try:
            bounds = np.broadcast_arrays(*inputs)
        except ValueError as exc:
            raise ValueError("Bounds arrays must have matching lengths or length one.") from exc
        bounds = [np.atleast_1d(value) for value in bounds]
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        if (not len(xmin) or not all(np.isfinite(value).all() for value in bounds[:4])
            or np.isnan(zmin).any() or np.isnan(zmax).any()
            or np.any(xmin >= xmax) or np.any(ymin >= ymax)
            or np.any(zmin >= zmax)):
            raise ValueError(
                "Bounds must be ordered (xmin < xmax, ymin < ymax, zmin < zmax); "
                "XY bounds must be finite and Z bounds must not be NaN."
            )
        nboxes = len(xmin)
        lows = np.column_stack((xmin, ymin, zmin))
        highs = np.column_stack((xmax, ymax, zmax))
        per_mesh = []
        if type == "leaf":
            type_id = self.leaf_id
        elif type == "wood":
            type_id = self.wood_id
        else:
            raise ValueError(f"Invalid type: {type}. Choose from 'leaf' or 'wood'.")
        reader = SceneReader(self.scene)
        summary = self._read_mesh_summary()
        summary_changed = not self.mesh_summary_path.exists()
        for filepath in self.filepaths:
            if type_id not in filepath:
                continue
            suffix = pathlib.Path(filepath).suffix.lower()
            if suffix not in (".obj", ".ply"):
                raise ValueError(f"Unsupported file format: {filepath}")
            resolved_filepath = pathlib.Path(reader.resolve_filepath(filepath)).resolve()
            key = str(resolved_filepath)
            stat = resolved_filepath.stat()
            fingerprint = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            entry = summary.get(key)
            triangles = None
            areas = None
            if entry is None or entry[:3] != fingerprint:
                vertices, triangles, areas = self._read_mesh_geometry(
                    reader, resolved_filepath, suffix, fingerprint
                )
                has_faces = bool(len(triangles))
                if has_faces:
                    mesh_min, mesh_max = vertices.min(axis=0), vertices.max(axis=0)
                    total = float(np.sum(areas))
                    if not np.isfinite(mesh_min).all() or not np.isfinite(mesh_max).all() or not np.isfinite(total):
                        raise ValueError(f"Mesh contains non-finite geometry: {resolved_filepath}")
                else:
                    mesh_min = mesh_max = np.zeros(3)
                    total = 0.0
                entry = (*fingerprint, has_faces, mesh_min, mesh_max, total)
                summary[key] = entry
                summary_changed = True
            _, _, _, has_faces, mesh_min, mesh_max, total = entry
            if not has_faces:
                continue
            overlaps = np.all((mesh_max >= lows) & (mesh_min < highs), axis=1)
            if not np.any(overlaps):
                continue
            fully_contained = overlaps & np.all((mesh_min >= lows) & (mesh_max < highs), axis=1)
            mesh_area = np.zeros(nboxes)
            mesh_area[fully_contained] = total
            partial = overlaps & ~fully_contained
            if np.any(partial) and triangles is None:
                _, triangles, areas = self._read_mesh_geometry(
                    reader, resolved_filepath, suffix, fingerprint
                )
            if np.any(partial):
                if method == "centroid":
                    centroids = np.mean(triangles, axis=1)
                else:
                    face_min = np.min(triangles, axis=1)
                    face_max = np.max(triangles, axis=1)
            for index in np.flatnonzero(partial):
                low, high = lows[index], highs[index]
                if method == "centroid":
                    selected = np.all((centroids >= low) & (centroids < high), axis=1)
                    mesh_area[index] = np.sum(areas[selected])
                else:
                    candidates = np.all((face_max >= low) & (face_min < high), axis=1)
                    contained = candidates & np.all((face_min >= low) & (face_max <= high), axis=1)
                    mesh_area[index] = np.sum(areas[contained])
                    for triangle in triangles[candidates & ~contained]:
                        mesh_area[index] += self._clipped_triangle_area(triangle, low, high)
            per_mesh.append(mesh_area)

        if summary_changed:
            self._write_mesh_summary(summary)
        area_by_mesh = np.stack(per_mesh, axis=1) if per_mesh else np.zeros((nboxes, 0))
        total_area = np.sum(area_by_mesh, axis=1)
        ground_area = (xmax - xmin) * (ymax - ymin)
        if scalar:
            return total_area[0], ground_area[0], area_by_mesh[0]
        return total_area, ground_area, area_by_mesh

    @staticmethod
    def _clipped_triangle_area(triangle, low, high):
        """Clip a triangle to an axis-aligned box and return its 3D area."""
        polygon = list(triangle)
        for axis in range(3):
            for bound, lower in ((low[axis], True), (high[axis], False)):
                if not np.isfinite(bound):
                    continue
                if not polygon:
                    return 0.0
                clipped = []
                previous = polygon[-1]
                previous_inside = previous[axis] >= bound if lower else previous[axis] <= bound
                for current in polygon:
                    current_inside = current[axis] >= bound if lower else current[axis] <= bound
                    if current_inside != previous_inside:
                        fraction = (bound - previous[axis]) / (current[axis] - previous[axis])
                        clipped.append(previous + fraction * (current - previous))
                    if current_inside:
                        clipped.append(current)
                    previous, previous_inside = current, current_inside
                polygon = clipped
        if len(polygon) < 3:
            return 0.0
        polygon = np.asarray(polygon)
        edges = polygon[1:] - polygon[0]
        return float(0.5 * np.linalg.norm(np.cross(edges[:-1], edges[1:]).sum(axis=0)))

    def calculate_leaf_area_index(self, xmin, xmax, ymin, ymax, zmin=-np.inf, zmax=np.inf,
                                  *, method="centroid"):
        """
        Calculate the leaf area index (LAI) of the scene.

        Parameters:
        -----------
        xmin, xmax, ymin, ymax : float or one-dimensional array
                                 Paired XY bounds passed to calculate_area_index.
        zmin, zmax : float or one-dimensional array, default=-inf, +inf
                     Paired height bounds passed to calculate_area_index.
        method : {'centroid', 'clip'}, default='centroid'
                 How faces are assigned to each box.

        Returns:
        --------
        total_leaf_area : float or np.ndarray
                          Leaf area in each box.
        ground_area : float or np.ndarray
                      XY ground area of each box.
        leaf_area : np.ndarray, float
                    Per-mesh contributions; see calculate_area_index for shape.
        """
        total_leaf_area, ground_area, leaf_area = self.calculate_area_index(
            xmin, xmax, ymin, ymax, zmin=zmin, zmax=zmax, type="leaf", method=method
        )
        return total_leaf_area, ground_area, leaf_area

    def calculate_plant_area_index(self, xmin, xmax, ymin, ymax, zmin=-np.inf, zmax=np.inf,
                                   *, method="centroid"):
        """
        Calculate the plant area index (PAI) of the scene.

        Parameters:
        -----------
        xmin, xmax, ymin, ymax : float or one-dimensional array
                                 Paired XY bounds passed to calculate_area_index.
        zmin, zmax : float or one-dimensional array, default=-inf, +inf
                     Paired height bounds passed to calculate_area_index.
        method : {'centroid', 'clip'}, default='centroid'
                 How faces are assigned to each box.

        Returns:
        --------
        total_plant_area : float or np.ndarray
                           Leaf and wood area in each box.
        ground_area : float or np.ndarray
                      XY ground area of each box.
        plant_area : np.ndarray, float
                     Leaf and wood contributions; see calculate_area_index
                     for shape. Leaf meshes precede wood meshes.
        """
        total_leaf_area, ground_area, leaf_area = self.calculate_area_index(
            xmin, xmax, ymin, ymax, zmin=zmin, zmax=zmax, type="leaf", method=method
        )
        total_wood_area, _, wood_area = self.calculate_area_index(
            xmin, xmax, ymin, ymax, zmin=zmin, zmax=zmax, type="wood", method=method
        )
        total_plant_area = total_leaf_area + total_wood_area
        # Leaf and wood object counts need not match; concatenate their areas rather
        # than requiring the arrays to have the same length.
        plant_area = np.concatenate((leaf_area, wood_area), axis=-1)
        return total_plant_area, ground_area, np.array(plant_area)

    def calculate_leaf_area_index_profile(
        self, xmin, xmax, ymin, ymax, min_height=0.0, height_bins=10,
        height_range=None,
    ):
        """
        Calculate the LAI contribution in equally spaced height bands.

        Parameters:
        -----------
        xmin, xmax, ymin, ymax : float
                                 XY bounds used for the ground-area denominator
                                 and to select fully contained leaf meshes.
        min_height : float, default=0.0
                     Exclude faces whose highest vertex is below this height.
        height_bins : int, default=10
                      Number of equally spaced vertical bands.
        height_range : tuple of float, optional
                       Shared (minimum, maximum) height for consistent profile
                       comparisons. All included face heights must fit in range.

        Returns:
        --------
        profile : np.ndarray, float
                  n x 2 array containing height-band centers and the LAI
                  contribution of each band. Contributions sum to the scene LAI.
        """
        return self.calculate_area_index_profile(
            xmin, xmax, ymin, ymax, min_height=min_height,
            type="leaf", height_bins=height_bins, height_range=height_range,
        )

    def calculate_plant_area_index_profile(
        self, xmin, xmax, ymin, ymax, min_height=0.0, height_bins=10,
        height_range=None,
    ):
        """
        Calculate the PAI contribution in equally spaced height bands.

        Parameters:
        -----------
        xmin, xmax, ymin, ymax : float
                                 XY bounds used for the ground-area denominator
                                 and to select fully contained leaf and wood meshes.
        min_height : float, default=0.0
                     Exclude faces whose highest vertex is below this height.
        height_bins : int, default=10
                      Number of equally spaced vertical bands.
        height_range : tuple of float, optional
                       Shared (minimum, maximum) height for consistent profile
                       comparisons. All included face heights must fit in range.

        Returns:
        --------
        profile : np.ndarray, float
                  n x 2 array containing height-band centers and the PAI
                  contribution of each band. Contributions sum to the scene PAI.
        """
        return self.calculate_area_index_profile(
            xmin, xmax, ymin, ymax, min_height=min_height,
            type="plant", height_bins=height_bins, height_range=height_range,
        )

    def calculate_area_index_profile(
        self, xmin, xmax, ymin, ymax, min_height=0.0, type="leaf", height_bins=10,
        height_range=None,
    ):
        """
        Calculate per-height-band LAI or PAI contributions from scene meshes.

        Each triangular face is assigned to a height band using its mean vertex
        height. Its 3D surface area is divided by the XY ground area. Unlike
        calculate_area_index, this method selects only meshes fully contained
        in the XY bounds and uses face-height overlap for min_height, so the
        summed profile can differ when meshes or faces cross those limits.

        Parameters:
        -----------
        xmin, xmax, ymin, ymax : float
                                 XY bounds used for the ground-area denominator
                                 and to select fully contained meshes.
        min_height : float, default=0.0
                     Exclude faces whose highest vertex is below this height.
        type : str, default='leaf'
               Area index to profile: 'leaf' for LAI or 'plant' for PAI
               (leaves plus wood).
        height_bins : int, default=10
                      Number of equally spaced vertical bands.
        height_range : tuple of float, optional
                       Shared (minimum, maximum) height for consistent profile
                       comparisons. All included face heights must fit in range.

        Returns:
        --------
        profile : np.ndarray, float
                  n x 2 array containing band-center heights and the area's
                  contribution to LAI or PAI.

        Raises:
        -------
        ValueError
            If type is unsupported, height_bins is not positive, the ground
            area is not positive, or height_range is invalid or too narrow.
        """
        if type == "leaf":
            type_ids = (self.leaf_id,)
        elif type == "plant":
            type_ids = (self.leaf_id, self.wood_id)
        else:
            raise ValueError(f"Invalid type: {type}. Choose from 'leaf' or 'plant'.")
        if not isinstance(height_bins, int) or height_bins < 1:
            raise ValueError("height_bins must be a positive integer.")
        if height_range is not None:
            if len(height_range) != 2:
                raise ValueError("height_range must contain a minimum and maximum.")
            min_profile_height, max_profile_height = map(float, height_range)
            if (
                not np.isfinite(min_profile_height)
                or not np.isfinite(max_profile_height)
                or min_profile_height >= max_profile_height
            ):
                raise ValueError("height_range must be finite and increasing.")

        ground_area = (xmax - xmin) * (ymax - ymin)
        if ground_area <= 0:
            raise ValueError("The scene ground area must be greater than zero.")

        face_heights = []
        face_areas = []
        reader = SceneReader(self.scene)
        for filepath in self.filepaths:
            if not any(type_id in filepath for type_id in type_ids):
                continue
            resolved_filepath = reader.resolve_filepath(filepath)
            suffix = pathlib.Path(filepath).suffix.lower()
            if suffix == ".obj":
                vertices, faces = reader.read_obj(resolved_filepath)
            elif suffix == ".ply":
                vertices, faces = reader.read_ply(resolved_filepath)
            else:
                raise ValueError(f"Unsupported file format: {filepath}")

            vertices = np.asarray(vertices, dtype=float)
            faces = np.asarray(faces, dtype=int)
            if not len(vertices) or not len(faces):
                continue
            if (
                np.min(vertices[:, 0]) < xmin or np.max(vertices[:, 0]) > xmax
                or np.min(vertices[:, 1]) < ymin or np.max(vertices[:, 1]) > ymax
            ):
                continue

            triangles = vertices[faces[:, :3] - 1]
            eligible = np.max(triangles[:, :, 2], axis=1) >= min_height
            triangles = triangles[eligible]
            if not len(triangles):
                continue
            areas = 0.5 * np.linalg.norm(
                np.cross(
                    triangles[:, 1] - triangles[:, 0],
                    triangles[:, 2] - triangles[:, 0],
                ),
                axis=1,
            )
            face_heights.append(np.mean(triangles[:, :, 2], axis=1))
            face_areas.append(areas)

        if not face_heights:
            return np.empty((0, 2), dtype=float)

        heights = np.concatenate(face_heights)
        areas = np.concatenate(face_areas)
        if height_range is not None:
            if (
                np.min(heights) < min_profile_height
                or np.max(heights) > max_profile_height
            ):
                raise ValueError("height_range must contain all included face heights.")
            edges = np.linspace(min_profile_height, max_profile_height, height_bins + 1)
        else:
            min_profile_height = float(np.min(heights))
            max_profile_height = float(np.max(heights))
            if min_profile_height == max_profile_height:
                edges = np.linspace(
                    min_profile_height - 0.5,
                    max_profile_height + 0.5,
                    height_bins + 1,
                )
            else:
                edges = np.linspace(
                    min_profile_height, max_profile_height, height_bins + 1
                )
        band_indices = np.searchsorted(edges, heights, side="right") - 1
        band_indices = np.clip(band_indices, 0, height_bins - 1)
        area_by_band = np.bincount(
            band_indices, weights=areas, minlength=height_bins
        )
        band_centers = (edges[:-1] + edges[1:]) / 2
        return np.column_stack((band_centers, area_by_band / ground_area))

    @staticmethod
    def calculate_area_triangle(v1, v2, v3):
        """
        Calculate the area of a triangle given its vertices.

        Parameters:
        -----------
        v1, v2, v3 : np.ndarray, float
                     XYZ coordinates of the three triangle vertices.

        Returns:
        --------
        area : float
               Triangle area in squared coordinate units.
        """
        return polygon_area_3d(np.array([v1, v2, v3]))


# class to map canopy cover spatially
class CanopyCoverSpatial:
    """Store crown locations and areas and provide canopy-cover plots."""

    def __init__(self, scene):
        """
        Load scene mesh locations and crown areas for plotting.

        Parameters:
        -----------
        scene : str or pathlib.Path
                Path to the scene XML file.
        """
        self.scene = scene
        sr = SceneReader(scene)
        self.filepaths = sr.read_scene()
        cc = CanopyCover(scene)
        _, _, self.xy_area, self.cluster_points = cc.calculate_canopy_cover()
        self.union = cc.union
        mean_xyz = []
        for filepath in self.filepaths:
            if filepath.endswith('.obj'):
                vertices, _ = sr.read_obj(filepath)
            elif filepath.endswith('.ply'):
                vertices, _ = sr.read_ply(filepath)
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
            mean_xyz.append(np.mean(vertices, axis=0))

        self.locations = np.array(mean_xyz)[:, :2]

    def plot_canopy_cover(self):
        """
        Plot crown locations with marker sizes proportional to crown area.

        Returns:
        --------
        None
            Displays the plot with Matplotlib.
        """
        plt.scatter(self.locations[:, 0], self.locations[:, 1], s=(self.xy_area/np.pi)**0.5, alpha=0.5)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Canopy Cover Spatial Distribution')
        plt.show()

    def plot_canopy_union(self):
        """
        Plot the union of all the crowns (highlighted) against the area outside
        of all the crowns. The outside area is bounded by the convex hull around
        all the crowns in the x-y plane.

        Returns:
        --------
        None
            Displays the plot with Matplotlib.
        """

        fig, ax = plt.subplots()
        for crown_clust in self.cluster_points:
            crown_clust = np.array(crown_clust)
            ax.fill(crown_clust[:,0], crown_clust[:,1], c="g")
        ax.set_aspect('equal')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        plt.show()

    def plot_crown_size_distribution(self):
        """
        Plot the crown size profile, with the crowns ranked from smallest
        to largest.

        Returns:
        --------
        None
            Displays the plot with Matplotlib.
        """
        ranked = np.sort(self.xy_area)
        order = np.arange(1, len(ranked) + 1)
        plt.plot(order, ranked, marker='o', markersize=3)
        plt.xlabel('Crown Rank (smallest to largest)')
        plt.ylabel('Crown Area')
        plt.title('Crown Size Profile')
        plt.show()


# class to remove trees from a scene and create a new one
class CanopyCoverNewScene:
    """Create a scene with a lower canopy cover by retaining a tree subset."""

    def __init__(self, scene, new_cc, orig_cc=None, leaf_id="leaves", terrain_id="terrain", wood_id="wood"):
        """
        Class to create a new scene with a reduced canopy cover compared to
        the original scene. The new canopy cover is specified by the user.

        Parameters:
        -----------
        scene:      str
                    The path to the original scene.
        new_cc:     float
                    The desired canopy cover for the new scene (between 0 and 1).
        orig_cc:    float, optional
                    The original canopy cover of the scene. If not provided, it will be calculated.
        leaf_id:    str, optional
                    String identifier for the leaf objects in the scene.
        terrain_id: str, optional
                    String identifier for the terrain objects in the scene.
        wood_id:    str, optional
                    String identifier for the wood objects in the scene. 

        Returns:
        --------
        None
            Initializes the reducer and calculates the original canopy cover
            when it is not supplied.
        """
        self.scene = scene
        self.sr = SceneReader(scene)
        rscene = self.sr.read_scene()
        leaf_dir = [pathlib.Path(fp).parent for fp in rscene if leaf_id in fp]
        wood_dir = [pathlib.Path(fp).parent for fp in rscene if wood_id in fp]
        can_dirs = [np.unique(leaf_dir)[0], np.unique(wood_dir)[0]]
        self.new_cc = new_cc
        self.orig_cc = orig_cc
        cc = CanopyCover(scene)
        self.ccs = CanopyCoverSpatial(scene)
        if orig_cc is None:
            self.total_area, self.ground_area, self.xy_area, self.cluster_points = cc.calculate_canopy_cover(directories=can_dirs, 
                                                                                                             globstring="*_{iter}_*", 
                                                                                                             iterable=range(len(leaf_dir)),
                                                                                                             ga="can_outline")
            self.orig_cc = self.total_area / self.ground_area
        self.leaf_id = leaf_id
        self.terrain_id = terrain_id
        self.wood_id = wood_id

    def reduce_cc(self):
        """
        Reduce the canopy cover of the scene by removing trees until the new
        canopy cover is reached. The trees are removed based on their spatial
        distribution and conformity to the original crown cover distribution. The
        aim is to preserve, in as much as is possible, the original distributions.

        Returns:
        --------
        retained_leaf_paths : np.ndarray, str
                              Leaf mesh paths retained in the reduced scene.
        """
        if self.new_cc >= self.current_cc:
            raise ValueError("New canopy cover must be less than current canopy cover.")

        ranked = self.xy_area[np.argsort(self.xy_area)]  # rank trees by area
        order = np.arange(len(ranked))
        locations = self.ccs.locations[np.argsort(self.xy_area)]
        new_ranked = self.xy_area[ranked]
        ref_diff = np.abs(np.diff(self.xy_area[ranked]))
        new_bool = np.ones(len(ranked))

        update_cc = self.current_cc
        while update_cc > self.new_cc:
            # calculate crown area weights
            new_diff = np.abs(np.diff(new_ranked))
            diffdiff = np.abs(ref_diff - new_diff) / np.sum(np.abs(ref_diff - new_diff))
            nweights_diff = (1 - diffdiff) / np.sum(1 - diffdiff)

            # calculate distance weights
            if len(locations[new_bool.astype(bool)]) > 0:
                all_dists = []
                for x in locations[~new_bool.astype(bool)]:
                    dists = np.linalg.norm(locations[new_bool.astype(bool)] - x, axis=1)
                    all_dists.append(dists)
                avg_dists = np.mean(all_dists, axis=0)
                nweights_dist = avg_dists / np.sum(avg_dists)
            else:
                nweights_dist = np.ones(new_ranked) / len(new_ranked)

            # merge the weights
            nweights = nweights_diff + nweights_dist
            nweights = nweights / np.sum(nweights)

            # pick the number based on the weights and then determine whether you want the
            # one there or above
            pick = np.random.choice(order[new_bool.astype(bool)][:-1], p=nweights) + np.random.randint(0, 2)
            new_bool[order==pick] = 0
            nranked_int = interp.interp1d(order[new_bool.astype(bool)], 
                                          ranked[new_bool.astype(bool)],
                                          fill_value="extrapolate")
            new_ranked = nranked_int(order)

            new_cc_area = np.sum(new_ranked[new_bool.astype(bool)])
            update_cc = new_cc_area / self.ground_area

        return np.array(self.ccs.filepaths[new_bool])

    def get_matching_stems(self, new_filepaths):
        """
        Get the matching stem filepaths for the new scene based on the new leaf filepaths.

        Parameters:
        -----------
        new_filepaths : iterable of str
                        Leaf mesh paths to match with stems.

        Returns:
        -----------
        matching_stems : list of str
                         Paths to stem meshes paired with the supplied leaves.
        """
        matching_stems = []
        for leaf_path in new_filepaths:
            stem_path = leaf_path.replace(self.leaf_id, self.wood_id)
            if stem_path in self.ccs.filepaths:
                matching_stems.append(stem_path)
        return matching_stems

    def create_new_scene(self, new_filepaths):
        """
        Create a new scene with the specified new leaf filepaths and their matching stem filepaths.

        Parameters:
        -----------
        new_filepaths : iterable of str
                        Leaf mesh paths to include in the new scene.

        Returns:
        --------
        scene_path : str
                     Path to the generated scene XML.
        """
        new_scene_path = self.scene.replace('.xml', f'_cc{self.new_cc*100:.2f}.xml')
        matching_stems = self.get_matching_stems(new_filepaths)
        new_scene = ET.Element('scene')
        obj_elem = ET.SubElement(new_scene, 'object')
        filepath_elem = ET.SubElement(obj_elem, 'filepath')
        filepath_elem.text = self.terrain_id  # Add terrain object to the new scene
        for filepath in new_filepaths + matching_stems:
            obj_elem = ET.SubElement(new_scene, 'object')
            filepath_elem = ET.SubElement(obj_elem, 'filepath')
            filepath_elem.text = filepath

        tree = ET.ElementTree(new_scene)
        tree.write(new_scene_path, encoding='utf-8', xml_declaration=True)

# class to remove leaves from trees in a scene and create a new scene
class LeafAreaIndexNewScene:
    """Create a scene with a target LAI by reducing whole leaf components."""

    def __init__(self, scene, new_lai, orig_lai=None, leaf_id="leaves", terrain_id="terrain", wood_id="wood",
                 height_bins=10):
        """
        Create a scene with a reduced LAI by thinning leaf faces uniformly.

        Parameters:
        -----------
        scene : str or pathlib.Path
                Path to the original scene XML.
        new_lai : float
                  Desired scene LAI in the units of the source scene.
        orig_lai : float or None, default=None
                   Optional original scene LAI; calculated from the scene when omitted.
        leaf_id, terrain_id, wood_id : str
                                      Identifiers for the corresponding scene assets.
        height_bins : int, default=10
                      Number of equally spaced vertical bands used for thinning
                      each tree. Retaining a similar fraction per band preserves
                      the approximate vertical distribution.

        Returns:
        --------
        None
            Initializes scene bounds, original area indices, and the retained
            leaf-area fraction.
        """
        self.scene = scene
        self.sr = SceneReader(scene)
        self.filepaths = self.sr.read_scene()
        self.new_lai = new_lai
        self.leaf_id = leaf_id
        self.terrain_id = terrain_id
        self.wood_id = wood_id
        if not np.isfinite(new_lai) or new_lai < 0:
            raise ValueError("new_lai must be a finite, non-negative value.")
        if not isinstance(height_bins, int) or height_bins < 1:
            raise ValueError("height_bins must be a positive integer.")
        self.height_bins = height_bins
        self.leaf_filepaths = [
            filepath for filepath in self.filepaths
            if leaf_id in filepath and pathlib.Path(filepath).suffix.lower() in {".ply", ".obj"}
        ]
        if not self.leaf_filepaths:
            raise ValueError(f"No leaf mesh files containing '{leaf_id}' were found in the scene.")

        # Use the helper's per-object LAI and PAI routines to establish the common
        # scene bounds, so every tree is measured against the same ground area.
        leaf_meshes = [self._read_mesh(filepath) for filepath in self.leaf_filepaths]
        all_vertices = np.vstack([vertices for vertices, _ in leaf_meshes])
        self.xmin, self.ymin = np.min(all_vertices[:, :2], axis=0)
        self.xmax, self.ymax = np.max(all_vertices[:, :2], axis=0)
        if self.xmax <= self.xmin or self.ymax <= self.ymin:
            raise ValueError("Leaf meshes must span a non-zero ground area.")

        area_calculator = CalculateVegAreaIndex(
            scene, leaf_id=leaf_id, terrain_id=terrain_id, wood_id=wood_id
        )
        self.total_leaf_area, self.ground_area, self.leaf_areas = (
            area_calculator.calculate_leaf_area_index(
                self.xmin, self.xmax, self.ymin, self.ymax
            )
        )
        self.total_plant_area, _, self.plant_areas = (
            area_calculator.calculate_plant_area_index(
                self.xmin, self.xmax, self.ymin, self.ymax
            )
        )
        if self.ground_area <= 0:
            raise ValueError("The scene ground area must be greater than zero.")
        calculated_lai = float(self.total_leaf_area / self.ground_area)
        self.orig_lai = calculated_lai if orig_lai is None else orig_lai
        if not np.isfinite(self.orig_lai) or self.orig_lai < 0:
            raise ValueError("orig_lai must be a finite, non-negative value.")
        if self.new_lai > self.orig_lai:
            raise ValueError("new_lai cannot be greater than orig_lai.")
        if self.total_leaf_area <= 0 and self.new_lai > 0:
            raise ValueError("Cannot create a positive LAI from a scene with no leaf area.")
        if self.orig_lai == 0:
            self.retained_fraction = 0.0
        else:
            self.retained_fraction = self.new_lai / self.orig_lai

        self.original_pai = float(self.total_plant_area / self.ground_area)

    def _read_mesh(self, filepath):
        """Read a leaf mesh through the existing scene reader."""
        resolved_filepath = self.sr.resolve_filepath(filepath)
        suffix = pathlib.Path(resolved_filepath).suffix.lower()
        if suffix == ".ply":
            vertices, faces = self.sr.read_ply(resolved_filepath)
        elif suffix == ".obj":
            vertices, faces = self.sr.read_obj(resolved_filepath)
        else:
            raise ValueError(f"Unsupported leaf mesh format: {filepath}")
        vertices = np.asarray(vertices, dtype=float)
        faces = np.asarray(faces, dtype=int)
        if vertices.ndim != 2 or vertices.shape[1] < 3:
            raise ValueError(f"Leaf mesh has invalid vertices: {filepath}")
        if faces.size == 0:
            faces = np.empty((0, 3), dtype=int)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"Leaf mesh must contain triangular faces: {filepath}")
        if np.any(faces < 1) or np.any(faces > len(vertices)):
            raise ValueError(f"Leaf mesh has an out-of-range face index: {filepath}")
        return vertices, faces

    @staticmethod
    def _connected_leaf_faces(faces):
        """Group triangles into leaves by their shared mesh edges.

        Two triangles are adjacent when they share two vertex indices (an edge).
        Taking the transitive closure of that relation groups a triangulated leaf
        polygon, such as a six-sided leaf made of four triangles, as one unit.
        """
        parent = np.arange(len(faces), dtype=int)

        def find(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return int(index)

        def union(first, second):
            root_first = find(first)
            root_second = find(second)
            if root_first != root_second:
                parent[root_second] = root_first

        edge_faces = {}
        for face_index, face in enumerate(faces):
            for edge_index in range(3):
                edge = tuple(sorted((
                    int(face[edge_index]),
                    int(face[(edge_index + 1) % 3]),
                )))
                edge_faces.setdefault(edge, []).append(face_index)

        for adjacent_faces in edge_faces.values():
            for face_index in adjacent_faces[1:]:
                union(adjacent_faces[0], face_index)

        components = {}
        for face_index in range(len(faces)):
            components.setdefault(find(face_index), []).append(face_index)
        return [
            np.asarray(component, dtype=int)
            for component in components.values()
        ]

    def _select_faces(self, vertices, faces):
        """
        Select complete leaves to retain in each tree and height band.

        A leaf is a transitive component of triangles connected by shared edges.
        Whole leaves, rather than individual triangles, are selected while
        targeting the requested retained-area fraction in each height band.
        """
        if self.retained_fraction >= 1 or len(faces) == 0:
            return np.arange(len(faces), dtype=int)
        if self.retained_fraction <= 0:
            return np.empty(0, dtype=int)

        face_vertices = vertices[faces - 1]
        components = self._connected_leaf_faces(faces)
        face_areas = 0.5 * np.linalg.norm(
            np.cross(
                face_vertices[:, 1] - face_vertices[:, 0],
                face_vertices[:, 2] - face_vertices[:, 0],
            ),
            axis=1,
        )
        face_heights = np.mean(face_vertices[:, :, 2], axis=1)
        leaf_heights = np.asarray([
            np.average(face_heights[component], weights=face_areas[component])
            if np.sum(face_areas[component]) > 0
            else np.mean(face_heights[component])
            for component in components
        ])
        leaf_areas = np.asarray([
            np.sum(face_areas[component]) for component in components
        ])
        min_height = float(np.min(leaf_heights))
        height_range = float(np.max(leaf_heights) - min_height)
        if height_range == 0:
            band_ids = np.zeros(len(components), dtype=int)
        else:
            band_ids = np.minimum(
                ((leaf_heights - min_height) / height_range * self.height_bins).astype(int),
                self.height_bins - 1,
            )

        selected = []
        for band_id in range(self.height_bins):
            band_leaves = np.flatnonzero(band_ids == band_id)
            if len(band_leaves) == 0:
                continue
            band_leaves = band_leaves[
                np.argsort(leaf_heights[band_leaves], kind="stable")
            ]
            # Target the same share of total leaf surface area in each band,
            # while retaining whole leaves at evenly distributed heights.
            band_areas = leaf_areas[band_leaves]
            target_area = self.retained_fraction * np.sum(band_areas)
            expected_count = self.retained_fraction * len(band_leaves)
            candidate_counts = {0, len(band_leaves)}
            for count in (int(np.floor(expected_count)), int(np.ceil(expected_count))):
                candidate_counts.update(
                    max(0, min(len(band_leaves), count + offset))
                    for offset in (-1, 0, 1)
                )

            best_leaves = np.empty(0, dtype=int)
            best_score = (float("inf"), float("inf"))
            phase_count = min(len(band_leaves), 16)
            for keep_count in candidate_counts:
                if keep_count == 0:
                    candidates = [np.empty(0, dtype=int)]
                elif keep_count == len(band_leaves):
                    candidates = [band_leaves]
                else:
                    candidates = []
                    for phase in np.arange(phase_count) / phase_count:
                        positions = np.floor(
                            (np.arange(keep_count) + phase)
                            * len(band_leaves) / keep_count
                        ).astype(int)
                        candidates.append(band_leaves[positions])
                for candidate_leaves in candidates:
                    kept_area = np.sum(leaf_areas[candidate_leaves])
                    score = (
                        abs(target_area - kept_area),
                        abs(len(candidate_leaves) - expected_count),
                    )
                    if score < best_score:
                        best_score = score
                        best_leaves = candidate_leaves
            for leaf_index in best_leaves:
                selected.extend(components[int(leaf_index)])
        return np.asarray(sorted(selected), dtype=int)

    @staticmethod
    def _write_ascii_ply(source_path, output_path, faces, kept_face_ids):
        """Write a reduced ASCII PLY, preserving its original header and vertex records."""
        with open(source_path, "r", encoding="utf-8") as source:
            lines = source.readlines()
        if not lines or lines[0].strip() != "ply":
            raise ValueError(f"Not a PLY file: {source_path}")

        format_line = next((line for line in lines if line.startswith("format ")), "")
        if not format_line.startswith("format ascii "):
            raise ValueError(f"Only ASCII PLY files can be reduced: {source_path}")
        try:
            end_header = next(i for i, line in enumerate(lines) if line.strip() == "end_header")
        except StopIteration as exc:
            raise ValueError(f"PLY file has no end_header: {source_path}") from exc

        # Parse element sizes so vertex attributes and other PLY elements are
        # copied unchanged while only the triangular face records are filtered.
        elements = []
        for line in lines[1:end_header]:
            fields = line.split()
            if len(fields) == 3 and fields[0] == "element":
                elements.append((fields[1], int(fields[2])))
        if not any(name == "vertex" for name, _ in elements):
            raise ValueError(f"PLY file has no vertex element: {source_path}")
        face_count = next((count for name, count in elements if name == "face"), None)
        if face_count is None:
            raise ValueError(f"PLY file has no face element: {source_path}")
        if face_count != len(faces):
            raise ValueError(f"PLY face records do not match parsed faces: {source_path}")

        kept_face_ids = np.asarray(kept_face_ids, dtype=int)
        used_vertex_ids = (
            np.unique(faces[kept_face_ids, :3] - 1)
            if len(kept_face_ids)
            else np.empty(0, dtype=int)
        )
        vertex_element_count = next(count for name, count in elements if name == "vertex")
        vertex_remap = np.full(vertex_element_count, -1, dtype=int)
        vertex_remap[used_vertex_ids] = np.arange(len(used_vertex_ids))

        updated_header = list(lines[:end_header])
        vertex_element_index = next(
            i for i, line in enumerate(updated_header)
            if line.split()[:2] == ["element", "vertex"]
        )
        updated_header[vertex_element_index] = f"element vertex {len(used_vertex_ids)}\n"
        face_element_index = next(
            i for i, line in enumerate(updated_header)
            if line.split()[:2] == ["element", "face"]
        )
        updated_header[face_element_index] = f"element face {len(kept_face_ids)}\n"
        body = lines[end_header + 1:]
        output_body = []
        offset = 0
        for element_name, count in elements:
            records = body[offset:offset + count]
            if len(records) != count:
                raise ValueError(f"PLY element records are incomplete: {source_path}")
            if element_name == "vertex":
                output_body.extend(records[int(i)] for i in used_vertex_ids)
            elif element_name == "face":
                for face_id in kept_face_ids:
                    fields = records[int(face_id)].split()
                    vertex_count = int(fields[0])
                    old_indices = np.asarray(fields[1:vertex_count + 1], dtype=int)
                    new_indices = vertex_remap[old_indices]
                    if np.any(new_indices < 0):
                        raise ValueError(
                            f"PLY face references a removed vertex: {source_path}"
                        )
                    remapped = [str(vertex_count), *(str(index) for index in new_indices)]
                    remapped.extend(fields[vertex_count + 1:])
                    output_body.append(" ".join(remapped) + "\n")
            else:
                output_body.extend(records)
            offset += count
        if offset != len(body):
            raise ValueError(f"Unexpected data after PLY element records: {source_path}")

        with open(output_path, "w", encoding="utf-8") as output:
            output.writelines(updated_header)
            output.write(lines[end_header])
            output.write("\n" if not lines[end_header].endswith("\n") else "")
            output.writelines(output_body)

    @staticmethod
    def _write_obj_as_ply(output_path, vertices, faces, kept_face_ids):
        """Convert an OBJ leaf mesh to an ASCII PLY containing the retained faces."""
        kept_face_ids = np.asarray(kept_face_ids, dtype=int)
        kept_faces = faces[kept_face_ids, :3] - 1
        used_vertex_ids = (
            np.unique(kept_faces.reshape(-1))
            if len(kept_faces)
            else np.empty(0, dtype=int)
        )
        vertex_remap = np.full(len(vertices), -1, dtype=int)
        vertex_remap[used_vertex_ids] = np.arange(len(used_vertex_ids))
        remapped_faces = vertex_remap[kept_faces]
        with open(output_path, "w", encoding="utf-8") as output:
            output.write("ply\nformat ascii 1.0\n")
            output.write(f"element vertex {len(used_vertex_ids)}\n")
            output.write("property float x\nproperty float y\nproperty float z\n")
            output.write(f"element face {len(kept_face_ids)}\n")
            output.write("property list uchar int vertex_indices\nend_header\n")
            for vertex in vertices[used_vertex_ids, :3]:
                output.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
            for face in remapped_faces:
                output.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    def reduce_lai(self):
        """
        Create one reduced PLY per leaf mesh, applying the same proportional
        reduction to every tree and distributing it evenly over tree height.

        Returns:
        --------
        generated_paths : list of str
                          Generated leaf PLY paths in the order of the scene's
                          original leaf mesh references.
        """
        generated_paths = []
        suffix = f"_lai{self.new_lai:.4f}.ply"
        for filepath in self.leaf_filepaths:
            vertices, faces = self._read_mesh(filepath)
            kept_face_ids = self._select_faces(vertices, faces)
            source_path = pathlib.Path(self.sr.resolve_filepath(filepath))
            output_path = source_path.with_name(source_path.stem + suffix)
            if source_path.suffix.lower() == ".ply":
                self._write_ascii_ply(
                    source_path, output_path, faces, kept_face_ids
                )
            else:
                self._write_obj_as_ply(output_path, vertices, faces, kept_face_ids)
            generated_paths.append(str(output_path))
        self.new_filepaths = generated_paths
        return generated_paths

    def calculate_new_indices(self):
        """
        Calculate LAI and PAI for the generated scene.

        Returns:
        --------
        new_lai : float
                  LAI calculated from the reduced leaf meshes.
        new_pai : float
                  PAI calculated from the reduced scene's leaf and wood meshes.
        """
        if not hasattr(self, "new_filepaths"):
            raise RuntimeError("Call reduce_lai() before calculating the new indices.")
        if not hasattr(self, "new_scene_path"):
            self.create_new_scene(self.new_filepaths)
        area_calculator = CalculateVegAreaIndex(
            self.new_scene_path,
            leaf_id=self.leaf_id,
            terrain_id=self.terrain_id,
            wood_id=self.wood_id,
        )
        self.new_leaf_area, _, self.new_leaf_areas = (
            area_calculator.calculate_leaf_area_index(
                self.xmin, self.xmax, self.ymin, self.ymax
            )
        )
        self.new_plant_area, _, self.new_plant_areas = (
            area_calculator.calculate_plant_area_index(
                self.xmin, self.xmax, self.ymin, self.ymax
            )
        )
        self.new_lai_calculated = self.new_leaf_area / self.ground_area
        self.new_pai_calculated = self.new_plant_area / self.ground_area
        return self.new_lai_calculated, self.new_pai_calculated

    def create_new_scene(self, new_filepaths=None):
        """
        Write a copy of the input XML that references the generated leaf PLY files.

        If no generated paths are supplied, reduce_lai() is run automatically.
        The original scene is left untouched, and all non-leaf references are
        preserved as they appeared in the source XML.

        Parameters:
        -----------
        new_filepaths : iterable of str or None, default=None
                        Generated leaf mesh paths to reference. When omitted,
                        the reducer uses existing generated paths or creates them.

        Returns:
        --------
        scene_path : str
                     Path to the newly written scene XML.
        """
        if new_filepaths is None:
            new_filepaths = getattr(self, "new_filepaths", None)
        if new_filepaths is None:
            new_filepaths = self.reduce_lai()
        if len(new_filepaths) != len(self.leaf_filepaths):
            raise ValueError("One generated mesh path is required for every leaf mesh.")

        replacements = dict(zip(self.leaf_filepaths, new_filepaths))
        xml_tree = ET.parse(self.scene)
        for element in xml_tree.iter("param"):
            if element.get("key") == "filepath" and element.get("value") in replacements:
                element.set("value", replacements[element.get("value")])

        scene_path = pathlib.Path(self.scene)
        new_scene_path = scene_path.with_name(
            f"{scene_path.stem}_lai{self.new_lai:.4f}{scene_path.suffix}"
        )
        xml_tree.write(new_scene_path, encoding="utf-8", xml_declaration=True)
        self.new_scene_path = str(new_scene_path)
        return self.new_scene_path