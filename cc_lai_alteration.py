#######################################################################
#
# Code to alter the canopy cover and leaf area index of a scene, 
# by changing the scene's vegetation parameters.
#
# Written by Niall Origo, 02/10/2026
#
######################################################################

import matplotlib.pyplot as plt
import numpy as np
import shapely
from xml.etree import ElementTree as ET
from xml.dom import minidom
from scipy.spatial import ConvexHull, Delaunay
from scipy import interpolate as interp
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union
import pathlib
import pdb
import time

# read in scene
class SceneReader:
    def __init__(self, scene_path):
        self.scene_path = scene_path

    def resolve_filepath(self, filepath):
        """Resolve scene-relative assets when the path is not valid from cwd."""
        path = pathlib.Path(filepath)
        if path.exists() or path.is_absolute():
            return str(path)
        scene_relative_path = pathlib.Path(self.scene_path).parent / path
        return str(scene_relative_path) if scene_relative_path.exists() else str(path)

    def read_scene(self):
        """
        Read the scene from the specified path and return a Scene object.
        """
        xml_tree = minidom.parse(self.scene_path)
        ilist = xml_tree.getElementsByTagName("param")
        filepaths = []
        for i in ilist:
            if i.attributes["key"].value == "filepath":
                filepaths.append(i.attributes["value"].value)
        return filepaths

    def read_obj(self, filename):
        """
        Read the OBJ file and return a list of vertices and faces.
        """
        vertices = []
        faces = []
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    vertices.append(list(map(float, line.strip().split()[1:]))[:3])
                elif line.startswith('f '):
                    faces.append(list(map(int, line.strip().split()[1:])))
        return vertices, faces

    def read_ply(self, filename):
        """
        Read an ASCII PLY file and return its vertices and triangular faces.
        """
        vertices = []
        faces = []
        with open(filename, 'r') as f:
            lines = f.readlines()
        if not lines or lines[0].strip() != "ply":
            raise ValueError(f"Not a PLY file: {filename}")
        if not any(line.startswith("format ascii ") for line in lines):
            raise ValueError(f"Only ASCII PLY files are supported: {filename}")
        try:
            end_header = next(i for i, line in enumerate(lines) if line.strip() == "end_header")
        except StopIteration as exc:
            raise ValueError(f"PLY file has no end_header: {filename}") from exc

        elements = []
        for line in lines[1:end_header]:
            fields = line.split()
            if len(fields) == 3 and fields[0] == "element":
                elements.append((fields[1], int(fields[2])))
        body = lines[end_header + 1:]
        offset = 0
        for element_name, count in elements:
            records = body[offset:offset + count]
            if len(records) != count:
                raise ValueError(f"PLY element records are incomplete: {filename}")
            if element_name == "vertex":
                for record in records:
                    values = list(map(float, record.split()))
                    if len(values) < 3:
                        raise ValueError(f"PLY vertex has fewer than three coordinates: {filename}")
                    vertices.append(values[:3])
            elif element_name == "face":
                for record in records:
                    values = list(map(int, record.split()))
                    if not values or values[0] != 3 or len(values) < 4:
                        raise ValueError(f"Only triangular PLY faces are supported: {filename}")
                    # PLY face indices are zero-based; convert them to the
                    # one-based indexing shared by the OBJ reader and area code.
                    faces.append([index + 1 for index in values[1:4]])
            offset += count
        return vertices, faces

    def read_tree(self, directories, globstring, hmin=2.0):
        """
        Read the tree file and return a list of vertices and faces.
        
        Parameters:
        directories: list of str
            The directories containing the folders of the wood and leaf elements of the tree.
        globstring: str
            The glob string to match the leaf and wood elements of the tree.
        hmin: float
            The minimum height of the tree to be considered.
        
        Returns:
        list of tuples
            Vertices, faces tuples of the combined tree elements that match the globstring 
            and are above the minimum height.
        """
        vertl = [] 
        facel = []
        for directory in directories:
            if directory[-1] != '/':
                directory += '/'
            for filepath in pathlib.Path(directory).glob(globstring):
                if filepath.suffix == '.obj':
                    vertices, faces = self.read_obj(str(filepath))
                elif filepath.suffix == '.ply':
                    vertices, faces = self.read_ply(str(filepath))
                else:
                    raise ValueError(f"Unsupported file format: {filepath}")
                verts = np.array(vertices)
                faces = np.array(faces)
                """
                hfilt = verts[:,2] > hmin
                vfilt = verts[hfilt]
                fvals = np.where(hfilt)[0] + 1
                finds = []
                for f in faces:
                    if f[0] in fvals and f[1] in fvals and f[2] in fvals:
                        finds.append(True)
                    else:
                        finds.append(False)
                """
                vertl.append(verts)#vfilt)
                facel.append(faces)#[np.array(finds)])

        verto = np.vstack(vertl)
        faceo = np.vstack((facel[0], facel[1] + len(vertl[0])))  # adjust face indices for concatenated vertices
        return verto, faceo        


class GapFraction:

    def __init__(self, scene):
        self.scene = scene
        sr = SceneReader(scene)
        self.filepaths = sr.read_scene()

    def reproject_vertices(self, vertices, point, normal):
        """
        Reproject the vertices onto a plane defined by a point and a normal vector.
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
        vertices, faces: OBJ-style vertices and 1-indexed triangular faces.
        zenith: float
            The zenith angle in degrees.
        azimuth: float
            The azimuth angle in degrees.
        alpha_radius: float
            The radius of the alpha shape. Larger values will result in a more convex
            shape, while smaller values will result in a more concave shape.

        Returns:
        gap_fraction: float
            The gap fraction of the vegetation object.
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
    def __init__(self, scene):
        self.scene = scene
        self.sr = SceneReader(scene)
        self.filepaths = self.sr.read_scene()

    def calculate_canopy_cover(self, directories, globstring, iterable, ga="max_extent"):
        """
        Calculate the canopy cover of the scene by reading in the vertices and faces 
        of the scene's vegetation objects. Works for wood files where the stem is made
        of facets. Will need a new method if the stem is made of cylinders.

        Parameters:
        directories: list of str
            The directories containing the folders of the wood and leaf elements of the tree.
        globstring: str
            The glob string to match the leaf and wood elements of the tree.
            Note that globstring should have "{iter}" in it, which will be replaced by the 
            elements of the iterable. e.g. "tree_{iter}_*" will match "tree_1_*_leaves.obj", 
            "tree_1_*_wood.obj", etc.
        iterable: list, array, etc.
            An iterable of some kind that is fed to the globstring which will be used to 
            match the leaf and wood elements of the tree.
        ga: str
            The method to calculate the ground area. Options are "max_extent" (default), 
            "convexhull", or "can_outline". "max_extent" is the area of the bounding box 
            around all the crowns, "convex_hull" is the area of the convex hull around all 
            the crowns, and "can_outline" is the area of the shape around all the crowns (if 
            there is only one polygon for the whole canopy).
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
        directories: list of str
            The directories containing the folders of the wood and leaf elements of the tree.
        globstring: str
            The glob string to match the leaf and wood elements of the tree.
            Note that globstring should have "{iter}" in it, which will be replaced by the 
            elements of the iterable. e.g. "tree_{iter}_*" will match "tree_1_*_leaves.obj", 
            "tree_1_*_wood.obj", etc.
        iterable: list, array, etc.
            An iterable of some kind that is fed to the globstring which will be used to 
            match the leaf and wood elements of the tree.
        ga: str, optional
            The method to use for calculating the ground area. Options are "max_extent", "convexhull", and "can_outline".
        alpha_radius: float, optional
            The radius of the alpha shape (concave hull) to use for calculating the canopy cover.
        canopy_cover_outline: list of tuples, optional (come back to this)
            The outline of the canopy cover, as a list of (x, y) tuples.
        """
        something = 0

    def calculate_alpha_shape(self, points, alpha_radius=0.2):
        """
        Calculate the alpha shape (concave hull) of a set of points.
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
        vertices, faces: OBJ-style vertices and 1-indexed triangular faces.
        alpha_radius: maximum circumradius of a retained Delaunay triangle, in the
            units of the vertices (default: 0.2 * longest side of the bounding box).
            Large values approach the convex hull.
        Returns:
        crown_area: float
            The area enclosed by the alpha shape (holes filled).
        contour: np.array
            The exterior of the largest component as (x, y) points.
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
        Order points greedily: the next point is chosen from the 10 nearest remaining
        points using an equal weighting of counter-clockwise angle and distance.
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
class calculateVegAreaIndex:
    def __init__(self, scene, leaf_id="leaves", terrain_id="terrain", wood_id="wood"):
        self.scene = scene
        sr = SceneReader(scene)
        self.filepaths = sr.read_scene()
        self.leaf_id = leaf_id
        self.terrain_id = terrain_id
        self.wood_id = wood_id

    def calculate_area_index(self, xmin, xmax, ymin, ymax, min_height=0.0, type="leaf"):
        """
        Calculate the leaf/plant area index of the scene by reading in the vertices and 
        faces of the scene's vegetation objects. Works for wood files where the stem
        is made of facets. Will need a new method if the stem is made of cylinders.

        Note: if you wish to calculate the LAI without the bounding box just use the 
        total_leaf_area and divide by your specific ground area (e.g. canopy perimeter).

        Parameters:
        xmin, xmax, ymin, ymax: float
            The bounding box coordinates to calculate the ground area.
        Returns:
        total_leaf_area: float
            The total leaf area of the scene.
        ground_area: float
            The ground area of the scene.
        leaf_area: np.array
            The leaf area of each vegetation object in the scene.
        min_height: float
            The minimum height of the vegetation objects to be considered for leaf area 
            calculation.
        """
        area = []
        if type == "leaf":
            type_id = self.leaf_id
        elif type == "wood":
            type_id = self.wood_id
        else:
            raise ValueError(f"Invalid type: {type}. Choose from 'leaf' or 'wood'.")
        for filepath in self.filepaths:
            if type_id not in filepath:
                continue
            resolved_filepath = SceneReader(self.scene).resolve_filepath(filepath)
            if pathlib.Path(filepath).suffix.lower() == ".obj":
                vertices, faces = SceneReader(self.scene).read_obj(resolved_filepath)
            elif pathlib.Path(filepath).suffix.lower() == ".ply":
                vertices, faces = SceneReader(self.scene).read_ply(resolved_filepath)
            else:
                raise ValueError(f"Unsupported file format: {filepath}")

            minvx = np.min(np.array(vertices)[:, 0])
            maxvx = np.max(np.array(vertices)[:, 0])
            minvy = np.min(np.array(vertices)[:, 1])
            maxvy = np.max(np.array(vertices)[:, 1])
            if minvx < xmin or maxvx > xmax or minvy < ymin or maxvy > ymax:
                continue  # skip this vegetation object if it is outside the bounding box

            area_tree = []
            for face in faces:
                v1 = np.array(vertices[face[0] - 1])
                v2 = np.array(vertices[face[1] - 1])
                v3 = np.array(vertices[face[2] - 1])
                if np.max([v1[2], v2[2], v3[2]]) < min_height:
                    continue  # skip this face if it is below the minimum height
                tarea = self.calculate_area_triangle(v1, v2, v3)
                area_tree.append(tarea)

            area.append(np.sum(area_tree))

        total_area = np.sum(area)
        ground_area = (xmax - xmin) * (ymax - ymin)
        return total_area, ground_area, np.array(area)

    def calculate_leaf_area_index(self, xmin, xmax, ymin, ymax, min_height=0.0):
        """
        Calculate the leaf area index (LAI) of the scene.

        Parameters:
        xmin, xmax, ymin, ymax: float
            The bounding box coordinates to calculate the ground area.
        min_height: float
            The minimum height of the vegetation objects to be considered for leaf area 
            calculation.
        Returns:
        total_leaf_area: float
            The total leaf area of the scene.
        ground_area: float
            The ground area of the scene.
        leaf_area: np.array
            The leaf area of each vegetation object in the scene.
        """
        total_leaf_area, ground_area, leaf_area = self.calculate_area_index(xmin, xmax, ymin, ymax, 
                                                                            min_height=min_height, 
                                                                            type="leaf")
        return total_leaf_area, ground_area, leaf_area

    def calculate_plant_area_index(self, xmin, xmax, ymin, ymax, min_height=0.0):
        """
        Calculate the plant area index (PAI) of the scene.

        Parameters:
        xmin, xmax, ymin, ymax: float
            The bounding box coordinates to calculate the ground area.
        min_height: float
            The minimum height of the vegetation objects to be considered for plant area 
            calculation.
        Returns:
        total_plant_area: float
            The plant area index of the scene.
        """
        total_leaf_area, ground_area, leaf_area = self.calculate_area_index(xmin, xmax, ymin, ymax, 
                                                                            min_height=min_height, 
                                                                            type="leaf")
        total_wood_area, _, wood_area = self.calculate_area_index(xmin, xmax, ymin, ymax, 
                                                                  min_height=min_height, 
                                                                  type="wood")
        total_plant_area = total_leaf_area + total_wood_area
        # Leaf and wood object counts need not match; concatenate their areas rather
        # than requiring the arrays to have the same length.
        plant_area = np.concatenate((leaf_area, wood_area))
        return total_plant_area, ground_area, np.array(plant_area)

    @staticmethod
    def calculate_area_triangle(v1, v2, v3):
        """
        Calculate the area of a triangle given its vertices.
        """
        return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))


# Keep a conventional class-name spelling available while preserving the original API.
CalculateVegAreaIndex = calculateVegAreaIndex


# class to map canopy cover spatially
class CanopyCoverSpatial:
    def __init__(self, scene):
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
        Plot the canopy cover spatially using matplotlib.
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
    def __init__(self, scene, new_cc, orig_cc=None, leaf_id="leaves", terrain_id="terrain", wood_id="wood"):
        """
        Class to create a new scene with a reduced canopy cover compared to
        the original scene. The new canopy cover is specified by the user.

        Parameters:
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
        new_filepaths: list
                       List of filepaths for the new leaf objects in the scene.

        Returns:
        matching_stems: list
                        List of filepaths for the matching stem objects in the scene.
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
        new_filepaths: list
                       List of filepaths for the new leaf objects in the scene.
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
    def __init__(self, scene, new_lai, orig_lai=None, leaf_id="leaves", terrain_id="terrain", wood_id="wood",
                 height_bins=10):
        """
        Create a scene with a reduced LAI by thinning leaf faces uniformly.

        Parameters:
        scene: Path to the original scene XML.
        new_lai: Desired LAI, in the same units as the existing scene LAI.
        orig_lai: Optional LAI override; calculated from the scene if omitted.
        leaf_id, terrain_id, wood_id: Identifiers for the corresponding scene files.
        height_bins: Number of equally spaced vertical bands used for thinning each
            tree. Matching the retained fraction in every band keeps the reduction
            distributed throughout each tree's height.
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

    def _select_faces(self, vertices, faces):
        """
        Select a consistent share of faces per tree and height band.

        Face centers are distributed evenly through each band's vertical range;
        using the same retained share in every band avoids concentrating removal
        at the top or bottom of a tree.
        """
        if self.retained_fraction >= 1 or len(faces) == 0:
            return np.arange(len(faces), dtype=int)
        if self.retained_fraction <= 0:
            return np.empty(0, dtype=int)

        face_vertices = vertices[faces - 1]
        face_heights = np.mean(face_vertices[:, :, 2], axis=1)
        face_areas = 0.5 * np.linalg.norm(
            np.cross(
                face_vertices[:, 1] - face_vertices[:, 0],
                face_vertices[:, 2] - face_vertices[:, 0],
            ),
            axis=1,
        )
        min_height = float(np.min(face_heights))
        height_range = float(np.max(face_heights) - min_height)
        if height_range == 0:
            band_ids = np.zeros(len(faces), dtype=int)
        else:
            band_ids = np.minimum(
                ((face_heights - min_height) / height_range * self.height_bins).astype(int),
                self.height_bins - 1,
            )

        selected = []
        for band_id in range(self.height_bins):
            band_faces = np.flatnonzero(band_ids == band_id)
            if len(band_faces) == 0:
                continue
            band_faces = band_faces[
                np.argsort(face_heights[band_faces], kind="stable")
            ]
            # Target the same share of surface area in every tree-height band.
            # Evenly spaced face samples keep the selected geometry distributed
            # vertically, while trying nearby sample counts/phases reduces the
            # error caused by faces having different areas.
            band_areas = face_areas[band_faces]
            target_area = self.retained_fraction * np.sum(band_areas)
            expected_count = self.retained_fraction * len(band_faces)
            candidate_counts = {0, len(band_faces)}
            for count in (int(np.floor(expected_count)), int(np.ceil(expected_count))):
                candidate_counts.update(
                    max(0, min(len(band_faces), count + offset))
                    for offset in (-1, 0, 1)
                )

            best_faces = np.empty(0, dtype=int)
            best_score = (float("inf"), float("inf"))
            phase_count = min(len(band_faces), 16)
            for keep_count in candidate_counts:
                if keep_count == 0:
                    candidates = [np.empty(0, dtype=int)]
                elif keep_count == len(band_faces):
                    candidates = [band_faces]
                else:
                    candidates = []
                    for phase in np.arange(phase_count) / phase_count:
                        positions = np.floor(
                            (np.arange(keep_count) + phase)
                            * len(band_faces) / keep_count
                        ).astype(int)
                        candidates.append(band_faces[positions])
                for candidate in candidates:
                    kept_area = np.sum(face_areas[candidate])
                    score = (
                        abs(target_area - kept_area),
                        abs(len(candidate) - expected_count),
                    )
                    if score < best_score:
                        best_score = score
                        best_faces = candidate
            selected.extend(best_faces)
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

        updated_header = list(lines[:end_header])
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
            if element_name == "face":
                output_body.extend(records[int(i)] for i in kept_face_ids)
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
        with open(output_path, "w", encoding="utf-8") as output:
            output.write("ply\nformat ascii 1.0\n")
            output.write(f"element vertex {len(vertices)}\n")
            output.write("property float x\nproperty float y\nproperty float z\n")
            output.write(f"element face {len(kept_face_ids)}\n")
            output.write("property list uchar int vertex_indices\nend_header\n")
            for vertex in vertices[:, :3]:
                output.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
            for face_id in kept_face_ids:
                face = faces[int(face_id)] - 1
                output.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    def reduce_lai(self):
        """
        Create one reduced PLY per leaf mesh, applying the same proportional
        reduction to every tree and distributing it evenly over tree height.

        Returns the generated leaf PLY paths in the same order as the scene's
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
        """Calculate LAI and PAI for the generated scene leaf meshes."""
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