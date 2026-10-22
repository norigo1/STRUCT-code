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
        Read the PLY file and return a list of vertices and faces.
        """
        vertices = []
        faces = []
        with open(filename, 'r') as f:
            header = True
            for line in f:
                if header:
                    if line.startswith('end_header'):
                        header = False
                    continue
                if line.startswith('3 '):
                    faces.append(list(map(int, line.strip().split()[1:])))
                else:
                    vertices.append(list(map(float, line.strip().split()))[:3])
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


# class to calculate canopy cover
class CanopyCover:
    def __init__(self, scene):
        self.scene = scene
        self.sr = SceneReader(scene)
        self.filepaths = self.sr.read_scene()

    def calculate_canopy_cover(self, directories, globstring, iterable):
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
        """
        crown_outlines = []
        ca_total = 0.0
        for i in iterable:
            print(f"Processing tree {i/len(iterable)*100:.2f}%...", end='\r')
            for directory in directories:
                if directory[-1] != '/':
                    directory += '/'

            vertices, faces = self.sr.read_tree(directories, globstring.format(iter=i))
            ca, co = self.calculate_crown_area_fill_gaps(vertices, faces)
            crown_outlines.append(co)
            ca_total += ca

        crown = np.vstack([c for c in crown_outlines if len(c) > 0])
        if len(crown) > 0:
            canopy_outline = self.calculate_alpha_shape(crown)
            pdb.set_trace()
            if canopy_outline.geom_type == 'Polygon':
                canopy_outline = np.array(canopy_outline.exterior.coords)
                canopy_ga = Polygon(canopy_outline).area
            elif canopy_outline.geom_type == 'MultiPolygon':
                canopy_outline = np.array(canopy_outline.convex_hull.exterior.coords)
                canopy_ga = Polygon(canopy_outline).area
            else:
                canopy_ga = 0.0
        else:
            canopy_ga = 0.0
        
        return ca_total, canopy_ga, ca_total / canopy_ga, crown_outlines, canopy_outline
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

    def calculate_concave_hull(self, points, ratio=0.0):
        """Calculate Shapely's concave hull from point coordinates."""
        points = np.asarray(points)[:, :2]
        return shapely.concave_hull(MultiPoint(points), ratio=ratio)

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
            leaf_polygons.append(Polygon([v1, v2, v3]))
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
class calculateLeafAreaIndex:
    def __init__(self, scene):
        self.scene = scene
        sr = SceneReader(scene)
        self.filepaths = sr.read_scene()

    def calculate_leaf_area_index(self, xmin, xmax, ymin, ymax, min_height=0.0):
        """
        Calculate the leaf area index of the scene by reading in the vertices and 
        faces of the scene's vegetation objects. Works for wood files where the stem
        is made of facets. Will need a new method if the stem is made of cylinders.

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
        leaf_area = []
        for filepath in self.filepaths:
            if filepath.endswith('.obj'):
                vertices, faces = SceneReader(self.scene).read_obj(filepath)
            elif filepath.endswith('.ply'):
                vertices, faces = SceneReader(self.scene).read_ply(filepath)
            else:
                raise ValueError(f"Unsupported file format: {filepath}")

            minvx = np.min(np.array(vertices)[:, 0])
            maxvx = np.max(np.array(vertices)[:, 0])
            minvy = np.min(np.array(vertices)[:, 1])
            maxvy = np.max(np.array(vertices)[:, 1])
            if minvx < xmin or maxvx > xmax or minvy < ymin or maxvy > ymax:
                continue  # skip this vegetation object if it is outside the bounding box

            leaf_area_tree = []
            for face in faces:
                v1 = np.array(vertices[face[0] - 1])
                v2 = np.array(vertices[face[1] - 1])
                v3 = np.array(vertices[face[2] - 1])
                if np.max([v1[2], v2[2], v3[2]]) < min_height:
                    continue  # skip this face if it is below the minimum height
                area = self.calculate_area_triangle(v1, v2, v3)
                leaf_area_tree.append(area)

            leaf_area.append(np.sum(leaf_area_tree))

        total_leaf_area = np.sum(leaf_area)
        ground_area = (xmax - xmin) * (ymax - ymin)

        return total_leaf_area, ground_area, np.array(leaf_area)

    @staticmethod
    def calculate_area_triangle(v1, v2, v3):
        """
        Calculate the area of a triangle given its vertices.
        """
        return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))


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
    def __init__(self, scene, new_cc, leaf_id="leaves", terrain_id="terrain", wood_id="wood"):
        """
        Class to create a new scene with a reduced canopy cover compared to
        the original scene. The new canopy cover is specified by the user.

        Parameters:
        scene:      str
                    The path to the original scene.
        new_cc:     float
                    The desired canopy cover for the new scene (between 0 and 1).
        leaf_id:    str, optional
                    String identifier for the leaf objects in the scene.
        terrain_id: str, optional
                    String identifier for the terrain objects in the scene.
        wood_id:    str, optional
                    String identifier for the wood objects in the scene. 
        """
        self.scene = scene
        self.new_cc = new_cc
        cc = CanopyCover(scene)
        self.ccs = CanopyCoverSpatial(scene)
        self.total_area, self.ground_area, self.xy_area, self.cluster_points = cc.calculate_canopy_cover()
        self.current_cc = self.total_area / self.ground_area
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
