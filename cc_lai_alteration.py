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
from xml.etree import ElementTree as ET
from xml.dom import minidom
from scipy.spatial import ConvexHull
from scipy import interpolate as interp
from shapely.geometry import Polygon
from shapely.ops import unary_union
import pathlib
import pdb

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


# class to calculate canopy cover
class CanopyCover:
    def __init__(self, scene):
        self.scene = scene
        sr = SceneReader(scene)
        self.filepaths = sr.read_scene()

    def calculate_canopy_cover(self):
        something = 0

    def calculate_crown_area_fill_gaps(self, vertices, faces):
        """
        Calculate crown area from sides that do not intersect sides of other triangles.
        """
        lines = []
        vertices = np.asarray(vertices)[:, :2]
        for f,face in enumerate(faces):
            face_vertices = [vertices[index - 1] for index in face]
            for start_corner, end_corner in ((0, 1), (1, 2), (2, 0)):
                start = face_vertices[start_corner]
                end = face_vertices[end_corner]
                a = end[1] - start[1]
                b = start[0] - end[0]
                c = a * start[0] + b * start[1]
                lines.append((a, b, c, f, start_corner, end_corner))

        exterior_points = []
        for l,line in enumerate(lines):
            print(f"Processing line {l+1}/{len(lines)}")
            face_index = line[3]
            face = faces[face_index]
            start = vertices[face[line[4]] - 1]
            end = vertices[face[line[5]] - 1]
            side_vector = end - start
            side_length_squared = np.dot(side_vector, side_vector)
            if side_length_squared == 0:
                continue

            start_ray_intersects = False
            end_ray_intersects = False
            for other_line in lines:
                if other_line[3] == face_index:
                    continue

                other_face = faces[other_line[3]]
                other_start = vertices[other_face[other_line[4]] - 1]
                other_end = vertices[other_face[other_line[5]] - 1]
                other_vector = other_end - other_start
                if np.dot(other_vector, other_vector) == 0:
                    continue

                cross = side_vector[0] * other_vector[1] - side_vector[1] * other_vector[0]
                offset = other_start - start
                if np.isclose(cross, 0.0) and np.isclose(
                    side_vector[0] * offset[1] - side_vector[1] * offset[0], 0.0
                ):
                    other_start_position = np.dot(offset, side_vector) / side_length_squared
                    other_end_position = np.dot(other_end - start, side_vector) / side_length_squared
                    overlap_start = min(other_start_position, other_end_position)
                    overlap_end = max(other_start_position, other_end_position)
                    if overlap_start < 0.0 and not np.isclose(overlap_start, 0.0):
                        start_ray_intersects = True
                    if overlap_end > 1.0 and not np.isclose(overlap_end, 1.0):
                        end_ray_intersects = True
                    if start_ray_intersects and end_ray_intersects:
                        break
                    continue

                intersection = self.line_segment_intersection(line, other_start, other_end)
                if intersection is None:
                    continue

                side_position = np.dot(np.asarray(intersection) - start, side_vector) / side_length_squared
                if side_position < 0.0 and not np.isclose(side_position, 0.0):
                    start_ray_intersects = True
                elif side_position > 1.0 and not np.isclose(side_position, 1.0):
                    end_ray_intersects = True

                if start_ray_intersects and end_ray_intersects:
                    break

            if not start_ray_intersects:
                exterior_points.append(start)
            if not end_ray_intersects:
                exterior_points.append(end)

        if not exterior_points:
            return 0.0, np.empty((0, 2))

        exterior_points = np.unique(np.asarray(exterior_points), axis=0)
        if len(exterior_points) < 3:
            return 0.0, exterior_points

        exterior_points = self.rotational_sort(exterior_points)
        crown_area = Polygon(exterior_points).area
        return crown_area, exterior_points

    def calculate_crown_area_with_gaps(self, vertices, faces):
        """
        Calculate the crown area of a vegetation object given its vertices and faces.
        """
        leaf_polygons = []
        vertices = np.array(vertices)[:,:2]
        for f,face in enumerate(faces):
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
        Order points by counter-clockwise angle, preferring nearby next points.
        """
        points = np.asarray(points)
        if len(points) < 2:
            return points

        centroid = np.mean(points, axis=0)
        offsets = points - centroid
        angles = np.arctan2(offsets[:, 1], offsets[:, 0])
        extents = np.ptp(points, axis=0)
        distance_threshold = 0.25 * np.min(extents)
        order = [int(np.argmin(angles))]
        remaining = set(range(len(points)))
        remaining.remove(order[0])

        while remaining:
            current = order[-1]
            candidates = np.array(sorted(remaining), dtype=int)
            angular_gaps = (angles[candidates] - angles[current]) % (2.0 * np.pi)
            distances = np.linalg.norm(points[candidates] - points[current], axis=1)
            nearby_candidates = candidates[distances <= distance_threshold]
            if len(nearby_candidates) > 0:
                candidates = nearby_candidates
                angular_gaps = (angles[candidates] - angles[current]) % (2.0 * np.pi)
            next_point = int(candidates[np.argmin(angular_gaps)])
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
