#######################################################################
#
# Code to alter the canopy cover and leaf area index of a scene, 
# by changing the scene's vegetation parameters.
#
# Written by Niall Origo, 02/10/2026
#
######################################################################

import numpy as np
from xml.etree import ElementTree as ET
from scipy.spatial import ConvexHull
from scipy import interpolate as interp

# read in scene
class SceneReader:
    def __init__(self, scene_path):
        self.scene_path = scene_path

    def read_scene(self):
        """
        Read the scene from the specified path and return a Scene object.
        """
        xml_tree = ET.parse(self.scene_path)
        # get filepaths from the scene
        filepaths = [elem.text for elem in xml_tree.iter('filepath')]
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
                    vertices.append(list(map(float, line.strip().split()[1:])))
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
                    vertices.append(list(map(float, line.strip().split())))
        return vertices, faces


# class to calculate canopy cover
class CanopyCover:
    def __init__(self, scene):
        self.scene = scene
        sr = SceneReader(scene)
        self.filepaths = sr.read_scene()

    def calculate_canopy_cover(self):
        """
        Calculate the canopy cover of the scene by reading in the vertices and 
        faces of the scene's vegetation objects.
        """
        xy_area = []
        vertexx = []
        for filepath in self.filepaths:
            if filepath.endswith('.obj'):
                vertices, faces = SceneReader(self.scene).read_obj(filepath)
            elif filepath.endswith('.ply'):
                vertices, faces = SceneReader(self.scene).read_ply(filepath)
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
            
            # calculate canopy cover from vertices and faces using convex hull over x and y coordinates 
            cvh = ConvexHull(vertices)
            vertexx.append(vertices)
            xy_area.append(cvh.area)
        
        total_area = np.sum(xy_area)

        cvh = ConvexHull(np.concatenate(vertexx))
        ground_area = cvh.area
        return total_area, ground_area, np.array(xy_area)


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
        _, _, self.xy_area = cc.calculate_canopy_cover()
        mean_xyz = []
        for filepath in self.filepaths:
            if filepath.endswith('.obj'):
                vertices, _ = SceneReader(self.scene).read_obj(filepath)
            elif filepath.endswith('.ply'):
                vertices, _ = SceneReader(self.scene).read_ply(filepath)
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
            mean_xyz.append(np.mean(vertices, axis=0))

        self.locations = np.array(mean_xyz)[:, :2]


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
        self.total_area, self.ground_area, self.xy_area = cc.calculate_canopy_cover()
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

