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

# class to map canopy cover spatially

# class to remove trees from a scene and create a new one

# class to remove leaves from trees in a scene and create a new scene

