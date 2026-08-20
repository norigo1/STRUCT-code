##################################################################
#
# Code to process the outputs of the occlusion mapping, analyse
# LAI/PAI from TLS, simulated TLS and 3D models
#
# Written by N. Origo (19/08/2026)
#
##################################################################

import json
import matplotlib.pyplot as plt
import numpy as np
import plyfile as ply

from occpy_processing import RunOccpy

class OccpyAnalysis:

    def __init__(self, config):
        """
        """
        oconfig = json.loads(config)
        self.Nhit_norm = np.load(
            oconfig["out_dir"]+"merged_voxgrids/Nhit_norm.npy"
        )
        self.Nmiss_norm = np.load(
            oconfig["out_dir"]+"merged_voxgrids/Nmiss_norm.npy"
        )
        self.Nocc_norm = np.load(
            oconfig["out_dir"]+"merged_voxgrids/Nocc_norm.npy"
        )
        self.Classification_norm = np.load(
            oconfig["out_dir"]+"merged_voxgrids/Classification_norm.npy"
        )
        self.oconfig = oconfig

    def get_occluded_vertical_profile(self, ptype="percent"):
        """
        Method to retrieve the occlusion vertical profile

        Parameters:
        -----------
        ptype : str, default='percent'
                The type of profile to retrieve, options are 'percent' or
                'absolute'

        Return:
        -------
        mins : np.arr, float
               n x 2 array of giving the height in the first column and the
               occlusion profile in the second.
        """
        return self._get_vertical_profile(3, ptype)

    def get_hit_vertical_profile(self, ptype="percent"):
        """
        Method to retrieve the hit vertical profile

        Parameters:
        -----------
        ptype : str, default='percent'
                The type of profile to retrieve, options are 'percent' or
                'absolute'

        Return:
        -------
        mins : np.arr, float
               n x 2 array of giving the height in the first column and the
               hit profile in the second.
        """
        return self._get_vertical_profile(1, ptype)

    def get_empty_vertical_profile(self, ptype="percent"):
        """
        Method to retrieve the empty voxel vertical profile

        Parameters:
        -----------
        ptype : str, default='percent'
                The type of profile to retrieve, options are 'percent' or
                'absolute'

        Return:
        -------
        mins : np.arr, float
               n x 2 array of giving the height in the first column and the
               empy voxel profile in the second.
        """
        return self._get_vertical_profile(2, ptype)

    def get_unobserved_vertical_profile(self, ptype="percent"):
        """
        Method to retrieve the unobserved vertical profile

        Parameters:
        -----------
        ptype : str, default='percent'
                The type of profile to retrieve, options are 'percent' or
                'absolute'

        Return:
        -------
        mins : np.arr, float
               n x 2 array of giving the height in the first column and the
               unobserved profile in the second.
        """
        return self._get_vertical_profile(4, ptype)

    def _get_vertical_profile(self, num, ptype="percent"):
        """
        Private method to retrieve the vertical profile of the specific
        classification ID (binary).

        Parameters:
        -----------
        num   : int
                Number associated with the category:
                    1 = hit
                    2 = empty
                    3 = occluded
                    4 = unobserved
        ptype : str, default='percent'
                The type of profile to retrieve, options are 'percent' or
                'absolute'

        Return:
        -------
        mins : np.arr, float
               n x 2 array of giving the height in the first column and the
               profile in the second.
        """
        zco = np.arange(
            self.oconfig["plot_dim"][2],
            self.oconfig["plot_dim"][5],
            self.oconfig["vox_dim"]
        )
        # shift vertical axis to centre of the voxel
        zco += self.oconfig["vox_dim"] / 2.
        profile = np.sum(classif==num,axis=(0,1))

        if ptype=="percent":
            profile/=(classif.shape[0]*classif.shape[1])
            profile*=100
        elif ptype=="absolute":
            pass
        else:
            raise IOError(f"{ptype} not a recognised option, pick from 'percent'"
                           " or 'absolute'")

        return np.column_stack((zco, profile))


class SceneParameters:

    def __init__(self, scene):
        """
        Class to retrieve parameters based on the virtual forest model
        
        Parameters:
        -----------
        scene : str
                Filepath to the xml file which defines the scene (.xml)
        """
        # open the scene and read (just using ascii for personal ease)
        self.op = open(scene, "r").readlines()

    def get_leaf_files(self, idstring="/leaves/"):
        """
        Get the leaf files associated with the scene

        Parameters:
        -----------
        idstring : str
                   Unique identifier to search for all leaf files

        Returns:
        --------
        fnames   : list
                   List of filenames corresponding to the leaf files
        """
        return self._get_files(idstring)

    def get_wood_files(self, idstring="/wood/"):
        """
        Get the wood files associated with the scene

        Parameters:
        -----------
        idstring : str
                   Unique identifier to search for all wood files

        Returns:
        --------
        fnames   : list
                   List of filenames corresponding to the wood files
        """
        return self._get_files(idstring)
        
    def _get_files(self, idstring):
        """
        Get the files that are part of the scene that are associated with 
        'idstring'

        Parameters:
        -----------
        idstring : str
                   Unique identifier to search for all files of a specific
                   category

        Returns:
        --------
        fnames   : list
                   List of filenames corresponding to the idstring
        """
        fnames = []
        for line in self.op:
            if idstring in line:
                fnames.append(line.strip().split()[-2].split("=")[-1][1:-1])

        return fnames

    def get_all_leaves(self, idstring="/leaves/"):
        """
        Get the coordinates of all leaves

        Parameters:
        -----------
        idstring : str, default="/leaves/"
                   The idstring that is used to pick out the leaf files in the
                   scene file.

        Returns:
        --------
        leaves   : np.arr, float
                   n x m x 3 array of floats giving the vertex coordinates of each
                   leaf. m is usually 3 but could be > 3 too.
        """
        lfnames = self.get_leaf_files(idstring)
        leaves = []
        for lf in lfnames:
            leaves.append(read_obj(lf))

        leaves = np.vstack(leaves)
        return leaves


class LAICalculations:

    def voxel_lai_from_facets(xmin, xmax, ymin, ymax, zmin, zmax, facets):
        """
        Return the LAI from the voxel given the leaves present in that voxel

        Parameters:
        -----------
        xmin   : float
                 Voxel minimum along the x axis
        xmax   : float
                 Voxel maximum along the x axis
        ymin   : float
                 Voxel minimum along the y axis
        ymax   : float
                 Voxel maximum along the y axis
        zmin   : float
                 Voxel minimum along the z axis
        zmax   : float
                 Voxel maximum along the z axis
        facets : np.arr, float
                 n x m x 3 numpy array of floats giving the 
        """
        in_voxel = (facets >= xmin) & (facets < xmax) & \
                   (facets >= ymin) & (facets < ymax) & \
                   (facets >= zmin) & (facets < zmax)

        # carry on here...


    def polygon_area_3d_numpy(vertices):
        # Ensure input is a NumPy array of shape (N, 3)
        v = np.asarray(vertices)
        if len(v) < 3:
            return 0.0
    
        # Get the next vertices in the sequence (rolled by -1)
        v_next = np.roll(v, -1, axis=0)
    
        # Compute cross products of sequential vertices: sum(v_i x v_next)
        cross_products = np.cross(v, v_next)
    
        # Sum the components and take half the magnitude
        normal_vector = np.sum(cross_products, axis=0)
        area = 0.5 * np.linalg.norm(normal_vector)
    
        return area



def read_obj(self, fname):
    """
    Read an obj file
    """
    op = open(fname).readlines()
    vertices = []
    facets = []
    for line in op:
        if line[0]=="v":
            vertices.append(line.split()[1:])
        if line[0]=="f":
            facets.append(line.split()[1:])

    facet_coords = []
    for facet in facets:
        face = []
        for corner in facet:
            face.append(vertices[corner])
        facet_coords.append(face)

    fcoords = np.array(facet_coords).astype(float)
    return fcoords
