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
import shapely

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
        N.B. assumes all leaves have the same number of sides.

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
                 n x m x 3 numpy array of floats where n is the number of
                 facets and m is the number of sides the facet has

        Returns:
        --------
        vox_lai : float
                  The LAI contained within the voxel
        """
        in_voxel = (facets[:,:,0] >= xmin) & (facets[:,:,0] < xmax) & \
                   (facets[:,:,1] >= ymin) & (facets[:,:,1] < ymax) & \
                   (facets[:,:,2] >= zmin) & (facets[:,:,2] < zmax)

        # get area for leaves which are fully in the voxel while ignoring
        # leaves which are fully outside
        sum_in = np.sum(in_voxel, axis=(1,2))
        rel_facets = facets[sum_in>0, :, :]

        # work out their area
        areas_allin = self.polygon_area_3d(rel_facets[sum_in==9])

        # now sort the leaves which overlap with the voxel boundary,
        # replace the vertices outside the voxel with points on the
        # voxel where they intersect
        new_areas = []
        for j,facet in enumerate(rel_facets[sum_in<9]):
            # get the intersection points with the voxel plane
            facetmin1 = np.roll(facet, -1, axis=0)
            facetpls1 = np.roll(facet, 1, axis=0)
            ivj = in_voxel[j]
            ivjm = np.roll(ivj, -1, axis=0)
            ivjp = np.roll(ivj, 1, axis=0)

            # create the shapely bounding box
            

            new_verts = []
            for k,vertex in enumerate(ivj):
                if np.sum(vertex)==3:
                    new_vert.append(facet[k])
                else:
                    # look back
                    if np.sum(ivjm[k])==3:
                        ls = shapely.LineString([facetmin1[k], facet[k]])
                    if np.sum(ivjp[k])==3:
                        ls = shapely.LineString([facetpls1[k], facet[k]])
                    new_vert.append() 

            new_vert = np.array(new_vert)
            new_area.append(self.polygon_area_3d(new_vert))

        total_la = np.sum(areas_allin) + np.sum(new_area)
        ground_a = (xmax - xmin) * (ymax - ymin)
        vox_lai = total_la / ground_a 
        return vox_lai

    @staticmethod
    def get_voxel_normals(xmin, xmax, ymin, ymax, zmin, zmax):
        # get three points on the surface of each voxel side
        points = np.array([[[xmin, ymin, zmin],
                            [xmax, ymin, zmin],
                            [xmin, ymax, zmin]],
                           [[xmin, ymin, zmax],
                            [xmax, ymin, zmax],
                            [xmin, ymax, zmax]],
                           [[xmin, ymin, zmin],
                            [xmax, ymin, zmax],
                            [xmin, ymin, zmax]],
                           [[xmin, ymax, zmin],
                            [xmax, ymax, zmax],
                            [xmin, ymax, zmax]],
                           [[xmin, ymin, zmin],
                            [xmin, ymax, zmin],
                            [xmin, ymin, zmax]],
                           [[xmax, ymin, zmin],
                            [xmax, ymax, zmin],
                            [xmax, ymin, zmax]]])

        v1 = points[:,1] - points[:,0]
        v2 = points[:,2] - points[:,0]
        cp = np.cross(v1, v2)
        ncp = cp / np.linalg.norm(cp, axis=0)

    @staticmethod
    def intersect_segment_voxel_plane(A, B, P0, N, xmin, xmax, ymin, ymax, zmin, zmax):
        """
        Finds the intersection of a line segment AB with a plane.

        Adapted from Google AI.
    
        Parameters:
        -----------
        A    : np.arr, float
               3 x 1 numpy array of coordinates of the segment start point
        B    : np.arr, float
               3 x 1 numpy array of coordinates of the segment end point
        P0   : np.arr, float
               3 x 1 numpy array of a point on the plane
        N    : np.arr, float
               3 x 1 numpy array of the normal vector of the plane
        xmin : float
               Minimum of voxel along x axis
        xmax : float
               Maximum of voxel along x axis
        ymin : float
               Minimum of voxel along y axis
        ymax : float
               Maximum of voxel along y axis
        zmin : float
               Minimum of voxel along z axis
        zmax : float
               Maximum of voxel along z axis
    
        Returns:
        --------
        point : np.arr, float
                n x 3 array of the 3D intersection point of None if intersection
                is not in the voxel boundary
        """
        A = np.asarray(A)
        B = np.asarray(B)
        P0 = np.asarray(P0)
        N = np.asarray(N)
    
        V = B - A
        denominator = np.dot(V, N)
    
        # Calculate parameter t
        t = np.dot(P0 - A, N) / denominator
    
        # Check if the intersection point lies within the voxel boundary
        point = A + t * V
        if (point[0]>=xmin) & (point[0]<=xmax) & \
           (point[1]>=ymin) & (point[1]<=ymax) & \
           (point[2]>=zmin) & (point[2]<=zmax):
            return point
        
        return None

    @staticmethod
    def polygon_area_3d(vertices):
        """
        From Google AI
        """
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

    @staticmethod
    def clip_polygon_by_plane(vertices, plane_axis, plane_val, is_upper):
        """
        Clips an array of 3D vertices (N, 3) against a single axis-aligned voxel plane.

        Adapted from Google AI.
        """
        p1, p2 = vertices[i], next_vertices[i]
        # Linear interpolation parameter t
        t = (plane_val - p1[plane_axis]) / (p2[plane_axis] - p1[plane_axis])
        intersect = p1 + t * (p2 - p1)
        clipped_vertices.append(intersect)

        return np.array(clipped_vertices)


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
