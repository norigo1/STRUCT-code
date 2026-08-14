##############################################################
#
# Code to retrieve the occlusion map and subsequent profiles
# from Occpy based on LiDAR simulations or LiDAR data
#
# Written by N. Origo with contributions explicitly mentioned
#
##############################################################

import glob
import numpy as np
from osgeo import gdal
import plyfile
import scipy

from PIL import Image


class Preparation:

    def __init__(self):
        something = 0


class ModelBoundaries:

    def __init__(self, mesh_dir, terrain_fname=None, fmt="obj"):
        """
        Class to deal with calculating the scene vertical boundaries
        in the form that Helios requires (Tiff).

        Parameters:
        -----------
        mesh_dir      : str
                        Directory where the meshes of the model are stored.
                        The names expected are 'terrain', 'wood', 'leaves'
                        and 'snags' (wood without leaves).
        terrain_fname : str
                        Just the filename of the terrain mesh. If None (default) 
                        then it is assumed there is only one mesh file in the 
                        folder.
        fmt           : str; default='obj'
                        Format of the meshes: either 'obj' or 'ply'
        """
        self.fmt = fmt
        if mesh_dir[-1]=="/":
            pass
        else:
            mesh_dir+="/"

        if not terrain_fname:
            self.terrain = glob.glob(mesh_dir+f"terrain/*.{fmt}")[0]
        else:
            self.terrain = glob.glob(mesh_dir+f"terrain/"+terrain_fname)[0]

        self.leaves = glob.glob(mesh_dir+f"leaves/*.{fmt}")
        self.wood = glob.glob(mesh_dir+f"wood/*.{fmt}")
        self.snags = glob.glob(mesh_dir+f"snags/*.{fmt}")

    def create_DTM(self, fout):
        """
        Create a digital terrain model as a raster (GeoTiff) which is designed
        for use in Helios++

        Parameters:
        -----------
        fout : str
               Filepath to save the output tiff file to 
        """
        terrain_points = Reader(self.terrain)
        npix = len(terrain_points)**0.5
        xmin, xmax = np.min(terrain_points[:,0]), np.max(terrain_points[:,0])
        ymin, ymax = np.min(terrain_points[:,1]), np.max(terrain_points[:,1])
        
        gridx, gridy = np.meshgrid(
                           np.linspace(xmin, xmax, int(npix)+1), 
                           np.linspace(ymin, ymax, int(npix)+1)
                       )
        grid_z1 = scipy.interpolate.griddata(op[:,:2], op[:,2], (gridx, gridy), method='linear')
        im = Image.from_array(grid_z1)
        im.save(fout)

    def create_DSM(self, fout):
        """
        Create a digital surface model as a raster (GeoTiff) which is designed
        for use in Helios++

        Parameters:
        -----------
        fout : str
               Filepath to save the output tiff file to 
        """
        terrain_points = Reader(self.terrain)
        npix = len(terrain_points)**0.5
        xmin, xmax = np.min(terrain_points[:,0]), np.max(terrain_points[:,0])
        ymin, ymax = np.min(terrain_points[:,1]), np.max(terrain_points[:,1])

        leaf_points = []
        wood_points = []
        snag_points = []

        for leaf in self.leaves:
            leaf_points.append(Reader(leaf))

        for wood in self.wood:
            wood_points.append(Reader(wood))

        for snag in self.snags:
            snag_points.append(Reader(snag))

        all_points = np.vstack((terrain_points, leaf_points + wood_points + snag_points)) 
        max_vals = scipy.stats.binned_statistic_2d(
                       all_points[:,0], 
                       all_points[:,1], 
                       all_points[:,2],
                       statistic="max",
                       bins=(int(npix)+1, int(npix)+1)
                   )
        im = Image.fromarray(max_vals.statistic.T)
        im.save(fout)

    def add_geotransform(self, fname, ulx, uly, llx, lly, pix_res):
        """

        """
        ds = gdal.Open(fname, gdal.GA_Updates)
        fname_spl = fname.split(".")
        fname_dst = fname_spl[:-1]
        geotransform = gdal.Translate()
        # finish adding the geotransform

class Reader:

    def __init__(self, fname):
        """
        Class to read Helios scene files from ply or obj format.

        Parameters:
        -----------
        fname : str, or None
                Filepath of the scene object to read.
        """
        fmt = fname.split(".")[-1]
        if fmt=="obj":
            self.obj_reader(fname)
        elif fmt=="ply":
            self.ply_reader(fname)
        else:
            raise IOError(f"{fname} should be either '.obj' or '.ply'")

    @staticmethod
    def obj_reader(obj_file):
        """
        Read a Helios++ obj file and return the coordinates of all vertices

        Parameters:
        -----------
        obj_file : str
                   Path and filename of the obj file of interest

        Returns:
        --------
        points   : np.arr, float
                   n x 3 array of floats containing the vertices of all points
                   in the obj file
        """
        op = open(obj_file).readlines()
        points = []

        for j,line in enumerate(op):
            if line[0]=="v":
                points.append(line.split()[1:4])

        points = np.array(points).astype(float)
        return points

    @staticmethod
    def ply_reader(ply_file):
        """
        Read a ply file and return the coordinates of all vertices

        Parameters:
        -----------
        ply_file : str
                   Path and filename of the ply file of interest

        Returns:
        --------
        points   : np.arr, float
                   n x 3 array of floats containing the vertices of all points
                   in the ply file
        """
        op = plyfile.PlyData.read(ply_file)
        points = np.column_stack((op["vertex"]["x"], ["vertex"]["y"], ["vertex"]["z"]))
        return points
