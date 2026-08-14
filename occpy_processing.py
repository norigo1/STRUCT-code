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

    def add_geotransform(self, fname, ulx, uly, lrx, lry, pix_resx, pix_resy):
        """
        Create new Tiff with geotransform so Helios knows which way the images should
        be used with respect to the scene.

        Parameters:
        -----------
        fname    : str
                   Filename of the input Tiff image to add the geotransform to (the method
                   creates a new Tiff image which has the '_gtform' at the end.
        ulx      : float
                   Upper left x coordinate
        uly      : float
                   Upper left y coordinate
        lrx      : float
                   Lower right x coordinate
        lry      : float
                   Lower right y coordinate
        pix_res  : float
                   Pixel resolution x
        pix_resy : float
                   Pixel resolution y
        """
        ds = gdal.Open(fname, gdal.GA_Updates)
        fname_spl = fname.split(".")
        fname_dst = "".join(fpath_spl[:-1])+"_gtform."+fpath_spl[-1]
        geotransform = gdal.Translate(
                           fname_dst, 
                           fname, 
                           outputBounds=(ulx, uly, lrx, lry), 
                           xRes=pix_resx, 
                           yRes=pix_resy
                       )
        geotransform = None

class Reader:

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

    @staticmethod
    def platform_xml(xml_file, write_to_scanpos=True):
        """
        Read the platform xml file given to Helios++ and return the scan positions

        Parameters:
        -----------
        xml_file         : str
                           Filepath to the platform xml file used in the Helios++ simulations
        write_to_scanpos : bool, default=True
                           Determines whether to write to a file that can be read by Occpy (True)
                           or if the scan positions are just returned (False). If the write option
                           is chosen then a file called 'ScanPositions.txt' is created in the same
                           directory as the xml file.

        Returns:
        --------
        scan_pos : np.arr, float
                   n x 3 array of floats containing the x, y and z locations of the scanner position
                   in each column
        """
        op = open(xml_file, "r").readlines()
        positions = []
        for line in op:
            nline = line.strip()
            if nline[:17]=="<platformSettings":
                pos = [x[3:-1] for x in nline.split()[1:4]]
                positions.append(pos)

        scan_pos = np.array(positions).astype(float)
        if write_to_scanpos:
            oofname = "ScanPositions.txt"
            if "/" in xml_file:
                ofname = "/".join(xml_file.split("/")[:-1]) + f"/{oofname}"
            else:
                ofname = oofname
            ### need to change scan_pos to a string formatted output
            np.savetxt(ofname, scan_pos)
        else:
            return scan_pos


def OccpyConfig(
                self, 
                root_folder, 
                laz_folder, 
                out_folder, 
                DTM_in, 
                DSM_in, 
                voxel_dim, 
                plot_dim, 
                scanpos, 
                lthresh=0,
                ppiter=10000000,
                ovox=False, 
                single_return=False, 
                is_mobile=False
               ):
    """
    Create the config file needed to run Occpy

    Parameters:
    -----------
    root_folder   : str
                    Occpy uses this as a relative folder for all other folders specified
    laz_folder    : str
                    Folder where the laz files are
    out_folder    : str
                    Folder in which to store the Occpy outputs
    DTM_in        : str
                    Filepath to the tiff containing the digital terrain model for the site
    DSM_in        : str
                    Filepath to the tiff containing the digital surface model for the site
    voxel_dim     : float
                    The size of the voxel to use. When setting this consider the scientific
                    requirements and computational demands. The smaller the voxel_dim, the 
                    greater the number of voxels and corresponding RAM requirements. Currently,
                    only cube voxels are supported.
    plot_dim      : list, float
                    n x 6 of floats giving the plot dimensions. Must be in the following order:
                    xmin, ymin, zmin, xmax, ymax, zmax
    scanpos       : str
                    Filename to the file containing the positions of the scan locations.
                    File should be ascii and have the following columns:
                    #Name,X,Y,Z,Roll,Pitch,Yaw,ID which are comma separated
    lthresh       : float, default=0
                    Lower threshold above the ground to exclude from occlusion mapping
    ppiter        : int, default=10000000
                    Number of points read in from the laz file in each iteration
    ovox          : bool, default=False
                    Whether to export the output voxel grid to ply, slow if set to True
    single_return : bool, default=False
                    Whether the data is single return or multi return
    is_mobile     : bool, default=False
                    Flag to tell Occpy that the platform moves and therefore has a trajectory  
    """
    tif_in = {'DTM': DTM_in, 'DSM': DSM_in}
    config = {
              "root_folder": root_folder,
              "laz_in": laz_in,
              "tif_in": tif_in,
              "out_dir": out_folder,
              "vox_dim": voxel_dim,
              "lower_threshold": lthresh,
              "points_per_iter": ppiter,
              "plot_dim": plot_dim,
              "ScanPos": scanpos,
              "output_voxels": ovox,
              "single_return": single_return,
              "is_mobile": is_mobile
             }
    return config
