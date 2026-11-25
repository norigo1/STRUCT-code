##############################################################
#
# Code to retrieve the occlusion map and subsequent profiles
# from Occpy based on LiDAR simulations or LiDAR data
#
# Written by N. Origo with contributions explicitly mentioned
#
##############################################################

"""Prepare mesh-derived rasters and run or inspect Occpy voxel outputs."""

import glob
import os

import numpy as np
import occpy
import plyfile
import scipy
from osgeo import gdal
from PIL import Image

from scene_io import read_obj_mesh


class Preparation:
    """Compatibility placeholder for Occpy data preparation."""

    def __init__(self):
        """Initialize the placeholder without preparing any data."""
        something = 0


class RunOccpy:
    """Run Occpy ray tracing and retain its normalized voxel outputs."""

    def __init__(self, config):
        """
        Do the Occpy ray tracing and store the outputs

        Parameters:
        -----------
        config : dict
                 Dictionary defining the Occpy ray tracing parameters
                 See OccpyConfig for how this is formatted
        """
        self.config = config
        self.occpy = occpy.OccPy(config=config)
        self.occpy.define_sensor_pos(
            path2file=config["ScanPos"],
            delimiter=",",
            hdr_scanpos_id="#Name",
            hdr_x="X",
            hdr_y="Y",
            hdr_z="Z",
            sens_pos_id_offset=0,
        )
        self.occpy.do_raytracing()

        results = occpy.util.normalize_occlusion_output(
            input_folder=os.path.join(config["root_folder"], config["out_dir"]),
            PlotDim=config["plot_dim"],
            vox_dim=config["vox_dim"],
            dtm_file=os.path.join(config["root_folder"], config["tif_in"]["DTM"]),
            dsm_file=os.path.join(config["root_folder"], config["tif_in"]["DSM"]),
            lower_threshold=config["lower_threshold"],
            output_voxels=config["output_voxels"],
        )
        (
            self.Nhit_norm,
            self.Nmiss_norm,
            self.Nocc_norm,
            self.Classification_norm,
            self.chm,
        ) = results

    def get_voxel_coords(self):
        """
        Return the coordinates of each voxel at minx, miny and minz, and maxx, maxy and maxz of that
        voxel

        Returns:
        --------
        mins : np.arr, float
               n x 3 numpy array containing the minx, miny and minz coordinates of the voxel
        maxs : np.arr, float
               n x 3 numpy array containing the maxx, maxy and maxz coordinates of the voxel
        """
        minx = np.arange(
            self.config["plot_dim"][0],
            self.config["plot_dim"][3],
            self.config["vox_dim"],
        )
        miny = np.arange(
            self.config["plot_dim"][1],
            self.config["plot_dim"][4],
            self.config["vox_dim"],
        )
        minz = np.arange(
            self.config["plot_dim"][2],
            self.config["plot_dim"][5],
            self.config["vox_dim"],
        )
        maxx = minx + self.config["vox_dim"]
        maxy = miny + self.config["vox_dim"]
        maxz = minz + self.config["vox_dim"]

        mins = np.column_stack((minx, miny, minz))
        maxs = np.column_stack((maxx, maxy, maxz))

        return mins, maxs

    def get_occluded_voxels(self):
        """
        Return the coordinates of the occluded voxels

        Classification is binary and based on:
          Bienert, A., Queck, R., Schmidt, A., 2010. Voxel space analysis of terrestrial
          laser scans in forests for wind field modeling. Int. Arch. Photogramm. Remote
          Sens. Spat. Inf. Sci. ISPRS Arch. 38, 92–97.

        Returns:
        --------
        occ_min : np.arr, float
                  n x 3 array of the minimums of the occluded voxel coordinates
        occ_max : np.arr, float
                  n x 3 array of the maximums of the occluded voxel coordinates
        """
        return self._filter_coordinates_by_category(3)

    def get_hit_voxels(self):
        """
        Return the coordinates of the hit voxels (i.e. with interactions)

        Classification is binary and based on:
          Bienert, A., Queck, R., Schmidt, A., 2010. Voxel space analysis of terrestrial
          laser scans in forests for wind field modeling. Int. Arch. Photogramm. Remote
          Sens. Spat. Inf. Sci. ISPRS Arch. 38, 92–97.

        Returns:
        --------
        hit_min : np.arr, float
                  n x 3 array of the minimums of the occluded voxel coordinates
        hit_max : np.arr, float
                  n x 3 array of the maximums of the occluded voxel coordinates
        """
        return self._filter_coordinates_by_category(1)

    def get_empty_voxels(self):
        """
        Return the coordinates of the empty voxels

        Classification is binary and based on:
          Bienert, A., Queck, R., Schmidt, A., 2010. Voxel space analysis of terrestrial
          laser scans in forests for wind field modeling. Int. Arch. Photogramm. Remote
          Sens. Spat. Inf. Sci. ISPRS Arch. 38, 92–97.

        Returns:
        --------
        emp_min : np.arr, float
                  n x 3 array of the minimums of the occluded voxel coordinates
        emp_max : np.arr, float
                  n x 3 array of the maximums of the occluded voxel coordinates
        """
        return self._filter_coordinates_by_category(2)

    def get_unobserved_voxels(self):
        """
        Return the coordinates of the unobserved voxels

        Classification is binary and based on:
          Bienert, A., Queck, R., Schmidt, A., 2010. Voxel space analysis of terrestrial
          laser scans in forests for wind field modeling. Int. Arch. Photogramm. Remote
          Sens. Spat. Inf. Sci. ISPRS Arch. 38, 92–97.

        Returns:
        --------
        uno_min : np.arr, float
                  n x 3 array of the minimums of the occluded voxel coordinates
        uno_max : np.arr, float
                  n x 3 array of the maximums of the occluded voxel coordinates
        """
        return self._filter_coordinates_by_category(4)

    def _filter_coordinates_by_category(self, num):
        """
        Private method to return the coordinates based on the filter

        Parameters:
        -----------
        num : int
              Number associated with the category:
                  1 = hit
                  2 = empty
                  3 = occluded
                  4 = unobserved

        Return:
        -------
        mins : np.arr, float
               n x 3 array of the minimums of the classified voxel coordinates
        maxs : np.arr, float
               n x 3 array of the maximums of the classified voxel coordinates
        """
        classif = self.Classification_norm
        xinds, yinds, zinds = np.where(classif == num)
        vmin_coords, vmax_coords = self.get_voxel_coords()

        mins = np.column_stack(
            (
                vmin_coords[:, 0][xinds],
                vmin_coords[:, 1][yinds],
                vmin_coords[:, 2][zinds],
            )
        )
        maxs = np.column_stack(
            (
                vmax_coords[:, 0][xinds],
                vmax_coords[:, 1][yinds],
                vmax_coords[:, 2][zinds],
            )
        )
        return mins, maxs


class ModelBoundaries:
    """Generate terrain and surface rasters from scene mesh points."""

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
        if mesh_dir[-1] == "/":
            pass
        else:
            mesh_dir += "/"

        if not terrain_fname:
            self.terrain = glob.glob(mesh_dir + f"terrain/*.{fmt}")[0]
        else:
            self.terrain = glob.glob(mesh_dir + f"terrain/" + terrain_fname)[0]

        self.leaves = glob.glob(mesh_dir + f"leaves/*.{fmt}")
        self.wood = glob.glob(mesh_dir + f"wood/*.{fmt}")
        self.snags = glob.glob(mesh_dir + f"snags/*.{fmt}")

    def create_DTM(self, fout):
        """
        Create a digital terrain model as a raster (GeoTiff) which is designed
        for use in Helios++

        Parameters:
        -----------
        fout : str
               Filepath to save the output tiff file to

        Returns:
        --------
        None
            Writes the terrain raster to fout.
        """
        terrain_points = Reader(self.terrain)
        npix = len(terrain_points) ** 0.5
        xmin, xmax = np.min(terrain_points[:, 0]), np.max(terrain_points[:, 0])
        ymin, ymax = np.min(terrain_points[:, 1]), np.max(terrain_points[:, 1])

        gridx, gridy = np.meshgrid(
            np.linspace(xmin, xmax, int(npix) + 1),
            np.linspace(ymin, ymax, int(npix) + 1),
        )
        grid_z1 = scipy.interpolate.griddata(
            terrain_points[:, :2], terrain_points[:, 2], (gridx, gridy), method="linear"
        )
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

        Returns:
        --------
        None
            Writes the surface raster to fout.
        """
        terrain_points = Reader(self.terrain)
        npix = len(terrain_points) ** 0.5
        xmin, xmax = np.min(terrain_points[:, 0]), np.max(terrain_points[:, 0])
        ymin, ymax = np.min(terrain_points[:, 1]), np.max(terrain_points[:, 1])

        leaf_points = []
        wood_points = []
        snag_points = []

        for leaf in self.leaves:
            leaf_points.append(Reader(leaf))

        for wood in self.wood:
            wood_points.append(Reader(wood))

        for snag in self.snags:
            snag_points.append(Reader(snag))

        all_points = np.vstack(
            (terrain_points, leaf_points + wood_points + snag_points)
        )
        max_vals = scipy.stats.binned_statistic_2d(
            all_points[:, 0],
            all_points[:, 1],
            all_points[:, 2],
            statistic="max",
            bins=(int(npix) + 1, int(npix) + 1),
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

        Returns:
        --------
        None
            Writes a georeferenced copy of the input raster.
        """
        ds = gdal.Open(fname, gdal.GA_Updates)
        fname_spl = fname.split(".")
        fname_dst = "".join(fname_spl[:-1]) + "_gtform." + fname_spl[-1]
        geotransform = gdal.Translate(
            fname_dst,
            fname,
            outputBounds=(ulx, uly, lrx, lry),
            xRes=pix_resx,
            yRes=pix_resy,
        )
        geotransform = None


class Reader:
    """Read mesh vertices and scanner positions from supported scene files."""

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
        vertices, _ = read_obj_mesh(obj_file)
        return np.asarray(vertices, dtype=float)

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
        points = np.column_stack(
            (
                op["vertex"]["x"],
                op["vertex"]["y"],
                op["vertex"]["z"],
            )
        )
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
        scan_pos : np.ndarray, float or None
                   When write_to_scanpos is False, n x 3 array with one XYZ
                   scan position per row. When True, writes ScanPositions.txt
                   beside xml_file and returns None.
        """
        with open(xml_file, "r", encoding="utf-8") as platform_file:
            op = platform_file.readlines()
        positions = []
        for line in op:
            nline = line.strip()
            if nline[:17] == "<platformSettings":
                pos = [x[3:-1] for x in nline.split()[1:4]]
                positions.append(pos)

        scan_pos = np.array(positions).astype(float)
        if write_to_scanpos:
            header = "#Name,X,Y,Z,ID\n"
            oofname = "ScanPositions.txt"
            if "/" in xml_file:
                ofname = "/".join(xml_file.split("/")[:-1]) + f"/{oofname}"
            else:
                ofname = oofname
            scandid = [
                f"ScanPos{y:02d}," + ",".join(x) + f",{y}\n"
                for y, x in enumerate(positions)
            ]
            with open(ofname, "w", encoding="utf-8") as op:
                op.write(header)
                op.writelines(scandid)
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
    is_mobile=False,
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

    Returns:
    --------
    config : dict
             Configuration dictionary containing the supplied input paths,
             voxel settings, plot dimensions, and processing options.
    """
    tif_in = {"DTM": DTM_in, "DSM": DSM_in}
    config = {
        "root_folder": root_folder,
        "laz_in": laz_folder,
        "tif_in": tif_in,
        "out_dir": out_folder,
        "vox_dim": voxel_dim,
        "lower_threshold": lthresh,
        "points_per_iter": ppiter,
        "plot_dim": plot_dim,
        "ScanPos": scanpos,
        "output_voxels": ovox,
        "single_return": single_return,
        "is_mobile": is_mobile,
    }
    return config
