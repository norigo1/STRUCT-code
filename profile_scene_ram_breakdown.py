#####################################################################
#
# Python code to profile the RAM breakdown of scene elements in a 
# Helios simulation - this is intended to provide information to aid
# splitting up the overall scene so it can fit into normal RAM
# requirements (ideally sub 100GB)
#
# Written by N. Origo (22/09/2026)
#
#####################################################################

import glob
import numpy as np
import psutil
import time
import sys
sys.path.append("Users/niallorigo/Research/UGent/SpaceTwin/progs/python/eradiate_tools/virtual-forest-radiative-transfer/helios/")
import mesh_to_helios_obj as m2h
import helios_simulations as hs

class LogRAM:

    def __init__(self, pid, logtime=5):
        """
        Class to log memory usage over the course a process

        Parameters:
        -----------
        pid     : int
                  The PID of the process to track.
        logtime : int
                  The number of seconds to wait before querying.
        """
        ram = self.get_ram_use(pid)
        self.start_time = time.time()
        ram_profile = []
        while ram!=0:
            ram = self.get_ram_use(pid)
            ram_profile.append(ram)
            time.sleep(logtime)
        
        self.end_time = time.time()
        self.max_ram = np.max(ram_profile)
        self.ram_profile = np.array(ram_profile)


    def get_ram_use(self, pid):
        """
        Gets the RAM usage based on the PID, returns 0 if the process
        ended.
        """
        try:
            ram = psutil.Process(pid).memory_info().rss / (1024**2)
        except psutil.NoSuchProcess:
            ram = 0
        return ram


class GetSpatials:

    def __init__(self, stem_dir, leaf_dir, terrain_dir):
        """
        Class to map the spatial details of each of the scene elements

        Parameters:
        -----------
        stem_dir    : str
                      Path to the directory containing the Helios obj files of the stems
        leaf_dir    : str
                      Path to the directory containing the Helios obj files of the leaves
        terrain_dir : str
                      Path to the directory containing the Helios obj files of the terrain
        """
        if stem_dir[-1]!="/": stem_dir+="/"
        if leaf_dir[-1]!="/": leaf_dir+="/"
        if terrain_dir[-1]!="/": terrain_dir+="/"
        stem_files = glob.glob(stem_dir+"*.obj")
        leaf_files = glob.glob(leaf_dir+"*.obj")
        terrain_file = glob.glob(terrain_dir+"*.obj")

        stemfs, stem_stats = self.get_spatial(stem_files)
        leaffs, leaf_stats = self.get_spatial(leaf_files)
        terrf,  terr_stats = self.get_spatial(terrain_file)

    def read_file(self, fname):
        """Read a standard Helios obj file and retrieve the vertex points"""
        op = open(fname).readlines()
        vertices = []
        for x in op:
            if x[0]=="v":
                xx = x[:-1]
                vertices.append(xx.split()[1:-4])

        return np.array(vertices).astype(float)
        
    def get_spatial(self, fnames):
        """Get spatial location and extent of directory of obj files"""
        vals = []
        for f in fnames:
            verts = self.read_file(f)
            mins = np.min(verts, axis=0)
            maxs = np.max(verts, axis=0)
            avgs = np.mean(verts, axis=0)
            vals.append(np.column_stack((mins, maxs, avgs)))

        vals = np.array(vals)
        return fnames, vals


class TrackSceneElementRAM:

    def __init__(self, stem_dir, leaf_dir, terrain_dir, survey_template):
        """
        Class to track the RAM of each scene element
        """
        if stem_dir[-1]!="/": stem_dir+="/"
        if leaf_dir[-1]!="/": leaf_dir+="/"
        if terrain_dir[-1]!="/": terrain_dir+="/"
        stem_files = glob.glob(stem_dir+"*.obj")
        leaf_files = glob.glob(leaf_dir+"*.obj")
        terrain_file = glob.glob(terrain_dir+"*.obj")

    def generate_scene(self, outdir, scene_name, obj):
        """
        Method to generate a scene containing just one obj

        Parameters:
        -----------
        outdir     : str
                     Output directory of the scene that is created here
        scene_name : str
                     Name of the scene, also used for the filename of the scene
        obj        : str
                     The object file to include in the scene

        Returns:
        --------
        scene_fname : str
                      The output filepath of the newly created scene
        """
        if outdir[-1]!="/": outdir+="/"
        scene_fname = scene_name + ".xml"

        m2h.generate_scene([obj], outdir+scene_fname, scene_name=scene_name)
        return outdir+scene_fname

    def generate_survey(self, outdir, template, scene, scene_name):
        """
        Generate the survey to run the helios simulation for each object

        Parameters:
        -----------
        outdir     : str
                     Output directory of the survey that is created here
        template   : str
                     Filename for the survey template
        scene      : str
                     Filename for the scene to simulate
        scene_name : str
                     Name of the scene, also used for the filename of the scene
        """
        if outdir[-1]!="/": outdir+="/"
        hm.rebuild_survey_legs_from_grid(template, 
                                         outdir+scene, 
                                         0, 1, 0, 1, 1, 1, 
                                         upright=True, 
                                         tilt=True,
                                         name=name, 
                                         scene=scene,
                                         scene_name=scene_name)
