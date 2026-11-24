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

"""Profile HELIOS process memory and associate usage with scene assets."""

import argparse
import glob
import os
import subprocess
import sys
import time
from abc import ABC, abstractmethod

import numpy as np
import psutil

sys.path.append(
    "/Users/niallorigo/Research/UGent/SpaceTwin/progs/python/eradiate_tools/virtual-forest-radiative-transfer/helios/"
)
import pdb

import mesh_to_helios_obj as m2h

import helios_simulations as hs


class Ram(ABC):
    """Abstract interface for tracking process RAM usage.

    Implementations report resident memory and sample it over time.
    """

    @staticmethod
    @abstractmethod
    def get_ram_use(pid):
        """
        Return the current resident memory used by a process.

        Parameters:
        -----------
        pid : int
              Process identifier to query.

        Returns:
        --------
        ram_use : float
                   Resident memory usage, in the units selected by the
                   implementation.
        """
        raise NotImplementedError

    @abstractmethod
    def profile(self, pid, logtime=5, max_samples=None):
        """
        Sample process memory over time.

        Parameters:
        -----------
        pid : int
              Process identifier to sample.
        logtime : float, default=5
                  Seconds between memory samples.
        max_samples : int or None, default=None
                      Maximum sample count; None allows sampling until process exit.

        Returns:
        --------
        profile : np.ndarray, float
                  Memory samples collected by the implementation.
        """
        raise NotImplementedError


class LogRAM(Ram):
    """Sample and store a process's resident memory usage."""

    def __init__(self, pid, logtime=5, max_samples=None):
        """
        Class to log memory usage over the course a process

        Parameters:
        -----------
        pid          : int
                       The PID of the process to track.
        logtime      : int
                       The number of seconds to wait before querying.
        max_samples  : int or None
                       Maximum number of samples to record. None means until
                       the process exits.
        """
        self.pid = pid
        self.logtime = logtime
        self.max_samples = max_samples
        self.start_time = time.time()
        self.end_time = None
        self.max_ram = 0.0
        self.ram_profile = np.array([], dtype=float)
        self.profile(pid, logtime=logtime, max_samples=max_samples)

    @staticmethod
    def get_ram_use(pid):
        """
        Get a process's resident memory usage.

        Parameters:
        -----------
        pid : int
              Process identifier to query.

        Returns:
        --------
        ram_use : float
                   Resident memory usage in decimal gigabytes, or zero when
                   the process is unavailable or access is denied.
        """
        try:
            ram = psutil.Process(pid).memory_info().rss / (1024**2)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            ram = 0.0
        return float(ram) * 0.001  # Convert to GB

    def profile(self, pid, logtime=5, max_samples=None):
        """
        Sample RAM usage until the process exits or the sample limit is reached.

        Parameters:
        -----------
        pid : int
              Process identifier to sample.
        logtime : float, default=5
                  Seconds between memory samples.
        max_samples : int or None, default=None
                      Maximum number of samples; None samples until exit.

        Returns:
        --------
        ram_profile : np.ndarray, float
                      Collected memory samples in decimal gigabytes.
        """
        ram_profile = []
        self.start_time = time.time()

        while True:
            ram = self.get_ram_use(pid)
            if ram <= 0.0 and ram_profile:
                break
            ram_profile.append(ram)

            if max_samples is not None and len(ram_profile) >= max_samples:
                break

            if ram <= 0.0:
                break

            time.sleep(logtime)

        self.end_time = time.time()
        self.ram_profile = np.asarray(ram_profile, dtype=float)
        self.max_ram = float(np.max(self.ram_profile)) if self.ram_profile.size else 0.0
        return self.ram_profile


class GetSpatials:
    """Read OBJ meshes and calculate their coordinate extents and centroids."""

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
        if stem_dir[-1] != "/":
            stem_dir += "/"
        if leaf_dir[-1] != "/":
            leaf_dir += "/"
        if terrain_dir[-1] != "/":
            terrain_dir += "/"
        stem_files = glob.glob(stem_dir + "*.obj")
        leaf_files = glob.glob(leaf_dir + "*.obj")
        terrain_file = glob.glob(terrain_dir + "*.obj")

        self.stemfs, self.stem_stats = self.get_spatial(stem_files)
        self.leaffs, self.leaf_stats = self.get_spatial(leaf_files)
        self.terrf, self.terr_stats = self.get_spatial(terrain_file)

    def read_file(self, fname):
        """
        Read vertex coordinates from a Helios OBJ file.

        Parameters:
        -----------
        fname : str
                Path to the OBJ file.

        Returns:
        --------
        vertices : np.ndarray, float
                   XYZ coordinates parsed from the mesh.
        """
        op = open(fname).readlines()
        vertices = []
        for x in op:
            if x[0] == "v":
                xx = x[:-1]
                vertices.append(xx.split()[1:-4])

        return np.array(vertices).astype(float)

    def get_spatial(self, fnames):
        """
        Calculate bounds and mean coordinates for each OBJ file.

        Parameters:
        -----------
        fnames : iterable of str
                 OBJ file paths to process.

        Returns:
        --------
        filenames : iterable of str
                    Input file paths.
        statistics : np.ndarray, float
                     Per file, columns contain minimum, maximum, and mean
                     coordinates for each axis.
        """
        vals = []
        for f in fnames:
            verts = self.read_file(f)
            mins = np.min(verts, axis=0)
            maxs = np.max(verts, axis=0)
            avgs = np.mean(verts, axis=0)
            vals.append(np.column_stack((mins, maxs, avgs)))

        vals = np.array(vals)
        return fnames, vals


class TrackSceneElementsRAM:
    """Run isolated HELIOS scenes to measure per-asset RAM use."""

    def __init__(
        self, stem_dir, leaf_dir, terrain_dir, survey_template, tempdir="./temp/"
    ):
        """
        Measure RAM usage for each stem, leaf, and terrain mesh.

        Parameters:
        -----------
        stem_dir : str
                   Directory containing stem OBJ files.
        leaf_dir : str
                   Directory containing leaf OBJ files.
        terrain_dir : str
                      Directory containing terrain OBJ files.
        survey_template : str
                          Path to the HELIOS survey XML template.
        tempdir : str, default='./temp/'
                  Directory used to store temporary scenes and surveys. It must
                  not already exist.

        Returns:
        --------
        None
            Stores per-asset RAM measurements in ``ram`` and asset paths in
            ``fnames``.
        """
        if stem_dir[-1] != "/":
            stem_dir += "/"
        if leaf_dir[-1] != "/":
            leaf_dir += "/"
        if terrain_dir[-1] != "/":
            terrain_dir += "/"
        self.stem_files = glob.glob(stem_dir + "*.obj")
        self.leaf_files = glob.glob(leaf_dir + "*.obj")
        self.terrain_file = glob.glob(terrain_dir + "*.obj")
        if not os.path.exists(tempdir):
            os.makedirs(tempdir)
        else:
            raise ValueError(
                f"Temp directory {tempdir} already exists, please remove it or specify a different tempdir."
            )

        counter = len(self.stem_files) + len(self.leaf_files) + len(self.terrain_file)
        print(f"Tracking RAM usage for {counter} scene elements...")

        stem_ram = []
        for ff, f in enumerate(self.stem_files):
            sname = f.split("/")[-1].split(".")[0]
            scene_fname, scene_name = self.generate_scene(
                tempdir, f"scene_stem_{sname}", self.terrain_file[0], f
            )
            survey = self.generate_survey(
                tempdir, survey_template, scene_fname, scene_name
            )
            stem_ram.append(self.run_simulation(survey))
            print(
                f"Processed {ff + 1}/{counter} stem files, RAM: {stem_ram[-1]:.2f} GB",
                end="\r",
            )

        leaf_ram = []
        for ff, f in enumerate(self.leaf_files):
            sname = f.split("/")[-1].split(".")[0]
            scene_fname, scene_name = self.generate_scene(
                tempdir, f"scene_leaf_{sname}", self.terrain_file[0], f
            )
            survey = self.generate_survey(
                tempdir, survey_template, scene_fname, scene_name
            )
            leaf_ram.append(self.run_simulation(survey))
            print(
                f"Processed {ff + 1 + len(self.stem_files)}/{counter} leaf files, RAM: {leaf_ram[-1]:.2f} GB",
                end="\r",
            )

        terrain_ram = []
        for ff, f in enumerate(self.terrain_file):
            sname = f.split("/")[-1].split(".")[0]
            scene_fname, scene_name = self.generate_scene(
                tempdir, f"scene_terrain_{sname}", f
            )
            survey = self.generate_survey(
                tempdir, survey_template, scene_fname, scene_name
            )
            terrain_ram.append(self.run_simulation(survey))
            print(
                f"Processed {ff + 1 + len(self.stem_files) + len(self.leaf_files)}/{counter} terrain files, RAM: {terrain_ram[-1]:.2f} GB",
                end="\r",
            )

        self.ram = np.vstack(
            (np.array(stem_ram), np.array(leaf_ram), np.array(terrain_ram)), axis=1
        )
        self.fnames = np.hstack(
            (
                np.array(self.stem_files),
                np.array(self.leaf_files),
                np.array(self.terrain_file),
            )
        )

    def generate_scene(self, outdir, scene_name, obj, terrain_obj):
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
        terrain_obj : str
                      The terrain object file to include in the scene

        Returns:
        --------
        scene_path : str
                     Generated scene XML path.
        scene_name : str
                     Scene name passed to the HELIOS scene builder.
        """
        outdir = os.path.abspath(outdir)
        os.makedirs(outdir, exist_ok=True)
        scene_fname = os.path.join(outdir, scene_name + ".xml")

        m2h.generate_scene(
            [[0, obj], [1, terrain_obj]], scene_fname, scene_name=scene_name
        )
        return scene_fname, scene_name

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

        Returns:
        --------
        survey_path : str
                      Path to the generated survey XML.
        """
        outdir = os.path.abspath(outdir)
        os.makedirs(outdir, exist_ok=True)
        ofile = os.path.join(outdir, scene_name + "_survey.xml")
        hs.rebuild_survey_legs_from_grid(
            template,
            ofile,
            0,
            1,
            0,
            1,
            1,
            1,
            upright=True,
            tilt=True,
            name=scene_name,
            scene=scene,
            scene_name=scene_name,
        )
        return ofile

    def run_simulation(self, survey):
        """
        Run the helios simulation for the given survey

        Parameters:
        -----------
        survey     : str
                     Filename for the survey to run

        Returns:
        --------
        max_ram : float
                  Maximum sampled RAM usage of the HELIOS process in decimal
                  gigabytes.
        """
        helios_bin = os.environ.get("HELIOS")
        if not helios_bin:
            raise RuntimeError(
                "HELIOS is not set in the environment. Export it first, e.g.: "
                "export HELIOS=/path/to/helios"
            )

        proc = subprocess.Popen(
            [helios_bin, survey, "--lasOutput"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        ram = LogRAM(proc.pid, logtime=10)
        return ram.max_ram


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile RAM usage of scene elements in Helios simulations."
    )
    parser.add_argument(
        "--stem_dir",
        type=str,
        required=True,
        help="Path to the directory containing the Helios obj files of the stems.",
    )
    parser.add_argument(
        "--leaf_dir",
        type=str,
        required=True,
        help="Path to the directory containing the Helios obj files of the leaves.",
    )
    parser.add_argument(
        "--terrain_dir",
        type=str,
        required=True,
        help="Path to the directory containing the Helios obj files of the terrain.",
    )
    parser.add_argument(
        "--survey_template",
        type=str,
        required=True,
        help="Path to the survey template XML file.",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        required=True,
        help="Output file to save the RAM usage results.",
    )
    parser.add_argument(
        "--tempdir",
        default="./temp/",
        type=str,
        required=False,
        help="Output directory to save temporary files.",
    )
    args = parser.parse_args()

    stem_dir = args.stem_dir
    leaf_dir = args.leaf_dir
    terrain_dir = args.terrain_dir
    survey_template = args.survey_template
    outfile = args.outfile
    tempdir = args.tempdir

    tracker = TrackSceneElementsRAM(
        stem_dir, leaf_dir, terrain_dir, survey_template, tempdir=tempdir
    )

    # clean up temp files
    subprocess.run(["rm", "-rf", tempdir])

    spatials = GetSpatials(stem_dir, leaf_dir, terrain_dir)
    spatial_f = spatials.stemfs + spatials.leaffs + [spatials.terrf[0]]
    spatial_stats = np.vstack(
        (spatials.stem_stats, spatials.leaf_stats, spatials.terr_stats), axis=0
    )

    with open(outfile, "w") as f:
        f.write("Filename, X, Y, Z, Max_RAM_Usage_GB\n")
        for fname, ram in zip(tracker.fnames, tracker.ram / 1000.0):
            file_stats = spatial_stats[np.where(spatial_f == fname)[0][0]]
            f.write(
                f"{fname},{file_stats[0]}, {file_stats[1]}, {file_stats[2]}, {ram}\n"
            )
