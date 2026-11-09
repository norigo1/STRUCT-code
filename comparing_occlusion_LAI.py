##################################################################
#
# Code to process the outputs of the occlusion mapping, analyse
# LAI/PAI from TLS, simulated TLS and 3D models
#
# Written by N. Origo (19/08/2026)
#
##################################################################

"""Analyze normalized occlusion grids and voxel-based leaf area index."""

import json

import numpy as np

from geometrical_calculations import LAICalculations
from scene_io import SceneParameters, read_obj


class OccpyAnalysis:
    """Load normalized Occpy grids and derive vertical voxel profiles."""

    def __init__(self, config):
        """
        Load normalized occlusion grids and their configuration.

        Parameters:
        -----------
        config : str
                 JSON-encoded configuration containing the output directory.
        """
        oconfig = json.loads(config)
        self.Nhit_norm = np.load(
            oconfig["out_dir"] + "merged_voxgrids/Nhit_norm.npy"
        )
        self.Nmiss_norm = np.load(
            oconfig["out_dir"] + "merged_voxgrids/Nmiss_norm.npy"
        )
        self.Nocc_norm = np.load(
            oconfig["out_dir"] + "merged_voxgrids/Nocc_norm.npy"
        )
        self.Classification_norm = np.load(
            oconfig["out_dir"] + "merged_voxgrids/Classification_norm.npy"
        )
        self.oconfig = oconfig

    def get_occluded_vertical_profile(self, ptype="percent"):
        """
        Retrieve the vertical profile of occluded voxels.

        Parameters:
        -----------
        ptype : str, default='percent'
                Profile values to return as 'percent' or 'absolute' counts.

        Returns:
        --------
        profile : np.ndarray, float
                  n x 2 array with voxel-center heights in the first column
                  and the occluded-voxel profile in the second.
        """
        return self._get_vertical_profile(3, ptype)

    def get_hit_vertical_profile(self, ptype="percent"):
        """
        Retrieve the vertical profile of hit voxels.

        Parameters:
        -----------
        ptype : str, default='percent'
                Profile values to return as 'percent' or 'absolute' counts.

        Returns:
        --------
        profile : np.ndarray, float
                  n x 2 array with voxel-center heights in the first column
                  and the hit-voxel profile in the second.
        """
        return self._get_vertical_profile(1, ptype)

    def get_empty_vertical_profile(self, ptype="percent"):
        """
        Retrieve the vertical profile of empty voxels.

        Parameters:
        -----------
        ptype : str, default='percent'
                Profile values to return as 'percent' or 'absolute' counts.

        Returns:
        --------
        profile : np.ndarray, float
                  n x 2 array with voxel-center heights in the first column
                  and the empty-voxel profile in the second.
        """
        return self._get_vertical_profile(2, ptype)

    def get_unobserved_vertical_profile(self, ptype="percent"):
        """
        Retrieve the vertical profile of unobserved voxels.

        Parameters:
        -----------
        ptype : str, default='percent'
                Profile values to return as 'percent' or 'absolute' counts.

        Returns:
        --------
        profile : np.ndarray, float
                  n x 2 array with voxel-center heights in the first column
                  and the unobserved-voxel profile in the second.
        """
        return self._get_vertical_profile(4, ptype)

    def _get_vertical_profile(self, num, ptype="percent"):
        """
        Retrieve the vertical profile for a classification ID.

        Parameters:
        -----------
        num   : int
                Classification ID:
                    1 = hit
                    2 = empty
                    3 = occluded
                    4 = unobserved
        ptype : str, default='percent'
                Profile values to return as 'percent' or 'absolute' counts.

        Returns:
        --------
        profile : np.ndarray, float
                  n x 2 array with voxel-center heights in the first column
                  and the selected classification profile in the second.

        Raises:
        -------
        IOError
            If ptype is neither 'percent' nor 'absolute'.
        """
        classif = self.Classification_norm
        zco = np.arange(
            self.oconfig["plot_dim"][2],
            self.oconfig["plot_dim"][5],
            self.oconfig["vox_dim"],
            dtype=float,
        )
        zco += self.oconfig["vox_dim"] / 2.0
        profile = np.sum(classif == num, axis=(0, 1)).astype(float)

        if ptype == "percent":
            profile /= classif.shape[0] * classif.shape[1]
            profile *= 100
        elif ptype != "absolute":
            raise IOError(
                f"{ptype} not a recognised option, pick from 'percent' or 'absolute'"
            )

        return np.column_stack((zco, profile))
