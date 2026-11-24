########################################################
#
# Python script designed for running a simulation with
# slurm on a HPC
#
# Written by: N. Origo (03/06/2026)
########################################################

import argparse
import os
import pdb

parser = argparse.ArgumentParser()

## positional arguments
parser.add_argument(
    "canopy", type=str, help="File containing the canopy information - json format"
)
parser.add_argument(
    "background_surface",
    type=str,
    help="File containing the background surface information - json format",
)
parser.add_argument(
    "atmosphere",
    type=str,
    help="File containing the atmosphere information - json format",
)
parser.add_argument(
    "illumination",
    type=str,
    help="File containing the illumination information - json format",
)
parser.add_argument(
    "measure", type=str, help="File containing the measure information - json format"
)
parser.add_argument(
    "results_fname", type=str, help="String giving the name of the results file"
)
parser.add_argument(
    "project_dir",
    type=str,
    help=(
        "String giving the path to the directory containing the mesh and"
        " spectra dirs for the canopy being simulated"
    ),
)

options = parser.parse_args()

canopy = options.canopy
background_surface = options.background_surface
atmosphere = options.atmosphere
illumination = options.illumination
measure = options.measure
results_fname = options.results_fname
project_dir = options.project_dir


os.environ["ERADIATE_PROJECT_DIR"] = project_dir
import build_canopy as bc

rs = bc.RunSimulations(
    [canopy],
    [background_surface],
    [atmosphere],
    [illumination],
    [measure],
    [project_dir],
)
rs.run_simulations()
