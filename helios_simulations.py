################################################################################################
#
# Code to create a Helios survey (taken from Wouter's code and adapted)
#
# Adaptation by N. Origo (05/08/2026)
################################################################################################

import os
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import pdb

def _scan_id_from_name(name: str) -> int:
    """Extract numeric scan ID from names like 'ScanPos002' -> 2."""
    match = re.fullmatch(r"\s*ScanPos(\d+)\s*", str(name))
    if not match:
        raise ValueError(
            f"Invalid scan position name '{name}'. Expected format like 'ScanPos002'."
        )
    return int(match.group(1))


def rebuild_survey_legs_from_scanpos(
    path_survey: str,
    path_scanpos: str,
    outpath: Optional[str] = None,
    scanposition_ids: Optional[Iterable[int]] = None,
    scanner_template: str = "tls",
) -> int:
    """
    Replace all existing <leg> nodes in a HELIOS survey XML using scan positions from CSV.


    Args:
        path_survey: Path to source survey XML file used as template.
        path_scanpos: Path to scan position CSV with columns #Name, X, Y, Z, Roll, Pitch, Yaw.
        outpath: Output path for the rebuilt survey XML. If None, overwrite path_survey.
        scanposition_ids: Optional iterable of IDs parsed from Name (ScanPos###).
            If None, all scan positions are used.
        scanner_template: scannerSettings template value (default: "tls").

    Returns:
        Number of <leg> entries written to the survey.
    """
    path_survey = str(Path(path_survey))
    path_scanpos = str(Path(path_scanpos))
    write_path = str(Path(outpath)) if outpath is not None else path_survey

    tree = ET.parse(path_survey)
    root = tree.getroot()
    survey = root.find("survey")
    if survey is None:
        raise ValueError("No <survey> element found in XML.")

    # Remove all existing legs so the survey is fully rebuilt from CSV.
    for leg in list(survey.findall("leg")):
        survey.remove(leg)

    df = pd.read_csv(path_scanpos)
    required_cols = ["#Name", "X", "Y", "Z", "Roll", "Pitch", "Yaw"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in scanpos CSV: {missing}")

    df["scan_id"] = df["#Name"].map(_scan_id_from_name)

    selected_ids = None if scanposition_ids is None else {int(v) for v in scanposition_ids}
    if selected_ids is not None:
        df = df[df["scan_id"].isin(selected_ids)]

    if df.empty:
        raise ValueError("No scan positions selected. Check provided scanposition_ids.")

    written = 0
    for row in df.itertuples(index=False):
        leg = ET.SubElement(survey, "leg")
        ET.SubElement(leg, "scannerSettings", {"template": scanner_template})
        ET.SubElement(
            leg,
            "platformSettings",
            {"x": f"{float(row.X):.6f}", "y": f"{float(row.Y):.6f}", "z": f"{float(row.Z):.6f}"},
        )

        scanner_mount = ET.SubElement(leg, "scannerMount", {"x": "0", "y": "0", "z": "0"})
        ET.SubElement(
            scanner_mount,
            "rot",
            {"axis": "x", "angle_deg": f"{float(row.Roll):.6f}"},
        )
        ET.SubElement(
            scanner_mount,
            "rot",
            {"axis": "y", "angle_deg": f"{float(row.Pitch):.6f}"},
        )
        ET.SubElement(
            scanner_mount,
            "rot",
            {"axis": "z", "angle_deg": f"{float(row.Yaw):.6f}"},
        )
        written += 1

    ET.indent(tree, space="    ")
    Path(write_path).parent.mkdir(parents=True, exist_ok=True)
    tree.write(write_path, encoding="UTF-8", xml_declaration=True)
    return written


def rebuild_survey_legs_from_grid(
    path_survey: str,
    outpath: str,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    x_step: float,
    y_step: float,
    upright: bool=True,
    tilt: bool=False,
    scanner_template: str = "tls",
) -> int:
    """
    Build a regular XY grid of scanner legs in a HELIOS survey XML.


    Each generated leg contains:
      - <platformSettings x="..." y="..." onGround="true" />
      - <scannerSettings template="..." />

    Args:
        path_survey: Existing survey XML path used as template.
        outpath: Output XML path to write.
        xmin, xmax, ymin, ymax: Grid bounds (inclusive when aligned with step).
        x_step, y_step: Positive grid spacing for x and y.
        scanner_template: scannerSettings template value.
        upright: 


    Returns:
        Number of <leg> entries written to the survey.
    """
    if x_step <= 0 or y_step <= 0:
        raise ValueError("x_step and y_step must be > 0.")
    if xmin > xmax or ymin > ymax:
        raise ValueError("Expected xmin <= xmax and ymin <= ymax.")

    path_survey = str(Path(path_survey))
    write_path = str(Path(outpath))

    tree = ET.parse(path_survey)
    root = tree.getroot()
    survey = root.find("survey")
    if survey is None:
        raise ValueError("No <survey> element found in XML.")

    # Remove existing legs and replace them with the generated grid.
    for leg in list(survey.findall("leg")):
        survey.remove(leg)

    pdb.set_trace()

    x_values = np.arange(xmin, xmax + 0.5 * x_step, x_step)
    y_values = np.arange(ymin, ymax + 0.5 * y_step, y_step)

    angles = []
    if upright:
        angles.append([0., 0., 0.])
    if tilt:
        angles.append([0., 90., 0.])

    written = 0
    for angle in angles:
        for y in y_values:
            for x in x_values:
                leg = ET.SubElement(survey, "leg")
                ET.SubElement(
                    leg,
                    "platformSettings",
                    {
                        "x": f"{float(x):.6f}",
                        "y": f"{float(y):.6f}",
                        "onGround": "true",
                    },
                )
                scanner_mount = ET.SubElement(leg, "scannerMount", {"x": "0", "y": "0", "z": "0"})
                ET.SubElement(
                    scanner_mount,
                    "rot",
                    {"axis": "x", "angle_deg": f"{float(angle[0]):.6f}"},
                )
                ET.SubElement(
                    scanner_mount,
                    "rot",
                    {"axis": "y", "angle_deg": f"{float(angle[1]):.6f}"},
                )
                ET.SubElement(
                    scanner_mount,
                    "rot",
                    {"axis": "z", "angle_deg": f"{float(angle[2]):.6f}"},
                )

                ET.SubElement(leg, "scannerSettings", {"template": scanner_template})
                written += 1

    ET.indent(tree, space="    ")
    Path(write_path).parent.mkdir(parents=True, exist_ok=True)
    tree.write(write_path, encoding="UTF-8", xml_declaration=True)
    return written


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a survey file defining a simulation in Helios ++."
    )
    parser.add_argument(
        "outfile",
        type=str,
        help="Output path for the rebuilt survey XML."
    )
    parser.add_argument(
        "--scan_pos",
        type=str,
        help="Path to scan position CSV with columns #Name, X, Y, Z, Roll, Pitch, Yaw."
    )
    parser.add_argument(
        "--grid_pos",
        nargs=8,
        type=float,
        help=("Grid settings in terms of xmin, xmax, ymin, ymax, xstep, ystep, upright, tilted."
              "upright and tilt should be 1 or 0 depending on whether the scans should be done"
              "in upright mode, tilt mode, or both - i.e. 1 0, 0 1, and 1 1 are the only options.")
    )
    parser.add_argument(
        "--path_survey_template", 
        type=str, 
        default="survey_template.xml", 
        help=("Template for the survey which can be a complete survey that has been used before. The method"
             " uses that and removes all data from it. Default is 'survey_template.xml', raises an error if"
             " this option is not used and the file does not exist.")
    )
    parser.add_argument(
        "--run_simulations",
        type=bool,
        default=False,
        help="If set to true then the simulations start running. Only recommend if you know that the computer"
             " has the memory resources to complete the simulation."
    )

    args = parser.parse_args()

    outfile  = args.outfile
    if not args.path_survey_template:
        if not os.path.isfile("survey_template.xml"):
            raise IOError("If you don't specify a template file then one must exist in the directory this code was executed in."
                          " This should be called 'survey_template.xml'.")
    survey_template = args.path_survey_template

    if args.scan_pos and args.grid_pos:
        raise IOError("Can only use one out of 'scan_pos' and 'grid_pos'.")
    if not args.scan_pos and not args.grid_pos:
        raise IOError("Must define one of 'scan_pos' or 'grid_pos'.")

    if args.scan_pos: 
        scan_pos = args.scan_pos
        n_legs = rebuild_survey_legs_from_scanpos(
            path_survey=survey_template,
            path_scanpos=scan_pos,
            outpath=outfile
        )
        
    if args.grid_pos: 
        grid_pos = args.grid_pos
        n_legs = rebuild_survey_legs_from_grid(
            path_survey=survey_template,
            outpath=outfile,
            xmin=grid_pos[0],
            xmax=grid_pos[1],
            ymin=grid_pos[2],
            ymax=grid_pos[3],
            x_step=grid_pos[4],
            y_step=grid_pos[5],
            upright=bool(grid_pos[6]),
            tilt=bool(grid_pos[7])
        )

    if args.run_simulations:
        subprocess.run(["$HELIOS", outfile, "--lasOutput"])

    print(f"Wrote {n_legs} grid legs to: {outfile}")
