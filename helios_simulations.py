################################################################################################
#
# Code to create a Helios survey (taken from Wouter's code and adapted)
#
# Adaptation by N. Origo (05/08/2026)
################################################################################################

"""Generate HELIOS scenes and surveys from scan positions and spatial grids."""

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


def _obj_centroid(obj_path: str) -> np.ndarray:
    """Return the mean XYZ coordinate of all vertices in an OBJ mesh."""
    obj_path = str(Path(obj_path))
    verts = []
    with open(obj_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().split()
            if len(s) >= 4 and s[0] == "v":
                verts.append(np.asarray(s[1:3], dtype=float))
    if not verts:
        raise ValueError(f"No vertices found in OBJ file: {obj_path}")
    return np.mean(np.asarray(verts, dtype=float), axis=0)

def _obj_distance_to_point(obj_path: str, point: np.ndarray) -> float:
    """Euclidean distance from a scan position to an object centroid."""
    return float(np.linalg.norm(_obj_centroid(obj_path) - point))

def generate_scene_for_radius(
    stem_files: Iterable[str],
    leaf_files: Iterable[str],
    terrain_file: str,
    scan_position: tuple[float, float],
    radius: float,
    outpath: str,
    scene_name: str,
) -> list[tuple[int, str]]:
    """
    Build a HELIOS scene containing terrain plus all stem/leaf objects whose centroid is within
    a specified radius of the scan position.

    The terrain is always included unconditionally. Stem and leaf parts share the same part ID.

    Parameters:
    -----------
    stem_files : iterable of str
                 Paths to stem OBJ meshes.
    leaf_files : iterable of str
                 Paths to leaf OBJ meshes, matched to stems by filename.
    terrain_file : str
                   Path to the terrain OBJ mesh.
    scan_position : tuple of float
                    XY coordinates of the scan position.
    radius : float
             Maximum centroid distance for including a stem and its matching leaves.
    outpath : str
              Destination path for the generated scene XML.
    scene_name : str
                 Name assigned to the generated HELIOS scene.

    Returns:
    --------
    parts : list of tuple[int, str]
            Part identifiers and mesh paths included in the scene.

    Raises:
    -------
    ValueError
        If radius is negative.
    """
    if radius < 0:
        raise ValueError("radius must be >= 0.")

    scan_point = np.asarray(scan_position, dtype=float)
    stem_list = [str(Path(p)) for p in stem_files]
    leaf_list = [str(Path(p)) for p in leaf_files]
    leaf_lookup = {Path(p).name: p for p in leaf_list}

    parts: list[tuple[int, str]] = [(0, str(Path(terrain_file)))]
    next_id = 1

    for stem_path in stem_list:
        if _obj_distance_to_point(stem_path, scan_point) > radius:
            continue

        stem_name = Path(stem_path).name
        parts.append((next_id, stem_path))
        leaf_name = stem_name.replace("_mesh.obj", "_leaves.obj")
        leaf_path = leaf_lookup.get(leaf_name)
        if leaf_path is not None:
            parts.append((next_id, str(Path(leaf_path))))
        next_id += 1

    ET_root = ET.Element("document")
    scene = ET.SubElement(ET_root, "scene", {"id": str(scene_name), "name": str(scene_name)})
    for part_id, part_path in parts:
        part = ET.SubElement(scene, "part", {"id": str(part_id)})
        flt = ET.SubElement(part, "filter", {"type": "objloader"})
        ET.SubElement(
            flt,
            "param",
            {"type": "string", "key": "filepath", "value": str(part_path)},
        )
        ET.SubElement(flt, "param", {"type": "string", "key": "up", "value": "z"})

    tree = ET.ElementTree(ET_root)
    ET.indent(tree, space="    ")
    outpath = str(Path(outpath))
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    tree.write(outpath, encoding="UTF-8", xml_declaration=True)
    return parts

def generate_radius_scenes_from_csv(
    path_scanpos: str,
    stem_dir: str,
    leaf_dir: str,
    terrain_dir: str,
    radius: float,
    outdir: str,
    prefix: str = "scene",
    include_terrain: bool = True,
) -> list[str]:
    """
    Create one scene XML per scan position from a CSV file.

    Parameters:
    -----------
    path_scanpos : str
                   CSV path containing scan positions and orientation columns.
    stem_dir : str
               Directory containing stem OBJ files.
    leaf_dir : str
               Directory containing leaf OBJ files.
    terrain_dir : str
                  Directory containing terrain OBJ files.
    radius : float
             Radius used to select tree meshes around each scan position.
    outdir : str
             Directory where scene XML files are written.
    prefix : str, default='scene'
             Prefix used for generated scene names.
    include_terrain : bool, default=True
                      Whether terrain is included. False is not implemented.

    Returns:
    --------
    scene_paths : list of str
                  Paths to generated scene XML files.

    Raises:
    -------
    FileNotFoundError
        If the terrain directory contains no OBJ file.
    ValueError
        If required scan-position CSV columns are missing.
    NotImplementedError
        If include_terrain is False.
    """
    stem_files = sorted(str(Path(p)) for p in Path(stem_dir).glob("*.obj"))
    leaf_files = sorted(str(Path(p)) for p in Path(leaf_dir).glob("*.obj"))
    terrain_files = sorted(str(Path(p)) for p in Path(terrain_dir).glob("*.obj"))
    if not terrain_files:
        raise FileNotFoundError(f"No terrain OBJ files found in {terrain_dir}")
    terrain_file = terrain_files[0]

    df = pd.read_csv(path_scanpos)
    required = ["#Name", "X", "Y", "Z", "Roll", "Pitch", "Yaw"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in scan-pos CSV: {missing}")

    written = []
    outdir = str(Path(outdir))
    Path(outdir).mkdir(parents=True, exist_ok=True)

    for row in df.itertuples(index=False):
        scan_name = str(getattr(row, "#Name"))
        scan_pos = (float(row.X), float(row.Y))
        scene_name = f"{prefix}_{scan_name.replace(' ', '_')}"
        scene_path = str(Path(outdir) / f"{scene_name}.xml")
        if include_terrain:
            generate_scene_for_radius(
                stem_files=stem_files,
                leaf_files=leaf_files,
                terrain_file=terrain_file,
                scan_position=scan_pos,
                radius=float(radius),
                outpath=scene_path,
                scene_name=scene_name,
            )
        else:
            raise NotImplementedError("include_terrain=False is not currently supported for the radius scene builder.")
        written.append(scene_path)

    return written

def generate_radius_scenes_from_grid(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    x_step: float,
    y_step: float,
    stem_dir: str,
    leaf_dir: str,
    terrain_dir: str,
    radius: float,
    outdir: str,
    prefix: str = "scene",
    z_level: float = 0.0,
) -> list[str]:
    """
    Create one scene XML per grid point in a rectangular scan grid.

    Parameters:
    -----------
    xmin, xmax, ymin, ymax : float
                             Inclusive XY bounds of the grid.
    x_step, y_step : float
                     Positive spacing between adjacent grid points.
    stem_dir, leaf_dir, terrain_dir : str
                                     Directories containing the respective OBJ meshes.
    radius : float
             Radius used to select tree meshes around each grid point.
    outdir : str
             Directory where the generated scene XML files are written.
    prefix : str, default='scene'
             Prefix used for generated scene names.
    z_level : float, default=0.0
              Retained for compatibility; scene selection uses XY coordinates.

    Returns:
    --------
    scene_paths : list of str
                  Paths to generated scene XML files.

    Raises:
    -------
    ValueError
        If grid spacing is non-positive or bounds are reversed.
    """
    if x_step <= 0 or y_step <= 0:
        raise ValueError("x_step and y_step must be > 0.")
    if xmin > xmax or ymin > ymax:
        raise ValueError("Expected xmin <= xmax and ymin <= ymax.")

    x_values = np.arange(xmin, xmax + 0.5 * x_step, x_step)
    y_values = np.arange(ymin, ymax + 0.5 * y_step, y_step)

    written = []
    for y in y_values:
        for x in x_values:
            scene_name = f"{prefix}_{x:.3f}_{y:.3f}"
            scene_path = str(Path(outdir) / f"{scene_name}.xml")
            generate_scene_for_radius(
                stem_files=sorted(str(Path(p)) for p in Path(stem_dir).glob("*.obj")),
                leaf_files=sorted(str(Path(p)) for p in Path(leaf_dir).glob("*.obj")),
                terrain_file=sorted(str(Path(p)) for p in Path(terrain_dir).glob("*.obj"))[0],
                scan_position=(float(x), float(y)),
                radius=float(radius),
                outpath=scene_path,
                scene_name=scene_name,
            )
            written.append(scene_path)
    return written

def rebuild_survey_legs_from_scanpos(
    path_survey: str,
    path_scanpos: str,
    outpath: Optional[str] = None,
    scanposition_ids: Optional[Iterable[int]] = None,
    scanner_template: str = "tls",
) -> int:
    """
    Replace all existing <leg> nodes in a HELIOS survey XML using scan positions from CSV.


    Parameters:
    -----------
    path_survey : str
                  Source survey XML used as a template.
    path_scanpos : str
                   CSV containing #Name, X, Y, Z, Roll, Pitch, and Yaw columns.
    outpath : str or None, default=None
              Output survey XML path; None overwrites path_survey.
    scanposition_ids : iterable of int or None, default=None
                       Optional scan IDs parsed from ScanPos### names.
    scanner_template : str, default='tls'
                       Template assigned to each generated scannerSettings element.

    Returns:
    --------
    written : int
              Number of survey legs written.

    Raises:
    -------
    ValueError
        If the source has no survey element, required CSV columns are absent, or
        no scan positions remain after filtering.
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

        ET.SubElement(leg, "scannerMount", {"x": "0", "y": "0", "z": "0"})
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
    scanner_template: str="tls",
    name: str="",
    scene: str="",
    scene_name: str="",
    platform_file: str="",
    platform: str="",
    scanner_file: str="",
    scanner: str=""
) -> int:
    """
    Build a regular XY grid of scanner legs in a HELIOS survey XML.


    Each generated leg contains:
      - <platformSettings x="..." y="..." onGround="true" />
      - <scannerSettings template="..." />

    Parameters:
    -----------
    path_survey : str
                  Existing survey XML path used as template.
    outpath : str
              Destination path for the generated survey XML.
    xmin, xmax, ymin, ymax : float
                             Inclusive XY grid bounds when aligned with the step.
    x_step, y_step : float
                     Positive grid spacing along the corresponding axes.
    scanner_template : str, default='tls'
                       Template assigned to generated scannerSettings elements.
    name : str, default=''
           Optional survey name.
    scene, scene_name : str, default=''
                       Optional scene reference and scene identifier.
    platform_file, platform : str, default=''
                              Optional platform file and platform identifier.
    scanner_file, scanner : str, default=''
                             Optional scanner file and scanner identifier.

    Returns:
    --------
    written : int
              Number of survey legs written.

    Raises:
    -------
    ValueError
        If grid spacing is non-positive, bounds are reversed, or the XML has no
        survey element.
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

    for leg in list(survey.findall("leg")):
        survey.remove(leg)

    # change attributes in template if requested
    if platform_file!="" and platform!="":
        survey.set("platform", platform_file+"#"+platform)
    if scanner_file!="" and scanner!="":
        survey.set("scanner", scanner_file+"#"+scanner)
    if name!="":
        survey.set("name", name)
    if scene!="":
        survey.set("scene", scene+"#"+scene_name)
    
    # Remove existing legs and replace them with the generated grid.
    x_values = np.arange(xmin, xmax + 0.5 * x_step, x_step)
    y_values = np.arange(ymin, ymax + 0.5 * y_step, y_step)

    written = 0
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
            ET.SubElement(leg, "scannerMount", {"x": "0", "y": "0", "z": "0"})
            ET.SubElement(leg, "scannerSettings", {"template": scanner_template})
            written += 1

    ET.indent(tree, space="    ")
    Path(write_path).parent.mkdir(parents=True, exist_ok=True)
    tree.write(write_path, encoding="UTF-8", xml_declaration=True)
    return written

def rebuild_survey_leg_from_point(
    path_survey: str,
    outpath: str,
    x: float,
    y: float,
    scanner_template: str="tls",
    name: str="",
    scene: str="",
    scene_name: str="",
    platform_file: str="",
    platform: str="",
    scanner_file: str="",
    scanner: str=""
) -> int:
    """
    Build a scanner leg in a HELIOS survey XML based on x, y location.


    Each generated leg contains:
      - <platformSettings x="..." y="..." onGround="true" />
      - <scannerSettings template="..." />

    Parameters:
    -----------
    path_survey : str
                  Existing survey XML path used as template.
    outpath : str
              Destination path for the generated survey XML.
    x, y : float
           XY coordinates for the single scanner leg.
    scanner_template : str, default='tls'
                       Template assigned to the generated scannerSettings element.
    name : str, default=''
           Optional survey name.
    scene, scene_name : str, default=''
                       Optional scene reference and scene identifier.
    platform_file, platform : str, default=''
                              Optional platform file and platform identifier.
    scanner_file, scanner : str, default=''
                             Optional scanner file and scanner identifier.

    Returns:
    --------
    None
        Writes the survey XML with one generated leg.

    Raises:
    -------
    ValueError
        If the source XML has no survey element.
    """
    path_survey = str(Path(path_survey))
    write_path = str(Path(outpath))

    tree = ET.parse(path_survey)
    root = tree.getroot()
    survey = root.find("survey")
    if survey is None:
        raise ValueError("No <survey> element found in XML.")

    for leg in list(survey.findall("leg")):
        survey.remove(leg)

    # change attributes in template if requested
    if platform_file!="" and platform!="":
        survey.set("platform", platform_file+"#"+platform)
    if scanner_file!="" and scanner!="":
        survey.set("scanner", scanner_file+"#"+scanner)
    if name!="":
        survey.set("name", name)
    if scene!="":
        survey.set("scene", scene+"#"+scene_name)
    
    # Remove existing legs and replace them with the specified point.
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
    ET.SubElement(leg, "scannerMount", {"x": "0", "y": "0", "z": "0"})
    ET.SubElement(leg, "scannerSettings", {"template": scanner_template})

    ET.indent(tree, space="    ")
    Path(write_path).parent.mkdir(parents=True, exist_ok=True)
    tree.write(write_path, encoding="UTF-8", xml_declaration=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a survey file defining a simulation in Helios ++."
    )
    parser.add_argument(
        "outfile",
        type=str,
        help=("Output path for the rebuilt survey XML. If split is used then this is a "
              "prefix and the output will be written to multiple files as a prefix, the"
              " next bit will be the scan number or the grid location (X and Y)."
        )
    )
    parser.add_argument(
        "--stem_dir",
        type=str,
        default=None,
        help="Directory containing stem OBJ meshes (used when splitting scenes by scan radius)."
    )
    parser.add_argument(
        "--leaf_dir",
        type=str,
        default=None,
        help="Directory containing leaf OBJ meshes (used when splitting scenes by scan radius)."
    )
    parser.add_argument(
        "--terrain_dir",
        type=str,
        default=None,
        help="Directory containing terrain OBJ meshes (used when splitting scenes by scan radius)."
    )
    parser.add_argument(
        "--scan_pos",
        type=str,
        help="Path to scan position CSV with columns #Name, X, Y, Z, Roll, Pitch, Yaw."
    )
    parser.add_argument(
        "--grid_pos",
        nargs=6,
        type=float,
        help="Grid settings in terms of xmin, xmax, ymin, ymax, xstep, ystep."
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
        action=argparse.BooleanOptionalAction,
        help=("If set to true then the simulations start running. Only recommend if you know that the computer"
              " has the memory resources to complete the simulation."
        )
    )
    parser.add_argument(
        "--run_simulations_slurm",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help=("If set to true then the simulations start running on a slurm cluster. Only recommend if you know"
             " that the cluster has the memory resources to complete the simulation."
        )
    )
    parser.add_argument(
        "--slurm_args",
        type=str,
        help="If you want to run the simulations on a slurm cluster then you must specify the slurm arguments."
    )
    parser.add_argument(
        "--split",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help=("If set to true then the scan positions are split into radial sections. This is useful if the whole"
             "scene does not fit into memory. If this is set to true then the radius must be specified."
        )
    )
    parser.add_argument(
        "--radius",
        type=float,
        help=("If you want to split the scan positions into radial sections then you must specify the radius of the scan"
             " in scene units. This is only used if the split option is set to true."
        )
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Name attribute for the survey XML. Defaults to the values in the survey template."
    )
    parser.add_argument(
        "--scene",
        type=str,
        help=("The scene file attribute for the survey XML. This should be the path to the scene file."
              " Defaults to the values in the survey template.")
    )
    parser.add_argument(
        "--scene_name",
        type=str,
        help="The scene name given in the scene file. Defaults to the values in the survey template."
    )
    parser.add_argument(
        "--upright",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="If set to true then the scans are done in upright mode."
    )
    parser.add_argument(
        "--tilt",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="If set to true then the scans are done in tilt mode."
    )

    args = parser.parse_args()

    outfile  = args.outfile
    if not args.path_survey_template:
        if not os.path.isfile("survey_template.xml"):
            raise IOError("If you don't specify a template file then one must exist in the directory this code was executed in."
                          " This should be called 'survey_template.xml'.")
    survey_template = args.path_survey_template

    helios_dir = "/".join(os.environ["HELIOS"].split("/")[:-2])+"/helios/data/scanners_and_platforms/"

    if args.scan_pos and args.grid_pos:
        raise IOError("Can only use one out of 'scan_pos' and 'grid_pos'.")
    if not args.scan_pos and not args.grid_pos:
        raise IOError("Must define one of 'scan_pos' or 'grid_pos'.")

    if args.name:
        name = args.name
    else:
        name = ""
    if args.scene:
        scene = args.scene
    else:
        scene = ""
    if args.scene_name:
        scene_name = args.scene_name
    else:
        scene_name = ""

    if args.scan_pos and not args.split: 
        scan_pos = args.scan_pos
        if args.upright:
            n_legs = rebuild_survey_legs_from_scanpos(
                path_survey=survey_template,
                path_scanpos=scan_pos,
                outpath=outfile,
                platform_file=helios_dir+"platforms.xml",
                platform="tripod",
                scanner_file=helios_dir+"scanners_tls.xml",
                scanner="riegl_vz400i",
                name=name,
                scene=scene,
                scene_name=scene_name
            )
        if args.tilt:
            n_legs = rebuild_survey_legs_from_scanpos(
                path_survey=survey_template,
                path_scanpos=scan_pos,
                outpath="/".join(outfile.split("/")[:-1])+"/"+".".join(outfile.split("/")[-1].split(".")[:-1])+"_tilt.xml",
                platform_file=helios_dir+"platforms.xml",
                platform="tripod_tilt",
                scanner_file=helios_dir+"scanners_tls.xml",
                scanner="riegl_vz400i",
                name=name,
                scene=scene,
                scene_name=scene_name
            )
        
    if args.grid_pos and not args.split: 
        grid_pos = args.grid_pos
        if args.upright:
            n_legs = rebuild_survey_legs_from_grid(
                path_survey=survey_template,
                outpath=outfile,
                xmin=grid_pos[0],
                xmax=grid_pos[1],
                ymin=grid_pos[2],
                ymax=grid_pos[3],
                x_step=grid_pos[4],
                y_step=grid_pos[5],
                platform_file=helios_dir+"platforms.xml",
                platform="tripod",
                scanner_file=helios_dir+"scanners_tls.xml",
                scanner="riegl_vz400i",
                name=name,
                scene=scene,
                scene_name=scene_name
            )
        if args.tilt:
            n_legs = rebuild_survey_legs_from_grid(
                path_survey=survey_template,
                outpath="/".join(outfile.split("/")[:-1])+"/"+".".join(outfile.split("/")[-1].split(".")[:-1])+"_tilt.xml",
                xmin=grid_pos[0],
                xmax=grid_pos[1],
                ymin=grid_pos[2],
                ymax=grid_pos[3],
                x_step=grid_pos[4],
                y_step=grid_pos[5],
                platform_file=helios_dir+"platforms.xml",
                platform="tripod_tilt",
                scanner_file=helios_dir+"scanners_tls.xml",
                scanner="riegl_vz400i",
                name=name,
                scene=scene,
                scene_name=scene_name
            )

    if args.radius:
        radius = args.radius

    if args.split and args.scan_pos:
        if not args.radius:
            raise IOError("If you want to split the scan positions "
                          "into radial sections, you must "
                          "specify the radius of the scan.")
        if not args.stem_dir or not args.leaf_dir or not args.terrain_dir:
            raise IOError("For split mode with scan positions, you must provide --stem_dir, --leaf_dir, and --terrain_dir.")

        scan_pos = args.scan_pos
        scene_dir = Path(outfile).parent / f"{Path(outfile).stem}_scene_parts"
        scene_dir.mkdir(parents=True, exist_ok=True)
        scene_paths = generate_radius_scenes_from_csv(
            path_scanpos=scan_pos,
            stem_dir=args.stem_dir,
            leaf_dir=args.leaf_dir,
            terrain_dir=args.terrain_dir,
            radius=radius,
            outdir=str(scene_dir)
        )

        survey_dir = Path(outfile).parent / f"{Path(outfile).stem}_surveys"
        survey_dir.mkdir(parents=True, exist_ok=True)

        scan_df = pd.read_csv(scan_pos)
        scan_ids = [int(_scan_id_from_name(str(name))) for name in scan_df["#Name"]]
        ######### change this to only have one scan location per survey file, i.e. one leg per survey file (can be upright and tilted)
        for scan_id, scene_path in zip(scan_ids, scene_paths):
            survey_path = survey_dir / f"{Path(scene_path).stem}_survey.xml"
            if args.upright and scan_id % 2 == 0:
                rebuild_survey_legs_from_scanpos(
                    path_survey=survey_template,
                    path_scanpos=scan_pos,
                    outpath=str(survey_path),
                    scanposition_ids=[scan_id],
                    platform_file=helios_dir+"platforms.xml",
                    platform="tripod",
                    scanner_file=helios_dir+"scanners_tls.xml",
                    scanner="riegl_vz400i",
                    name=scene_path.stem,
                    scene=scene_path,
                    scene_name=scene_path.stem
                )
            if args.tilt and scan_id % 2 == 1:
                rebuild_survey_legs_from_scanpos(
                    path_survey=survey_template,
                    path_scanpos=scan_pos,
                    outpath="/".join(str(survey_path).split("/")[:-1])+"/"+".".join(str(survey_path).split("/")[-1].split(".")[:-1])+"_tilt.xml",
                    scanposition_ids=[scan_id],
                    platform_file=helios_dir+"platforms.xml",
                    platform="tripod_tilt",
                    scanner_file=helios_dir+"scanners_tls.xml",
                    scanner="riegl_vz400i",
                    name=scene_path.stem,
                    scene=scene_path,
                    scene_name=scene_path.stem
                )

    if args.split and args.grid_pos:
        if not args.radius:
            raise IOError("If you want to split the grid "
                          "into radial sections, you must "
                          "specify the radius of the scan.")
        if not args.stem_dir or not args.leaf_dir or not args.terrain_dir:
            raise IOError("For split mode with a grid, you must provide --stem_dir, --leaf_dir, and --terrain_dir.")

        grid_pos = args.grid_pos
        scene_dir = Path(outfile).parent / f"{Path(outfile).stem}_scene_parts"
        scene_dir.mkdir(parents=True, exist_ok=True)
        scene_paths = generate_radius_scenes_from_grid(
            xmin=grid_pos[0],
            xmax=grid_pos[1],
            ymin=grid_pos[2],
            ymax=grid_pos[3],
            x_step=grid_pos[4],
            y_step=grid_pos[5],
            stem_dir=args.stem_dir,
            leaf_dir=args.leaf_dir,
            terrain_dir=args.terrain_dir,
            radius=radius,
            outdir=str(scene_dir)
        )

        survey_dir = Path(outfile).parent / f"{Path(outfile).stem}_surveys"
        survey_dir.mkdir(parents=True, exist_ok=True)

        for scene_path in scene_paths:
            scene_name = Path(scene_path).stem
            survey_path = survey_dir / f"{scene_name}_survey.xml"
            if args.upright:
                rebuild_survey_leg_from_point(
                    path_survey=survey_template,
                    outpath=str(survey_path),
                    x=float(scene_name.split('_')[-2]),
                    y=float(scene_name.split('_')[-1]),
                    name=scene_name,
                    scene=scene_path,
                    scene_name=scene_name,
                    platform_file=helios_dir+"platforms.xml",
                    platform="tripod",
                    scanner_file=helios_dir+"scanners_tls.xml",
                    scanner="riegl_vz400i"
                )
            if args.tilt:
                rebuild_survey_leg_from_point(
                    path_survey=survey_template,
                    outpath="/".join(str(survey_path).split("/")[:-1])+"/"+".".join(str(survey_path).split("/")[-1].split(".")[:-1])+"_tilt.xml",
                    x=float(scene_name.split('_')[-2]),
                    y=float(scene_name.split('_')[-1]),
                    name=scene_name,
                    scene=scene_path,
                    scene_name=scene_name,
                    platform_file=helios_dir+"platforms.xml",
                    platform="tripod_tilt",
                    scanner_file=helios_dir+"scanners_tls.xml",
                    scanner="riegl_vz400i"
                )

    if args.run_simulations:
        if not args.split:
            if args.upright and args.tilt:
                outfiles = [outfile, "/".join(outfile.split("/")[:-1])+"/"+".".join(outfile.split("/")[-1].split(".")[:-1])+"_tilt.xml"]
            elif args.upright:
                outfiles = [outfile]
            elif args.tilt:
                outfiles = ["/".join(outfile.split("/")[:-1])+"/"+".".join(outfile.split("/")[-1].split(".")[:-1])+"_tilt.xml"]
            for outfile in outfiles:
                subprocess.run([os.environ['HELIOS'], 
                                outfile, 
                                "--lasOutput", 
                                "--output", 
                                "/".join(outfile.split("/")[:-1])+"/output/"])
        else:
            for survey_file in Path(survey_dir).glob("*.xml"):
                subprocess.run([os.environ['HELIOS'], 
                                survey_file, 
                                "--lasOutput", 
                                "--output", 
                                "/".join(outfile.split("/")[:-1])+"/output/"])

    if args.run_simulations_slurm:
        if not args.slurm_args:
            raise IOError("If you want to run the simulations on a slurm cluster"
                          " then you must specify the slurm arguments.")
        if not args.split:
            if args.upright and args.tilt:
                outfiles = [outfile, "/".join(outfile.split("/")[:-1])+"/"+".".join(outfile.split("/")[-1].split(".")[:-1])+"_tilt.xml"]
            elif args.upright:
                outfiles = [outfile]
            elif args.tilt:
                outfiles = ["/".join(outfile.split("/")[:-1])+"/"+outfile.split("/")[-1].split(".")[0]+"_tilt.xml"]
            for outfile in outfiles:
                subprocess.run(["sbatch", 
                                args.slurm_args, 
                                f"--output={".".join(outfile.split('/')[-1].split('.')[:-1])}.log", 
                                f"--job-name={".".join(outfile.split('/')[-1].split('.')[:-1])}",
                                f"--wrap={os.environ['HELIOS']} {outfile} --lasOutput --output {"/".join(outfile.split("/")[:-1])+"/output/"}"])
        else:
            for survey_file in Path(survey_dir).glob("*.xml"):
                subprocess.run(["sbatch",
                                args.slurm_args, 
                                f"--output={".".join(survey_file.split('/')[-1].split('.')[:-1])}.log", 
                                f"--job-name={".".join(survey_file.split('/')[-1].split('.')[:-1])}",
                                f"--wrap={os.environ['HELIOS']} {survey_file} --lasOutput --output {"/".join(outfile.split("/")[:-1])+"/output/"}"])