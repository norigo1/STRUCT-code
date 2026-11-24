"""Scene and mesh readers shared by vegetation analysis modules."""

import pathlib
from xml.dom import minidom

import numpy as np


def read_obj_mesh(filename):
    """
    Read an OBJ mesh as vertices and one-based face indices.

    Parameters:
    -----------
    filename : str or pathlib.Path
               Path to the OBJ mesh.

    Returns:
    --------
    vertices : list of list of float
               XYZ coordinates for each mesh vertex.
    faces : list of list of int
            One-based vertex indices for each face.
    """
    vertices = []
    faces = []
    with open(filename, encoding="utf-8") as obj_file:
        for line in obj_file:
            if line.startswith("v "):
                vertices.append(list(map(float, line.strip().split()[1:]))[:3])
            elif line.startswith("f "):
                faces.append(list(map(int, line.strip().split()[1:])))
    return vertices, faces


def read_obj_facets(filename):
    """
    Read OBJ faces as an array of their vertex coordinates.

    Parameters:
    -----------
    filename : str or pathlib.Path
               Path to the OBJ mesh.

    Returns:
    --------
    facets : np.ndarray, float
             Face coordinates with shape (number of faces, vertices per face, 3).
    """
    vertices, faces = read_obj_mesh(filename)
    vertex_array = np.asarray(vertices, dtype=float)
    face_array = np.asarray(faces, dtype=int) - 1
    return vertex_array[face_array]


def read_obj(filename):
    """
    Read OBJ faces as vertex coordinates.

    Parameters:
    -----------
    filename : str or pathlib.Path
               Path to the OBJ mesh.

    Returns:
    --------
    facets : np.ndarray, float
             Face coordinates with shape (number of faces, vertices per face, 3).
    """
    return read_obj_facets(filename)


class SceneReader:
    """Read scene asset references and their OBJ or ASCII PLY meshes.

    Parameters:
    -----------
    scene_path : str or pathlib.Path
                 Path to the scene XML file.
    """

    def __init__(self, scene_path):
        """
        Initialize a reader for a scene XML file.

        Parameters:
        -----------
        scene_path : str or pathlib.Path
                     Path to the scene XML file.
        """
        self.scene_path = scene_path

    def resolve_filepath(self, filepath):
        """
        Resolve an asset path relative to the scene when necessary.

        Parameters:
        -----------
        filepath : str or pathlib.Path
                   Asset path stored in the scene XML.

        Returns:
        --------
        resolved_filepath : str
                            Existing absolute, working-directory, or scene-relative path.
        """
        path = pathlib.Path(filepath)
        if path.exists() or path.is_absolute():
            return str(path)
        scene_relative_path = pathlib.Path(self.scene_path).parent / path
        return str(scene_relative_path) if scene_relative_path.exists() else str(path)

    def read_scene(self):
        """
        Read file paths referenced by filepath parameters in the scene XML.

        Returns:
        --------
        filepaths : list of str
                    Asset paths in the same order as their XML parameters.
        """
        xml_tree = minidom.parse(self.scene_path)
        filepaths = []
        for param in xml_tree.getElementsByTagName("param"):
            if param.attributes["key"].value == "filepath":
                filepaths.append(param.attributes["value"].value)
        return filepaths

    def read_obj(self, filename):
        """
        Read an OBJ mesh and return vertices and one-based face indices.

        Parameters:
        -----------
        filename : str or pathlib.Path
                   Path to the OBJ mesh.

        Returns:
        --------
        vertices : list of list of float
                   XYZ coordinates for each mesh vertex.
        faces : list of list of int
                One-based vertex indices for each face.
        """
        return read_obj_mesh(filename)

    def read_ply(self, filename):
        """
        Read an ASCII PLY mesh and return vertices and one-based triangle faces.

        Parameters:
        -----------
        filename : str or pathlib.Path
                   Path to the ASCII PLY mesh.

        Returns:
        --------
        vertices : list of list of float
                   XYZ coordinates for each mesh vertex.
        faces : list of list of int
                One-based vertex indices for each triangular face.

        Raises:
        -------
        ValueError
            If the file is not ASCII PLY or contains unsupported or incomplete data.
        """
        vertices = []
        faces = []
        with open(filename, "r", encoding="utf-8") as ply_file:
            lines = ply_file.readlines()
        if not lines or lines[0].strip() != "ply":
            raise ValueError(f"Not a PLY file: {filename}")
        if not any(line.startswith("format ascii ") for line in lines):
            raise ValueError(f"Only ASCII PLY files are supported: {filename}")
        try:
            end_header = next(
                i for i, line in enumerate(lines) if line.strip() == "end_header"
            )
        except StopIteration as exc:
            raise ValueError(f"PLY file has no end_header: {filename}") from exc

        elements = []
        for line in lines[1:end_header]:
            fields = line.split()
            if len(fields) == 3 and fields[0] == "element":
                elements.append((fields[1], int(fields[2])))
        body = lines[end_header + 1 :]
        offset = 0
        for element_name, count in elements:
            records = body[offset : offset + count]
            if len(records) != count:
                raise ValueError(f"PLY element records are incomplete: {filename}")
            if element_name == "vertex":
                for record in records:
                    values = list(map(float, record.split()))
                    if len(values) < 3:
                        raise ValueError(
                            f"PLY vertex has fewer than three coordinates: {filename}"
                        )
                    vertices.append(values[:3])
            elif element_name == "face":
                for record in records:
                    values = list(map(int, record.split()))
                    if not values or values[0] != 3 or len(values) < 4:
                        raise ValueError(
                            f"Only triangular PLY faces are supported: {filename}"
                        )
                    faces.append([index + 1 for index in values[1:4]])
            offset += count
        return vertices, faces

    def read_tree(self, directories, globstring, hmin=2.0):
        """
        Read and concatenate matching tree meshes from supplied directories.

        Parameters:
        -----------
        directories : iterable of str or pathlib.Path
                      Directories searched for mesh files.
        globstring : str
                     Glob pattern used within each directory.
        hmin : float, default=2.0
               Retained for compatibility; no height filtering is currently applied.

        Returns:
        --------
        vertices : np.ndarray, float
                   Vertices concatenated across the matched meshes.
        faces : np.ndarray, int
                One-based face indices adjusted for the concatenated vertices.
        """
        vertl = []
        facel = []
        for directory in directories:
            if directory[-1] != "/":
                directory += "/"
            for filepath in pathlib.Path(directory).glob(globstring):
                if filepath.suffix == ".obj":
                    vertices, faces = self.read_obj(str(filepath))
                elif filepath.suffix == ".ply":
                    vertices, faces = self.read_ply(str(filepath))
                else:
                    raise ValueError(f"Unsupported file format: {filepath}")
                vertl.append(np.asarray(vertices))
                facel.append(np.asarray(faces))

        verto = np.vstack(vertl)
        faceo = np.vstack((facel[0], facel[1] + len(vertl[0])))
        return verto, faceo


class SceneParameters:
    """Read leaf and wood OBJ paths from virtual-forest scene XML.

    Parameters:
    -----------
    scene : str or pathlib.Path
            Path to the virtual-forest XML scene.
    """

    def __init__(self, scene):
        """
        Load a virtual-forest XML scene for path queries.

        Parameters:
        -----------
        scene : str or pathlib.Path
                Path to the XML scene.
        """
        with open(scene, "r", encoding="utf-8") as scene_file:
            self.op = scene_file.readlines()

    def get_leaf_files(self, idstring="/leaves/"):
        """
        Return leaf mesh paths matching the supplied path fragment.

        Parameters:
        -----------
        idstring : str, default='/leaves/'
                   Path fragment identifying leaf mesh references.

        Returns:
        --------
        filenames : list of str
                    Matching leaf mesh paths.
        """
        return self._get_files(idstring)

    def get_wood_files(self, idstring="/wood/"):
        """
        Return wood mesh paths matching the supplied path fragment.

        Parameters:
        -----------
        idstring : str, default='/wood/'
                   Path fragment identifying wood mesh references.

        Returns:
        --------
        filenames : list of str
                    Matching wood mesh paths.
        """
        return self._get_files(idstring)

    def _get_files(self, idstring):
        fnames = []
        for line in self.op:
            if idstring in line:
                fnames.append(line.strip().split()[-2].split("=")[-1][1:-1])
        return fnames

    def get_all_leaves(self, idstring="/leaves/"):
        """
        Return all referenced leaf faces as arrays of vertex coordinates.

        Parameters:
        -----------
        idstring : str, default='/leaves/'
                   Path fragment identifying leaf mesh references.

        Returns:
        --------
        facets : np.ndarray, float
                 Face coordinates stacked across the matched leaf meshes.
        """
        leaf_files = self.get_leaf_files(idstring)
        return np.vstack([read_obj_facets(filename) for filename in leaf_files])
