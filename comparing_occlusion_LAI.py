##################################################################
#
# Code to process the outputs of the occlusion mapping, analyse
# LAI/PAI from TLS, simulated TLS and 3D models
#
# Written by N. Origo (19/08/2026)
#
##################################################################

"""Analyze occlusion grids, mesh LAI, and point-cloud vegetation profiles."""

import json
import struct
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import simpson

from geometrical_calculations import LAICalculations
from scene_io import SceneParameters, read_obj

if TYPE_CHECKING:
    import laspy


@dataclass
class HeliosSimulationData:
    """
    HELIOS++ LAS returns and optional pulse and waveform outputs.

    Parameters:
    -----------
    las : laspy.LasData
          LAS point records and metadata, including HELIOS++ extra dimensions.
    las_path : pathlib.Path
               Source LAS path for accessing internally stored waveforms.
    pulses : np.ndarray or None
             Structured pulse table, including no-return pulses when supplied.
             GPS times are stored as text, exactly as written by HELIOS++.
    waveform_path : pathlib.Path or None
                    Separate HELIOS++ waveform text output, if supplied.

    Notes:
    ------
    LAS points and pulse records are loaded into memory, whereas waveform text
    and LAS waveform packet bytes are read on demand. No pulse association,
    timestamp conversion, height correction, or PAVD preparation is performed.
    """

    las: "laspy.LasData"
    las_path: Path
    pulses: np.ndarray | None = None
    waveform_path: Path | None = None

    def iter_waveforms(self):
        """
        Stream HELIOS++ waveform text records without loading the entire file.

        Returns:
        --------
        waveforms : iterator of dict
                    Records with 'fullwave_index', 'beam_origin', 'beam_direction',
                    'min_time_ns', 'max_time_ns', 'gps_time', and 'intensities'.
                    GPS time is preserved as text to avoid losing precision.

        Raises:
        -------
        ValueError
            If no waveform file is configured or a record is malformed.
        """
        if self.waveform_path is None:
            raise ValueError("No HELIOS++ waveform text file was supplied.")
        with self.waveform_path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                fields = line.split()
                if not fields or fields[0].startswith("#"):
                    continue
                if len(fields) < 11:
                    raise ValueError(f"Malformed waveform record at {self.waveform_path}:{line_number}.")
                try:
                    yield {
                        "fullwave_index": int(fields[0]),
                        "beam_origin": np.array(fields[1:4], dtype=float),
                        "beam_direction": np.array(fields[4:7], dtype=float),
                        "min_time_ns": float(fields[7]),
                        "max_time_ns": float(fields[8]),
                        "gps_time": fields[9],
                        "intensities": np.array(fields[10:], dtype=float),
                    }
                except ValueError as exc:
                    raise ValueError(
                        f"Malformed waveform record at {self.waveform_path}:{line_number}."
                    ) from exc

    def read_waveform_packet(self, point_index):
        """
        Read the raw LAS waveform packet and descriptor for one point.

        Supports internal packets and external .wdp files. The returned bytes
        are not decoded or scaled; use the descriptor's sample format, gain,
        and offset when processing them.

        Parameters:
        -----------
        point_index : int
                      Zero-based LAS point index.

        Returns:
        --------
        packet : tuple of bytes and laspy WaveformPacketStruct
                 Raw waveform packet and its descriptor.

        Raises:
        -------
        ValueError
            If the point, waveform metadata, or packet is invalid or missing.
        """
        if not isinstance(point_index, (int, np.integer)) or not 0 <= point_index < len(self.las.points):
            raise ValueError("point_index must identify a LAS point.")
        if self.las.header.point_format.id not in (4, 5, 9, 10):
            raise ValueError("The LAS point format has no embedded waveform packet fields.")
        descriptor_index = int(self.las.wavepacket_index[point_index])
        if descriptor_index == 0:
            raise ValueError("The selected point has no waveform packet.")
        descriptors = [
            vlr for vlr in self.las.header.vlrs
            if vlr.user_id == "LASF_Spec" and vlr.record_id == descriptor_index + 99
        ]
        if len(descriptors) != 1 or not hasattr(descriptors[0], "parsed_record"):
            raise ValueError("The waveform packet descriptor is missing or invalid.")
        encoding = self.las.header.global_encoding
        if encoding.waveform_data_packets_internal == encoding.waveform_data_packets_external:
            raise ValueError("Exactly one internal or external waveform storage flag must be set.")
        offset = int(self.las.wavepacket_offset[point_index])
        size = int(self.las.wavepacket_size[point_index])
        if offset < 60 or size <= 0:
            raise ValueError("The waveform packet offset or size is invalid.")
        if encoding.waveform_data_packets_internal:
            packet_path = self.las_path
            record_start = int(self.las.header.start_of_waveform_data_packet_record)
            if record_start <= 0:
                raise ValueError("The LAS internal waveform record location is missing.")
        else:
            packet_path = self.las_path.with_suffix(".wdp")
            record_start = 0
        with packet_path.open("rb") as file:
            file.seek(record_start)
            header = file.read(60)
            if len(header) != 60:
                raise ValueError(f"Truncated waveform data record header in {packet_path}.")
            _, user_id, record_id, length, _ = struct.unpack("<H16sHQ32s", header)
            if user_id.rstrip(b"\0") != b"LASF_Spec" or record_id != 65535:
                raise ValueError(f"Invalid waveform data record header in {packet_path}.")
            if offset - 60 + size > length:
                raise ValueError(f"Waveform packet extends beyond the record in {packet_path}.")
            file.seek(record_start + offset)
            packet = file.read(size)
        if len(packet) != size:
            raise ValueError(f"Truncated waveform packet in {packet_path}.")
        return packet, descriptors[0].parsed_record


@dataclass
class CaldersScan:
    """
    Pulse and return data for one ground-based scan at a single location.

    All arrays are one-dimensional. Pulse arrays include every emitted pulse,
    including those with no detected returns. Each return references its pulse
    by zero-based index. Heights must already be corrected to the chosen ground
    reference; this container does not apply a DEM or plane correction.

    Parameters:
    -----------
    pulse_zenith, pulse_azimuth : array-like
                                  Pulse viewing angles in degrees. Zenith is
                                  measured from +Z; azimuth is measured from
                                  +X towards +Y.
    pulse_return_count : array-like
                         Number of returns recorded for each pulse, including
                         zero for pulses without a return.
    return_pulse_index : array-like
                         Index of the parent pulse for each return.
    return_height : array-like
                    Ground-referenced return heights in metres.
    return_number : array-like
                    One-based return number within its pulse.
    return_classification : array-like
                            Classification code for each return.
    min_zenith, max_zenith : float
                             Zenith limits in degrees for this scan.
    min_azimuth, max_azimuth : float
                               Inclusive azimuth limits in degrees.
    """

    pulse_zenith: ArrayLike
    pulse_azimuth: ArrayLike
    pulse_return_count: ArrayLike
    return_pulse_index: ArrayLike
    return_height: ArrayLike
    return_number: ArrayLike
    return_classification: ArrayLike
    min_zenith: float = 35.0
    max_zenith: float = 70.0
    min_azimuth: float = 0.0
    max_azimuth: float = 360.0


@dataclass
class CaldersProfileInput:
    """
    Prepared single-position TLS scans and settings for PAVD calculation.

    Parameters:
    -----------
    scans : sequence of CaldersScan
            One or more scans from the same ground-based location.
    min_height, max_height : float
                             Height range in metres. Returns at min_height are
                             excluded. As in PyLidar, max_height limits bin
                             starts, so the final bin may extend beyond it.
    height_bin_size : float
                      Height-bin spacing in metres.
    zenith_bin_size : float
                      Zenith-bin spacing in degrees.
    weighted : bool
               If true, count each return as 1 / returns per pulse; otherwise
               count only the first return.
    excluded_classes : sequence of int
                       Return classifications to exclude from interception.
    total_pai_method : str
                       Scale the solid-angle profile using 'hinge', 'linear',
                       or 'external' PAI.
    external_pai : float or None
                   Total PAI when total_pai_method is 'external'.

    Example:
    --------
    One intercepted pulse and one pulse without a return:

    >>> scan = CaldersScan([52.5, 52.5], [0, 0], [1, 0], [0], [0.5], [1], [1],
    ...                    min_zenith=50, max_zenith=55)
    >>> inputs = CaldersProfileInput((scan,), max_height=2, height_bin_size=1)
    >>> analysis = PointCloudVegetationAnalysis("scan", "real", "ground")
    >>> analysis.calculate_pavd_profile(inputs)["pgap"].tolist()
    [[0.5, 0.5]]
    """

    scans: tuple[CaldersScan, ...]
    min_height: float = 0.0
    max_height: float = 50.0
    height_bin_size: float = 0.5
    zenith_bin_size: float = 5.0
    weighted: bool = False
    excluded_classes: tuple[int, ...] = ()
    total_pai_method: str = "hinge"
    external_pai: float | None = None


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


class PointCloudVegetationAnalysis:
    """
    Template for deriving PAVD profiles and PAI from point clouds.

    Subclasses provide the relevant input reader, acquisition-specific
    preparation, and any alternative processing method. The base class provides
    the Calders et al. (2014) single-position ground-scan profile calculation
    and PAVD integration. HELIOS++ single-position ground scans can be
    prepared; real-data loading and other platforms remain hooks.
    """

    def __init__(self, point_cloud_path, source, platform, *, pulse_path=None,
                 waveform_path=None, ground_elevation=None, calders_settings=None,
                 zenith_range=(35.0, 70.0), azimuth_range=(0.0, 360.0)):
        """
        Configure a point-cloud analysis without loading or processing data.

        Parameters:
        -----------
        point_cloud_path : str or pathlib.Path
                           Path to the simulated or real point cloud.
        source : str
                 Input origin: 'simulated' or 'real'. This selects the reader.
        platform : str
                   Acquisition platform: 'ground', 'uav', or 'airborne'.
                   This selects the preparation hook independently of source.
        pulse_path : str or pathlib.Path or None, default=None
                     Optional HELIOS++ --writePulse text file.
        waveform_path : str or pathlib.Path or None, default=None
                        Optional HELIOS++ --writeWaveform text file.
        ground_elevation : float, array-like, or None, default=None
                           Ground elevation in LAS coordinates, either a flat
                           value or one value per LAS return in its original
                           order. Required for HELIOS++ ground preparation.
        calders_settings : CaldersProfileInput or None
                           Profile settings with an empty scans tuple. Defaults
                           to CaldersProfileInput defaults.
        zenith_range, azimuth_range : pair of float
                                      Scan limits in degrees; azimuth is
                                      measured from +X towards +Y.

        Raises:
        -------
        ValueError
            If the source or platform is not supported by this template.
        """
        if source not in ("simulated", "real"):
            raise ValueError("source must be 'simulated' or 'real'.")
        if platform not in ("ground", "uav", "airborne"):
            raise ValueError("platform must be 'ground', 'uav', or 'airborne'.")
        self.point_cloud_path = point_cloud_path
        self.source = source
        self.platform = platform
        self.pulse_path = Path(pulse_path).resolve() if pulse_path is not None else None
        self.waveform_path = Path(waveform_path).resolve() if waveform_path is not None else None
        self.ground_elevation = ground_elevation
        if calders_settings is not None and (
            not isinstance(calders_settings, CaldersProfileInput) or calders_settings.scans
        ):
            raise ValueError("calders_settings must be a CaldersProfileInput with no scans.")
        self.calders_settings = (
            calders_settings if calders_settings is not None else CaldersProfileInput(scans=())
        )
        self.zenith_range = zenith_range
        self.azimuth_range = azimuth_range

    def run(self):
        """
        Run the common loading, preparation, PAVD, and PAI workflow.

        Returns:
        --------
        pavd_profile : object
                       Profile returned by the selected processing method.
        pai : object
              PAI returned by the selected processing method.

        Raises:
        -------
        NotImplementedError
            If a required reader, preparation step, or calculation is not
            implemented by a subclass.
        """
        point_cloud = self.load_point_cloud()
        prepared_cloud = self.prepare_point_cloud(point_cloud)
        pavd_profile = self.calculate_pavd_profile(prepared_cloud)
        pai = self.calculate_pai(pavd_profile)
        return pavd_profile, pai

    def load_point_cloud(self):
        """
        Select the point-cloud reader based on the configured source.

        Returns:
        --------
        point_cloud : object
                      Data and metadata in the representation chosen by the
                      reader and processing implementation.
        """
        if self.source == "simulated":
            return self.load_simulated_point_cloud()
        return self.load_real_point_cloud()

    def load_simulated_point_cloud(self):
        """
        Read HELIOS++ LAS returns and optional pulse and waveform outputs.

        Point records, including extra dimensions, are loaded into memory with
        laspy. Pulse data are loaded when supplied; waveform text records and
        embedded LAS packets are read on demand. A LAS point cloud or waveform
        file alone does not include emitted pulses with no returns and cannot
        supply the complete pulse counts required by the Calders calculation.
        For .laz files laspy also needs a compatible LAZ backend. HELIOS++
        binary-compressed waveform or pulse files must first be decompressed.

        Returns:
        --------
        point_cloud : HeliosSimulationData
                      Loaded LAS data and separately supplied HELIOS++ outputs.

        Raises:
        -------
        ValueError
            If the LAS suffix or pulse table format is invalid.
        ImportError
            If laspy is not installed.
        OSError
            If a requested input file cannot be read.
        """
        import laspy

        path = Path(self.point_cloud_path).resolve()
        if path.suffix.lower() not in (".las", ".laz"):
            raise ValueError(f"Expected a LAS or LAZ point cloud: {path}")
        if self.waveform_path is not None:
            with self.waveform_path.open(encoding="utf-8"):
                pass
        pulses = None
        if self.pulse_path is not None:
            pulse_dtype = np.dtype([
                ("origin_x", "f8"), ("origin_y", "f8"), ("origin_z", "f8"),
                ("direction_x", "f8"), ("direction_y", "f8"), ("direction_z", "f8"),
                ("gps_time", object), ("pulse_index", "i8"), ("device_index", "i8"),
            ])
            pulses = np.loadtxt(self.pulse_path, dtype=pulse_dtype, ndmin=1)
            if not all(
                np.isfinite(pulses[column]).all()
                for column in ("origin_x", "origin_y", "origin_z",
                               "direction_x", "direction_y", "direction_z")
            ):
                raise ValueError("Pulse positions and directions must be finite.")
        # Do not eagerly load potentially huge waveform EVLRs into memory.
        with laspy.open(path, read_evlrs=False) as reader:
            las = reader.read()
        return HeliosSimulationData(las, path, pulses, self.waveform_path)

    def load_real_point_cloud(self):
        """
        Read measured point-cloud data.

        Returns:
        --------
        point_cloud : object
                      Measured points and any required acquisition metadata.

        Raises:
        -------
        NotImplementedError
            Until a real-data reader is implemented.
        """
        # Add measured-data IO, coordinate information, and metadata here.
        raise NotImplementedError("Implement a real point-cloud reader.")

    def prepare_point_cloud(self, point_cloud):
        """
        Select preparation for the configured acquisition platform.

        Parameters:
        -----------
        point_cloud : object
                      Point-cloud data and metadata returned by the reader.

        Returns:
        --------
        prepared_cloud : object
                         Data and metadata for the chosen processing method.
        """
        if self.platform == "ground":
            return self.prepare_ground_point_cloud(point_cloud)
        if self.platform == "uav":
            return self.prepare_uav_point_cloud(point_cloud)
        return self.prepare_airborne_point_cloud(point_cloud)

    def prepare_ground_point_cloud(self, point_cloud):
        """
        Prepare HELIOS++ single-position ground data for Calders et al. (2014).

        Match LAS fullwaveIndex to pulse_index, retain no-return pulses and
        calculate viewing angles from the emitted beam direction. Pulse files
        must contain all emitted pulses from the same leg as the LAS file;
        this cannot be verified solely from the detected returns. Heights are
        referenced using the explicitly supplied ground elevations. The
        waveform samples are not required by the discrete-return calculation.

        Parameters:
        -----------
        point_cloud : HeliosSimulationData
                      HELIOS++ LAS data and complete pulse table.

        Returns:
        --------
        prepared_cloud : CaldersProfileInput
                         Single-position pulse and return data and settings.

        Raises:
        -------
        TypeError
            If the reader returned an unsupported data representation.
        ValueError
            If the pulse file, ground elevations, scan, or LAS returns are
            missing, ambiguous, or inconsistent.
        """
        if not isinstance(point_cloud, HeliosSimulationData):
            raise TypeError("Ground preparation requires HeliosSimulationData.")
        pulses = point_cloud.pulses
        if pulses is None or len(pulses) == 0:
            raise ValueError("A non-empty HELIOS++ pulse file including no-return pulses is required.")
        if np.unique(pulses["device_index"]).size != 1:
            raise ValueError("Multiple scanner devices cannot be associated unambiguously with LAS returns.")
        origins = np.column_stack([pulses[f"origin_{axis}"] for axis in "xyz"])
        directions = np.column_stack([pulses[f"direction_{axis}"] for axis in "xyz"])
        if not np.isfinite(origins).all() or not np.isfinite(directions).all():
            raise ValueError("Pulse origins and directions must be finite.")
        # HELIOS++ writes coordinates to four decimal places.
        if np.any(np.ptp(origins, axis=0) > 0.00010001):
            raise ValueError("Calders ground preparation requires a single scanner position.")
        lengths = np.linalg.norm(directions, axis=1)
        if np.any(lengths == 0) or not np.isfinite(lengths).all():
            raise ValueError("Pulse directions must be finite and non-zero.")
        zenith = np.degrees(np.arccos(np.clip(directions[:, 2] / lengths, -1, 1)))
        azimuth = np.degrees(np.arctan2(directions[:, 1], directions[:, 0])) % 360

        las = point_cloud.las
        if "fullwaveIndex" not in list(las.point_format.extra_dimension_names):
            raise ValueError("HELIOS++ LAS must contain the fullwaveIndex extra dimension.")
        pulse_ids = pulses["pulse_index"]
        order = np.argsort(pulse_ids)
        sorted_ids = pulse_ids[order]
        if np.any(sorted_ids < 0) or np.any(np.diff(sorted_ids) == 0):
            raise ValueError("Pulse indices must be unique and non-negative.")
        return_ids = np.asarray(las["fullwaveIndex"])
        if (not np.issubdtype(return_ids.dtype, np.integer)
            or np.any(return_ids < 0)
            or (np.issubdtype(return_ids.dtype, np.unsignedinteger)
                and np.any(return_ids > np.iinfo(np.int64).max))):
            raise ValueError("LAS fullwaveIndex values must be non-negative 64-bit integers.")
        return_ids = return_ids.astype(np.int64, copy=False)
        positions = np.searchsorted(sorted_ids, return_ids)
        if np.any(positions == len(sorted_ids)) or np.any(
            sorted_ids[np.minimum(positions, len(sorted_ids) - 1)] != return_ids
        ):
            raise ValueError("Some LAS returns have no matching emitted pulse.")
        parent = order[positions]
        counts = np.bincount(parent, minlength=len(pulses))
        numbers = np.asarray(las.return_number, dtype=np.int64)
        declared_counts = np.asarray(las.number_of_returns, dtype=np.int64)
        if (
            np.any(declared_counts != counts[parent])
            or np.any(numbers < 1) or np.any(numbers > counts[parent])
            or len(np.unique(np.column_stack((parent, numbers)), axis=0)) != len(parent)
        ):
            raise ValueError("LAS return numbers or per-pulse return counts are inconsistent.")

        if self.ground_elevation is None:
            raise ValueError("ground_elevation is required for ground-referenced return heights.")
        ground = np.asarray(self.ground_elevation, dtype=float)
        if (ground.ndim != 0 and ground.shape != (len(las.points),)) or not np.isfinite(ground).all():
            raise ValueError("ground_elevation must be finite and scalar or one value per LAS return.")
        height = np.asarray(las.z) - ground
        if not np.isfinite(height).all():
            raise ValueError("Ground-referenced return heights must be finite.")
        zenith_range = np.asarray(self.zenith_range, dtype=float)
        azimuth_range = np.asarray(self.azimuth_range, dtype=float)
        if (zenith_range.shape != (2,) or not np.isfinite(zenith_range).all()
            or not 0 <= zenith_range[0] < zenith_range[1] <= 90
            or azimuth_range.shape != (2,) or not np.isfinite(azimuth_range).all()
            or not 0 <= azimuth_range[0] <= azimuth_range[1] <= 360):
            raise ValueError("Scan angular ranges must be ordered and in degrees.")
        scan = CaldersScan(
            pulse_zenith=zenith, pulse_azimuth=azimuth,
            pulse_return_count=counts, return_pulse_index=parent,
            return_height=height, return_number=numbers,
            return_classification=np.asarray(las.classification, dtype=np.int64),
            min_zenith=float(zenith_range[0]), max_zenith=float(zenith_range[1]),
            min_azimuth=float(azimuth_range[0]), max_azimuth=float(azimuth_range[1]),
        )
        return replace(self.calders_settings, scans=(scan,))

    def prepare_uav_point_cloud(self, point_cloud):
        """
        Prepare a UAV point cloud for the selected method.

        Parameters:
        -----------
        point_cloud : object
                      Point-cloud data and associated metadata.

        Returns:
        --------
        prepared_cloud : object
                         UAV data ready for processing.

        Raises:
        -------
        NotImplementedError
            Until UAV preparation is implemented.
        """
        # Add any required UAV trajectory and viewing-geometry handling here.
        raise NotImplementedError("Implement UAV point-cloud preparation.")

    def prepare_airborne_point_cloud(self, point_cloud):
        """
        Prepare an airborne point cloud for the selected method.

        Parameters:
        -----------
        point_cloud : object
                      Point-cloud data and associated metadata.

        Returns:
        --------
        prepared_cloud : object
                         Airborne data ready for processing.

        Raises:
        -------
        NotImplementedError
            Until airborne preparation is implemented.
        """
        # Add any required airborne trajectory and viewing-geometry handling here.
        raise NotImplementedError("Implement airborne point-cloud preparation.")

    def calculate_pavd_profile(self, prepared_cloud: CaldersProfileInput) -> dict[str, np.ndarray]:
        """
        Calculate single-position TLS profiles using Calders et al. (2014).

        Bins and return weighting follow PyLidar's PAVD_CALDERS2014 convention.
        Hinge, linear, and solid-angle-weighted PAI and PAVD are returned.
        Height values are bin starts; PAVD is the gradient of cumulative PAI.
        Unobserved bins and estimates that cannot be computed are NaN; zero gap
        probabilities can produce infinite hinge PAI or undefined gradients.
        The linear estimate requires at least three observed zenith bins and
        is undefined if any of their gap probabilities is zero. Solid-angle
        profiles are undefined when all top-bin gaps are one or the selected
        total PAI is undefined. Angular limits may differ between scans.
        The reader must supply every emitted pulse, including no-return pulses.

        See Calders et al. (2014), Agricultural and Forest Meteorology 194,
        104-117, and https://www.pylidar.org/en/latest/commandline_canopy.html.

        Parameters:
        -----------
        prepared_cloud : CaldersProfileInput
                         Ground-referenced pulses, returns, and settings from
                         one ground-based scan position.

        Returns:
        --------
        pavd_profile : dict
                       Arrays named 'height', 'zenith', 'pgap', 'hinge_pai',
                       'hinge_pavd', 'linear_pai', 'linear_pavd',
                       'linear_mla', 'weighted_pai', and 'weighted_pavd'.
                       pgap has shape (zenith bins, height bins); the other
                       profiles have one value per height bin. PAVD is in m-1,
                       PAI is dimensionless, and linear_mla is in degrees.

        Raises:
        -------
        ValueError
            If the platform, settings, or pulse/return data are invalid.
        TypeError
            If prepared_cloud is not a CaldersProfileInput.
        """
        if self.platform != "ground":
            raise ValueError("Calders profiles require a single-position ground scan.")
        if not isinstance(prepared_cloud, CaldersProfileInput):
            raise TypeError("prepared_cloud must be a CaldersProfileInput.")
        config = prepared_cloud
        if not config.scans or not all(isinstance(s, CaldersScan) for s in config.scans):
            raise ValueError("scans must contain at least one CaldersScan.")
        if (
            not np.isfinite([config.min_height, config.max_height,
                             config.height_bin_size, config.zenith_bin_size]).all()
            or config.min_height >= config.max_height
            or config.height_bin_size <= 0 or config.zenith_bin_size <= 0
        ):
            raise ValueError("Height and zenith bin settings must be finite and valid.")
        if config.total_pai_method not in ("hinge", "linear", "external"):
            raise ValueError("total_pai_method must be 'hinge', 'linear', or 'external'.")
        if config.total_pai_method == "external" and (
            config.external_pai is None or not np.isfinite(config.external_pai)
            or config.external_pai < 0
        ):
            raise ValueError("external_pai must be finite and non-negative.")
        for scan in config.scans:
            limits = (scan.min_zenith, scan.max_zenith,
                      scan.min_azimuth, scan.max_azimuth)
            if (
                not np.isfinite(limits).all()
                or not 0 <= scan.min_zenith < scan.max_zenith <= 90
                or not 0 <= scan.min_azimuth <= scan.max_azimuth <= 360
            ):
                raise ValueError("Scan angular limits must be ordered and in degrees.")

        zenith = np.arange(
            min(s.min_zenith for s in config.scans) + config.zenith_bin_size / 2,
            max(s.max_zenith for s in config.scans), config.zenith_bin_size,
        )
        height = np.arange(min(0.0, config.min_height), config.max_height,
                           config.height_bin_size)
        if len(zenith) == 0 or len(height) < 2:
            raise ValueError("The settings must produce zenith bins and at least two height bins.")
        shot_counts = np.zeros(len(zenith))
        interceptions = np.zeros((len(zenith), len(height)))
        for scan in config.scans:
            self._accumulate_calders_scan(
                scan, config, zenith, height, shot_counts, interceptions
            )
        if not np.any(shot_counts):
            raise ValueError("No pulses fall within the selected angular bins.")

        gaps = np.full(interceptions.shape, np.nan)
        np.divide(np.cumsum(interceptions, axis=1), shot_counts[:, None],
                  out=gaps, where=shot_counts[:, None] > 0)
        gaps = 1.0 - gaps
        # Round-off in weighted returns must not turn a closed canopy negative.
        gaps = np.clip(gaps, 0.0, 1.0)
        radians = np.radians(zenith)
        hinge = np.argmin(np.abs(radians - np.arctan(np.pi / 2)))
        with np.errstate(divide="ignore", invalid="ignore"):
            optical_depth = -np.log(gaps)
        hinge_pai = 1.1 * optical_depth[hinge]
        linear_pai = np.full(len(height), np.nan)
        linear_mla = np.full(len(height), np.nan)
        x = np.abs(2 * np.tan(radians) / np.pi)
        for index in range(len(height)):
            depth = optical_depth[:, index]
            observed = ~np.isnan(depth)
            if np.count_nonzero(observed) < 3 or not np.isfinite(depth[observed]).all():
                continue
            design = np.column_stack((x[observed], np.ones(np.count_nonzero(observed))))
            if np.linalg.matrix_rank(design) < 2:
                continue
            vertical, horizontal = np.linalg.lstsq(design, depth[observed], rcond=None)[0]
            if vertical < 0:
                vertical, horizontal = 0.0, np.mean(depth[observed])
            if horizontal < 0:
                vertical, horizontal = np.mean(depth[observed] / x[observed]), 0.0
            linear_pai[index] = vertical + horizontal
            linear_mla[index] = np.degrees(np.arctan2(vertical, horizontal))

        if config.total_pai_method == "external":
            total_pai = config.external_pai
        elif config.total_pai_method == "linear":
            total_pai = np.max(linear_pai)
        else:
            total_pai = np.max(hinge_pai)
        # Normalize the angular profiles at the top bin as in PyLidar.
        active = (gaps[:, -1] < 1) & ~np.isnan(gaps[:, -1])
        weighted_pai = np.full(len(height), np.nan)
        if np.any(active):
            weights = np.sin(radians[active])
            with np.errstate(divide="ignore", invalid="ignore"):
                proportions = optical_depth[active] / optical_depth[active, -1, None]
                weighted_pai = total_pai * np.average(proportions, axis=0, weights=weights)
        with np.errstate(invalid="ignore"):
            return {
                "height": height,
                "zenith": zenith,
                "pgap": gaps,
                "hinge_pai": hinge_pai,
                "hinge_pavd": np.gradient(hinge_pai, config.height_bin_size),
                "linear_pai": linear_pai,
                "linear_pavd": np.gradient(linear_pai, config.height_bin_size),
                "linear_mla": linear_mla,
                "weighted_pai": weighted_pai,
                "weighted_pavd": np.gradient(weighted_pai, config.height_bin_size),
            }

    @staticmethod
    def _accumulate_calders_scan(scan, config, zenith, height, shots, interceptions):
        """Validate one scan and accumulate pulses and weighted interceptions."""
        pulse_zenith = np.asarray(scan.pulse_zenith, dtype=float)
        pulse_azimuth = np.asarray(scan.pulse_azimuth, dtype=float)
        return_count = np.asarray(scan.pulse_return_count)
        parent = np.asarray(scan.return_pulse_index)
        return_height = np.asarray(scan.return_height, dtype=float)
        number = np.asarray(scan.return_number)
        classification = np.asarray(scan.return_classification)
        if (
            any(a.ndim != 1 for a in
                (pulse_zenith, pulse_azimuth, return_count, parent,
                 return_height, number, classification))
            or len(pulse_zenith) != len(pulse_azimuth)
            or len(pulse_zenith) != len(return_count)
            or len(parent) != len(return_height)
            or len(parent) != len(number)
            or len(parent) != len(classification)
        ):
            raise ValueError("Pulse and return arrays must be one-dimensional and aligned.")
        if (
            not np.isfinite(pulse_zenith).all() or not np.isfinite(pulse_azimuth).all()
            or not np.isfinite(return_height).all()
            or np.any((pulse_zenith < 0) | (pulse_zenith > 180))
            or np.any((pulse_azimuth < 0) | (pulse_azimuth > 360))
        ):
            raise ValueError("Pulse angles and return heights must be finite and valid.")
        if not all(a.size == 0 or np.issubdtype(a.dtype, np.integer) for a in
                   (return_count, parent, number, classification)):
            raise ValueError("Pulse counts, indices, return numbers, and classes must be integers.")
        return_count = return_count.astype(np.int64)
        parent = parent.astype(np.int64)
        number = number.astype(np.int64)
        if (
            np.any(return_count < 0) or np.any(parent < 0)
            or np.any(parent >= len(return_count)) or np.any(number < 1)
            or np.any(number > return_count[parent])
            or not np.array_equal(np.bincount(parent, minlength=len(return_count)), return_count)
        ):
            raise ValueError("Return indices, numbers, and pulse return counts are inconsistent.")
        if len(parent) and len(np.unique(np.column_stack((parent, number)), axis=0)) != len(parent):
            raise ValueError("Each pulse must have unique return numbers.")

        for i, center in enumerate(zenith):
            if not scan.min_zenith < center <= scan.max_zenith:
                continue
            pulse_mask = (
                (pulse_zenith > center - config.zenith_bin_size / 2)
                & (pulse_zenith <= center + config.zenith_bin_size / 2)
                & (pulse_azimuth >= scan.min_azimuth)
                & (pulse_azimuth <= scan.max_azimuth)
            )
            shots[i] += np.count_nonzero(pulse_mask)
            selected = (
                pulse_mask[parent] & (return_height > config.min_height)
                & (return_height >= height[0])
                & (return_height < height[-1] + config.height_bin_size)
                & ~np.isin(classification, config.excluded_classes)
            )
            if not config.weighted:
                selected &= number == 1
            indices = np.floor((return_height[selected] - height[0]) /
                               config.height_bin_size).astype(int)
            weights = (1.0 / return_count[parent[selected]] if config.weighted
                       else np.ones(len(indices)))
            interceptions[i] += np.bincount(indices, weights=weights, minlength=len(height))

    def calculate_pai(self, pavd_profile, height_range=None):
        """
        Integrate each PAVD profile to obtain plant area index.

        Integration uses Simpson's rule on the sampled height coordinates.
        With two samples, SciPy's Simpson integration uses a trapezoid.
        The default range spans the first and last profile samples; it is only
        a full-canopy estimate if those samples cover the measured canopy.
        A non-finite PAVD sample in the selected range makes that profile's PAI
        unavailable (NaN); such samples are never silently skipped.

        PAVD is derived from the gradient of cumulative PAI, so its numerical
        integral need not exactly equal the change in the original PAI profile.
        The integral is PAI, not leaf-only LAI.

        Parameters:
        -----------
        pavd_profile : dict
                       Profile produced by calculate_pavd_profile, containing
                       height, hinge_pavd, linear_pavd, and weighted_pavd.
        height_range : tuple of float or None, default=None
                       Inclusive minimum and maximum sample heights in metres.
                       Both must match existing profile sample heights; None
                       selects the full sampled height range.

        Returns:
        --------
        pai : dict
              Results keyed by 'hinge', 'linear', and 'weighted'. Each contains
              'pai', 'height_min', 'height_max', 'nan_bins', and
              'infinite_bins'. Bin arrays contain zero-based indices into the
              original profile height array, restricted to the selected range.

        Raises:
        -------
        ValueError
            If heights, profile shapes, or the requested height range are invalid.
        KeyError
            If a required profile field is missing.
        """
        heights = np.asarray(pavd_profile["height"], dtype=float)
        if (
            heights.ndim != 1 or len(heights) < 2
            or not np.isfinite(heights).all()
            or np.any(np.diff(heights) <= 0)
        ):
            raise ValueError("Profile heights must be finite, increasing, and contain at least two samples.")

        if height_range is None:
            first, last = 0, len(heights) - 1
        else:
            bounds = np.asarray(height_range, dtype=float)
            if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
                raise ValueError("height_range must contain two finite, increasing sample heights.")
            indices = []
            for bound in bounds:
                # Permit only floating-point round-off, not interpolation or
                # snapping a materially different height to a sample.
                tolerance = 8 * np.finfo(float).eps * max(1.0, abs(bound))
                matches = np.flatnonzero(np.abs(heights - bound) <= tolerance)
                if len(matches) != 1:
                    raise ValueError("height_range bounds must match existing profile sample heights.")
                indices.append(int(matches[0]))
            first, last = indices
            if first >= last:
                raise ValueError("height_range must include at least two profile samples.")

        selected_heights = heights[first:last + 1]
        results = {}
        for estimator in ("hinge", "linear", "weighted"):
            values = np.asarray(pavd_profile[f"{estimator}_pavd"], dtype=float)
            if values.ndim != 1 or values.shape != heights.shape:
                raise ValueError(f"{estimator}_pavd must match the height array.")
            selected = values[first:last + 1]
            nan_bins = np.flatnonzero(np.isnan(selected)) + first
            infinite_bins = np.flatnonzero(np.isinf(selected)) + first
            pai = (
                float("nan") if len(nan_bins) or len(infinite_bins)
                else float(simpson(selected, x=selected_heights))
            )
            results[estimator] = {
                "pai": pai,
                "height_min": float(selected_heights[0]),
                "height_max": float(selected_heights[-1]),
                "nan_bins": nan_bins,
                "infinite_bins": infinite_bins,
            }
        return results
