####################################################################################
#
# Build eradiate canopy scenes using 3D trees
#
# Written by W. Van den Broeck & N. Origo (18/08/2025)
#
####################################################################################

import getpass
import glob
import inspect
import itertools
import json
import os
import pathlib
import pdb
import subprocess
import time

import eradiate
import numpy as np
import open3d as o3d
import rasterio
import xarray as xr
from eradiate import unit_registry as ureg
from scipy.spatial import Delaunay

try:
    os.environ["ERADIATE_PROJECT_DIR"]
except KeyError:
    raise AttributeError(
        "Environmental variable 'ERADIATE_PROJECT_DIR' is not set - this needs "
        "to point to the directory containing the mesh and spectra dirs."
    )


class Illumination:
    def __init__(
        self, out_json=None, angle_units=ureg.degrees, default_irradiance=False
    ):
        """
        Class to house different illumination types

        Parameters:
        -----------
        angle_units        : str
                             Gives the units of the angles to be used in the class instance.
                             Default: "degrees"; otherwise "radians".
        out_json           : str
                             Filename (including directory) of the .json file to write to.
        default_irradiance : bool
                             Determines whether the default solar irradiance should be used.
                             Note that setting this to true results in long simulation times.
        """
        if angle_units == ureg.degrees or angle_units == ureg.radians:
            self.angle_units = angle_units
        else:
            raise IOError(
                "angle_units must be either 'degrees' or 'radians'. \n"
                "use the pint unit registry (e.g. ureg.degrees"
            )
        self.out_json = out_json
        self.def_ir = default_irradiance

    def directional(self, zenith, azimuth, irradiance=None, override_name_change=False):
        """
        Create directional illumination source

        Parameters:
        -----------
        zenith               : float
                               Zenith angle of the directional light source
                               (0-180 or 0-pi, with 0 being the normal)
        azimuth              : float
                               Azimuth angle of the directional light source
                               (0-360 or 0=2*pi)
        irradiance           : define...
                               Spectrum of irradiance for the light source...
        override_name_change : bool
                               Default is False, overrides the addition of the parameters
                               to the json filename, e.g. test.json -> test_zen_az.json
        """
        if self.angle_units == ureg.radians:
            if zenith >= 0 and zenith <= np.pi:
                pass
            else:
                raise IOError("zenith must be in the range of 0-pi")
            if azimuth >= 0 and azimuth <= 2 * np.pi:
                pass
            else:
                raise IOError("azimuth must be in the range of 0-2*pi")
        else:
            if zenith >= 0 and zenith <= 180:
                pass
            else:
                raise IOError("zenith must be in the range of 0-180")
            if azimuth >= 0 and azimuth <= 360:
                pass
            else:
                raise IOError("azimuth must be in the range of 0-360")

        illum = {
            "type": "directional",
            "zenith": zenith * self.angle_units,
            "azimuth": azimuth * self.angle_units,
        }

        if irradiance != None:
            illum["irradiance"] = irradiance
        elif self.def_ir:
            illum["irradiance"] = eradiate.scenes.spectra.SolarIrradianceSpectrum()

        if self.out_json:
            jm = JsonManipulation()
            oname = jm.change_json_fname(
                self.out_json, override_name_change, "direct", [zenith, azimuth]
            )
            jm.write_json(illum, oname)
        return illum

    def isotropic(self, radiance=None, override_name_change=False):
        """Create isotropic illumination source"""
        illum = {"type": "constant"}

        if radiance != None:
            illum["radiance"] = radiance

        if self.out_json:
            jm = JsonManipulation()
            oname = jm.change_json_fname(
                self.out_json, override_name_change, "isotropic", None
            )
            jm.write_json(illum, oname)
        return illum

    def conical(
        self,
        zenith,
        azimuth,
        angular_diameter,
        irradiance=None,
        override_name_change=False,
    ):
        """
        Create a conical light source

        Parameters:
        -----------
        zenith           : float
                           Zenith angle of the directional light source (0-180 or
                           0-pi, with 0 being the normal)
        azimuth          : float
                           Azimuth angle of the directional light source (0-360 or
                           0=2*pi)
        angular_diameter : float
                           Angular diameter of the illumination source as seen from
                           the target
        irradiance       : define...
                           Spectrum of irradiance for the light source...
        override_name_change : bool
                               Default is False, overrides the addition of the parameters
                               to the json filename, e.g. test.json -> test_zen_az.json
        """
        if self.angle_units == ureg.radians:
            if zenith >= 0 and zenith <= np.pi:
                pass
            else:
                raise IOError("zenith must be in the range of 0-pi")
            if azimuth >= 0 and azimuth <= 2 * np.pi:
                pass
            else:
                raise IOError("azimuth must be in the range of 0-2*pi")
        else:
            if zenith >= 0 and zenith <= 180:
                pass
            else:
                raise IOError("zenith must be in the range of 0-180")
            if azimuth >= 0 and azimuth <= 360:
                pass
            else:
                raise IOError("azimuth must be in the range of 0-360")

        illum = {
            "type": "astroobject",
            "zenith": zenith * self.angle_units,
            "azimuth": azimuth * self.angle_units,
            "angular_diameter": angular_diameter * self.angle_units,
        }

        if irradiance != None:
            illum["irradiance"] = irradiance
        elif self.def_ir:
            illum["irradiance"] = eradiate.scenes.spectra.SolarIrradianceSpectrum()

        if self.out_json:
            jm = JsonManipulation()
            oname = jm.change_json_fname(
                self.out_json,
                override_name_change,
                "conical",
                [zenith, azimuth, angular_diameter],
            )
            jm.write_json(illum, oname)
        return illum

    def sun(self, zenith, azimuth, irradiance=None, override_name_change=False):
        """
        Create a default sun as the light source with an angular diameter of 0.5358
        degrees - if you want a different diameter then use the conical method.

        Parameters:
        -----------
        zenith           : float
                           Solar zenith angle (0-180 or 0-pi, with 0 being the normal)
        azimuth          : float
                           Solar azimuth angle (0-360 or 0=2*pi)
        irradiance       : define...
                           Spectrum of irradiance for the light source...
        override_name_change : bool
                               Default is False, overrides the addition of the parameters
                               to the json filename, e.g. test.json -> test_zen_az.json
        """
        if self.angle_units == ureg.radians:
            if zenith >= 0 and zenith <= np.pi:
                pass
            else:
                raise IOError("zenith must be in the range of 0-pi")
            if azimuth >= 0 and azimuth <= 2 * np.pi:
                pass
            else:
                raise IOError("azimuth must be in the range of 0-2*pi")
            sun_fov = np.deg2rad(0.5385)
        else:
            if zenith >= 0 and zenith <= 180:
                pass
            else:
                raise IOError("zenith must be in the range of 0-180")
            if azimuth >= 0 and azimuth <= 360:
                pass
            else:
                raise IOError("azimuth must be in the range of 0-360")
            sun_fov = 0.5385
        
        # need to unset and set the filename to avoid writing the file twice
        out_json = self.out_json
        self.out_json = None
        illum = self.conical(zenith, azimuth, sun_fov, irradiance=irradiance)
        self.out_json = out_json

        if self.out_json:
            jm = JsonManipulation()
            oname = jm.change_json_fname(
                self.out_json, override_name_change, "sun", [zenith, azimuth]
            )
            jm.write_json(illum, oname)
        return illum


class Camera:
    def __init__(self, distance_units=ureg.m, angle_units=ureg.degrees, out_json=None):
        """Class to house different camera types"""
        # check and sort out dimensions
        angle_options = [ureg.degrees, ureg.radians]
        if angle_units in angle_options:
            pass
        else:
            raise IOError(f"{angle_units} is not an angle option")
        self.angle_units = angle_units

        test_unit = 1 * distance_units
        if test_unit.check("[length]"):
            pass
        else:
            raise IOError(f"{distance_units} not a unit of length")
        self.distance_units = distance_units
        self.out_json = out_json

    def perspective_camera(
        self,
        spp,
        npix,
        fov,
        camloc,
        tarloc,
        srf=None,
        sampler="independent",
        rfilter="box",
        up=[0.0, 0.0, 1.0],
        far_clip=10000000,
        idd=None,
        override_name_change=False,
    ):
        """
        Create a generic perspective camera

        Parameters:
        -----------
        spp      : int
                   Number of samples per pixel
        npix     : tuple, int
                   Number of pixels in the X and Y directions
        fov      : float
                   Field of view of the camera in the angle units used during the
                   class instantiation
        camloc   : tuple, float
                   Camera location in (X,Y,Z) space in the distance units used during
                   the class instantiation
        tarloc   : tuple, float
                   Target location (where you want to image) in (X,Y,Z) space in the
                   distance units used during the class instantiation
        srf      : Path, str, eradiate.spectral.SpectralResponseFunction, or dict
                   Spectral response function for the camera
        sampler  : str, default="independent"
                   Pseudo random number generator type. Options are: "independent",
                   "stratified", "multijitter", "orthogonal", and "ldsampler"
        rfilter  : str, default="box"
                   Reconstruction filter used to scatter samples on a pixel. Options are:
                   "box", "gaussian", "tent", "mitchell", "catmullrom", and "lanczos"
        up       : tuple, float
                   Up vector of the camera (unitless)
        far_clip : int or float
                   Distance to the far clip plane in the distance units used during the
                   class instantiation, default=10000 km (10000000 m)
        idd      : str or None
                   ID of the camera, if None then the ID is created based on the parameters
                   given. Note that any "." given in the id are converted to "," as Mitsuba
                   (which underpins Eradiate) cannot handle these.
        override_name_change : bool
                               Default is False, overrides the addition of the parameters
                               to the json filename, e.g. test.json -> test_<>_<>.json
        """
        # parameter checking
        self.check_var("spp", spp, [int])
        self.check_var("fov", fov, [float])
        self.check_var("sampler", sampler, [str])
        self.check_var("rfilter", rfilter, [str])
        self.check_var("far_clip", far_clip, [int, float])
        self.check_tuple("npix", npix, 2, int)
        self.check_tuple("camloc", camloc, 3, float)
        self.check_tuple("tarloc", tarloc, 3, float)
        self.check_tuple("up", up, 3, float)
        if (
            isinstance(srf, str)
            or isinstance(srf, pathlib.Path)
            or self.check_srf_type(srf)
            or isinstance(srf, dict)
            or not srf
        ):
            pass
        else:
            raise IOError(
                "'srf' must be a str, pathlib.Path, eradiate.spectral" + " or dict"
            )

        # ID creation or ID checking
        if idd == None:
            if srf is not None:
                srf_idd = self.get_id_from_srf(srf)
            else:
                srf_idd = "none"
            idd = (
                f"perspective_{spp}_{npix[0]}-{npix[1]}_{fov}_{camloc[0]}-{camloc[1]}-"
                + f"{camloc[2]}_{tarloc[0]}-{tarloc[1]}-{tarloc[2]}_{srf_idd}_{sampler}_"
                + f"{rfilter}_{up[0]}-{up[1]}-{up[2]}_{far_clip}"
            )
        else:
            self.check_var("idd", idd, [str])

        # sort out SRF
        if type(srf) != type({}):
            # account for if srf is a str or a path
            srfs = str(srf)
            if srfs[-3:] == ".nc":
                srfi = {"category": "srf", "path": srf}
            else:
                if type(srf) == str:
                    srf += ".nc"
                elif type(srf) == pathlib.PosixPath:
                    srf = pathlib.PosixPath(srfs + ".nc")
                srfi = {"category": "srf", "path": srf}
        else:
            srfi = srf

        # create measure dictionary
        measure = {
            "id": idd.replace(".", ","),
            "type": "perspective",
            "film_resolution": tuple(npix),
            "spp": spp,
            "fov": fov * self.angle_units,
            "origin": tuple(camloc) * self.distance_units,
            "target": tuple(tarloc) * self.distance_units,
            "up": tuple(up),
            "far_clip": far_clip * self.distance_units,
            "srf": srfi,
            "sampler": sampler,
            "rfilter": rfilter,
        }

        if self.out_json:
            jm = JsonManipulation()
            oname = jm.change_json_fname(
                self.out_json, override_name_change, "perspective", [idd[12:]]
            )
            jm.write_json(measure, oname)
        return measure

    def mpdistant_camera(
        self,
        spp,
        npix,
        camloc,
        tarloc,
        srf=None,
        sampler="independent",
        rfilter="box",
        idd=None,
        override_name_change=False,
    ):
        """
        Create a generic multi-pixel distant camera - this is effectively a camera where each pixel is
        considered directional in its sampling. Only works for cameras that are really far from the
        target (e.g. satellites).

        Parameters:
        -----------
        spp      : int
                   Number of samples per pixel
        npix     : tuple, int
                   Number of pixels in the X and Y directions
        camloc   : tuple, float
                   Camera location in (X,Y,Z) space in the distance units used during
                   the class instantiation
        tarloc   : tuple, float
                   Target location (where you want to image) in (X,Y,Z) space in the
                   distance units used during the class instantiation
        srf      : Path, str, eradiate.spectral.SpectralResponseFunction, or dict
                   Spectral response function for the camera
        sampler  : str, default="independent"
                   Pseudo random number generator type. Options are: "independent",
                   "stratified", "multijitter", "orthogonal", and "ldsampler"
        rfilter  : str, default="box"
                   Reconstruction filter used to scatter samples on a pixel. Options are:
                   "box", "gaussian", "tent", "mitchell", "catmullrom", and "lanczos"
        idd      : str or None
                   ID of the camera, if None then the ID is created based on the parameters
                   given. Note that any "." given in the id are converted to "," as Mitsuba
                   (which underpins Eradiate) cannot handle these.
        override_name_change : bool
                               Default is False, overrides the addition of the parameters
                               to the json filename, e.g. test.json -> test_<>_<>.json

        """
        # parameter checking
        self.check_var("spp", spp, [int])
        self.check_var("sampler", sampler, [str])
        self.check_var("rfilter", rfilter, [str])
        self.check_tuple("npix", npix, 2, int)
        self.check_tuple("camloc", camloc, 3, float)
        self.check_tuple("tarloc", tarloc, 3, float)
        if (
            isinstance(srf, str)
            or isinstance(srf, pathlib.Path)
            or self.check_srf_type(srf)
            or isinstance(srf, dict)
            or not srf
        ):
            pass
        else:
            raise IOError(
                "'srf' must be a str, pathlib.Path, eradiate.spectral" + " or dict"
            )

        # ID creation or ID checking
        if idd == None:
            if srf is not None:
                srf_idd = self.get_id_from_srf(srf)
            else:
                srf_idd = "none"
            idd = (
                f"mpdistant_{spp}_{npix[0]}-{npix[1]}_{camloc[0]}-{camloc[1]}-"
                + f"{camloc[2]}_{tarloc[0]}-{tarloc[1]}-{tarloc[2]}_{srf_idd}_{sampler}_"
                + f"{rfilter}"
            )
        else:
            self.check_var("idd", idd, [str])

        # sort out SRF
        if type(srf) != type({}):
            # account for if srf is a str or a path
            srfs = str(srf)
            if srfs[-3:] == ".nc":
                srf = pathlib.PosixPath(srfs)
            else:
                srf = pathlib.PosixPath(srfs + ".nc")
            srfi = {"category": "srf", "path": srf}
        else:
            if not isinstance(srf["path"], pathlib.PosixPath):
                srf["path"] = pathlib.PosixPath(srf["path"])
            else:
                srfi = srf

        direction = np.array(camloc) - np.array(tarloc)
        ray_offset = (direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2) ** 0.5

        measure = {
            "id": idd.replace(".", ","),
            "type": "mpdistant",
            "film_resolution": tuple(npix),
            "spp": spp,
            "direction": tuple(direction),
            "target": tuple(tarloc) * self.distance_units,
            "ray_offset": ray_offset,
            "srf": srfi,
            "sampler": sampler,
            "rfilter": rfilter,
        }
        if self.out_json:
            jm = JsonManipulation()
            oname = jm.change_json_fname(
                self.out_json, override_name_change, "mpdistant", [idd[12:]]
            )
            jm.write_json(measure, oname)
        return measure

    def hemispherical_camera(
        self,
        spp,
        npix,
        camloc,
        tarloc,
        srf=None,
        sampler="independent",
        rfilter="box",
        up=[0.0, 0.0, 1.0],
        far_clip=10000000,
        idd=None,
        override_name_change=False,
    ):
        """
        Create a generic perspective camera

        Parameters:
        -----------
        spp      : int
                   Number of samples per pixel
        npix     : tuple, int
                   Number of pixels in the X and Y directions
        camloc   : tuple, float
                   Camera location in (X,Y,Z) space in the distance units used during
                   the class instantiation
        tarloc   : tuple, float
                   Target location (where you want to image) in (X,Y,Z) space in the
                   distance units used during the class instantiation
        srf      : Path, str, eradiate.spectral.SpectralResponseFunction, or dict
                   Spectral response function for the camera
        sampler  : str, default="independent"
                   Pseudo random number generator type. Options are: "independent",
                   "stratified", "multijitter", "orthogonal", and "ldsampler"
        rfilter  : str, default="box"
                   Reconstruction filter used to scatter samples on a pixel. Options are:
                   "box", "gaussian", "tent", "mitchell", "catmullrom", and "lanczos"
        up       : tuple, float
                   Up vector of the camera (unitless)
        far_clip : int or float
                   Distance to the far clip plane in the distance units used during the
                   class instantiation, default=10000 km (10000000 m)
        idd      : str or None
                   ID of the camera, if None then the ID is created based on the parameters
                   given. Note that any "." given in the id are converted to "," as Mitsuba
                   (which underpins Eradiate) cannot handle these.
        override_name_change : bool
                               Default is False, overrides the addition of the parameters
                               to the json filename, e.g. test.json -> test_<>_<>.json
        """
        # ID creation or ID checking
        if idd == None:
            if srf is not None:
                srf_idd = self.get_id_from_srf(srf)
            else:
                srf_idd = "none"
            idd = (
                f"hemispherical_{spp}_{npix[0]}-{npix[1]}_{camloc[0]}-{camloc[1]}-"
                + f"{camloc[2]}_{tarloc[0]}-{tarloc[1]}-{tarloc[2]}_{srf_idd}_{sampler}_"
                + f"{rfilter}_{up[0]}-{up[1]}-{up[2]}_{far_clip}"
            )

        if self.angle_units == ureg.degrees:
            fov = 180.0
        else:
            fov = np.pi

        # need to unset and set the filename to avoid writing the file twice
        out_json = self.out_json
        self.out_json = None
        measure = self.perspective_camera(
            spp,
            npix,
            fov,
            camloc,
            tarloc,
            srf,
            sampler,
            rfilter,
            up,
            far_clip,
            idd.replace(".", ","),
        )
        self.out_json = out_json
        if self.out_json:
            jm = JsonManipulation()
            oname = jm.change_json_fname(
                self.out_json, override_name_change, "hemispherical", [idd[14:]]
            )
            jm.write_json(measure, oname)
        return measure

    def sentinel2_msi(
        self,
        spp,
        npix,
        camlocx,
        camlocy,
        tarloc,
        version,
        up=[0.0, 1.0, 0.0],
        idd=None,
        bands="all",
        res="10m",
        override_name_change=False,
    ):
        """
        Create the "measure" instance for Sentinel 2 MSI

        Parameters:
        -----------
        spp      : int
                   Number of samples per pixel
        npix     : tuple, int
                   Number of pixels in the X and Y directions. The Y axis is taken as the along-track
                   direction which is relevant for the FOV calculation.
        camlocx  : float
                   Camera location in (X) space in the distance units used during
                   the class instantiation
        camlocy  : float
                   Camera location in (Y) space in the distance units used during
                   the class instantiation
        tarloc   : tuple, float
                   Target location (where you want to image) in (X,Y,Z) space in the
                   distance units used during the class instantiation
        version  : str
                   Options are "a" or "b" for Sentinel-2A or Sentinel-2B
        up       : tuple, float
                   Up vector of the camera (unitless), default along y axis [0., 1., 0.]
        idd      : str or None
                   ID of the camera, if None then the ID is created based on the parameters
                   given. Note that any "." given in the id are converted to "," as Mitsuba
                   (which underpins Eradiate) cannot handle these.
        bands    : List of str or "all" (default)
                   List of band numbers as strings or "all" to give all bands. Band numbers
                   specified without zero padding.
        res      : str
                   Resolution of the simulation. Default is '10m', other options are '20m' and '60m'.
        override_name_change : bool
                               Default is False, overrides the addition of the parameters
                               to the json filename, e.g. test.json -> test_<>_<>.json
        """
        try:
            vers = version.lower()
        except AttributeError:
            raise IOError("'version' should be a string with either 'a' or 'b'")
        if vers == "a" or vers == "b":
            pass
        else:
            raise IOError("'version' should be a string with either 'a' or 'b'")

        if self.angle_units == ureg.degrees:
            fov = 20.6
        else:
            fov = np.deg2rad(20.6)
        npixels_10m = 10980.0
        npixels_20m = 5490.0
        npixels_60m = 1830.0
        if res == "10m":
            reso = npixels_10m
        elif res == "20m":
            reso = npixels_20m
        elif res == "60m":
            reso = npixels_60m
        else:
            raise IOError("'res' should be '10m', '20m' or '60m'")

        srf_loc = self.srf_path_solving()

        # calculate the actual fov to use
        fovv = fov / reso * npix[0]

        # calculate the camera location based on S2 orbital height
        orbit_height = 786000.0  # in metres
        # earth_radius = 6371000 # mean earth radius
        camloc = (
            float(camlocx),
            float(camlocy),
            orbit_height,
        )  # works of small differences of x and y

        # ID creation or ID checking
        if idd == None:
            if isinstance(bands, list):
                bnds = bands[0]
                for i in bands[1:]:
                    bnds += f"-{i}"
            else:
                bnds = bands

            idd1 = f"sentinel-2{vers}-"
            idd2 = "b{bnds}_"
            idd3 = (
                f"{spp}_{npix[0]}-{npix[1]}_{camloc[0]}-{camloc[1]}"
                + f"-{camloc[2]}_{tarloc[0]}-{tarloc[1]}-{tarloc[2]}"
            )
            iddd = (
                f"sentinel-2{vers}-b{bnds}_{spp}_{npix[0]}-{npix[1]}_{camloc[0]}-{camloc[1]}"
                + f"-{camloc[2]}_{tarloc[0]}-{tarloc[1]}-{tarloc[2]}"
            )

        s2bands = list(np.arange(1, 13).astype(str))
        s2bands.insert(8, "8a")
        sensor_id = f"sentinel_2{vers}-msi-"
        if bands == "all":
            # need to see if you can simulate all S2 bands at once...
            # bandss = eradiate.spectral.SpectralResponseFunction(sensor_id)
            bandss = [srf_loc + sensor_id + b for b in s2bands]
            bandids = [idd1 + idd2.format(bnds=b) + idd3 for b in s2bands]
        else:
            bandss = []
            bandids = []
            for b in bands:
                if b in s2bands:
                    bandss.append(srf_loc + sensor_id + b)
                    bandids.append(idd1 + idd2.format(bnds=b) + idd3)
                else:
                    raise IOError(f"'{b}' is not a recognised {sensor_id} band")

        # need to unset and set the filename to avoid writing the file twice
        out_json = self.out_json
        self.out_json = None

        # measure = [self.perspective_camera(spp, npix, fovv, camloc, tarloc, up=up, idd=bandids[b].replace(".",","), srf=band)
        measure = [
            self.mpdistant_camera(
                spp, npix, camloc, tarloc, idd=bandids[b].replace(".", ","), srf=band
            )
            for b, band in enumerate(bandss)
        ]
        self.out_json = out_json

        if self.out_json:
            jm = JsonManipulation()
            oname = jm.change_json_fname(
                self.out_json, override_name_change, f"sentinel2{version}", [iddd[12:]]
            )
            jm.write_json(measure, oname)

        return measure

    @staticmethod
    def check_var(varname, var, vtype):
        isit = []
        for vt in vtype:
            if isinstance(var, vt):
                isit.append(True)
            else:
                isit.append(False)
        if any(isit):
            pass
        else:
            raise IOError(f"'{varname}' must be one of: {vtype}")

    @staticmethod
    def check_tuple(varname, var, nvar, vtype):
        try:
            if len(var) == nvar:
                pass
            else:
                raise IOError(f"'{varname}' must have {nvar} values")
            for n in range(nvar):
                if isinstance(var[n], vtype):
                    pass
                else:
                    raise IOError(f"'{varname}' must be {vtype}")
        except TypeError:
            raise IOError(f"'{varname}' must be a tuple of {vtype}")

    def get_id_from_srf(self, srf):
        """
        Get an appropriate ID string from the srf input

        Parameters:
        -----------
        srf : Path, str, eradiate.spectral.SpectralResponseFunction, or dict
              Spectral response function for the camera

        Returns:
        --------
        idd : str
              Appropriate ID for the SRF specified
        """
        if isinstance(srf, str):
            idd = split_fname(srf.split("/")[-1])[0]
        elif isinstance(srf, pathlib.Path):
            idd = split_fname(str(srf).split("/")[-1])[0]
        elif self.check_srf_type(srf):
            idd = f"{srf.wmin.magnitude}-{srf.wmax.magnitude}"
        elif isinstance(srf, dict):
            if len(srf["wavelengths"]) == 1:
                idd = str(srf["wavelengths"])
            else:
                idd = f"{srf['wavelengths'][0]}-{srf['wavelengths'][-1]}"
        return idd

    @staticmethod
    def check_srf_type(srf):
        """
        Check what the srf eradiate spectral type is
        """
        members = inspect.getmembers(eradiate.spectral)
        isit = False
        for member in members:
            if isinstance(srf, member[1]):
                isit = True

        return isit

    @staticmethod
    def srf_path_solving():
        """
        Return the srf path location or raise error if not found
        """
        fr_paths = eradiate.fresolver.paths
        srf_loc = 0
        for path in fr_paths:
            if "Caches" in str(path):
                srf_loc = str(path) + "/srf/"

        if srf_loc != 0:
            pass
        else:
            raise AttributeError(
                "It doesn't look like you have a standard location for your SRF data ",
                "- you need to set 'eradiate.fresolver.paths' to point to that location.\n"
                f" Currently your paths are set as: {fr_paths}",
            )
        return srf_loc


class Lidar:
    def __init__(self):
        """Class to house special case of LiDAR sensors"""
        something = 0


class Atmosphere:
    def __init__(self, out_json=None):
        """Class to create different atmospheres"""
        self.out_json = out_json

    def simple_atmosphere(self, mode="ckd", override_name_change=False):
        """
        Create a simple atmosphere; because this is a default atmosphere, the affix "default"
        will be at the end of the filename

        Parameters:
        -----------
        mode                 : str
                               Default is 'ckd', also accepts 'mono'; determines whether results
                               are band integrated ('ckd') or per wavelength ('mono')
        override_name_change : bool
                               Default is False, overrides the addition of the parameters
                               to the json filename, e.g. test.json -> test_<>_<>.json
        """
        atmosp = "MolecularAtmosphere"
        atmos = {"atmosphere": atmosp, "mode": mode}

        if self.out_json:
            jm = JsonManipulation()
            oname = jm.change_json_fname(
                self.out_json, override_name_change, f"simple_atmos", ["default"]
            )
            jm.write_json(atmos, oname)
        return atmos

    def atmosphere_checks(self, atmosphere):
        """Returns the eradiate atmosphere instance based on saved dict if it is a default type"""
        if atmosphere["atmosphere"] == "MolecularAtmosphere":
            if atmosphere["mode"] == "mono":
                eradiate.set_mode("mono")
            elif atmosphere["mode"] == "ckd":
                eradiate.set_mode("ckd")
            return eradiate.scenes.atmosphere.MolecularAtmosphere()


class Spectra:
    def __init__(self, spectra_dir):
        spec_fnames = glob.glob(spectra_dir + "*.*")
        wvs = 0
        refl = 0
        trans = 0

        if len(spec_fnames) == 1:
            spectra = np.loadtxt(spec_fnames[0])
            self.wvs = spectra[:, 0]
            self.refl = spectra[:, 1]
            self.fnames = [spec_fnames[0].split("/")[-1]]
            try:
                self.trans = spectra[:, 2]
            except IndexError:
                self.trans = None
        else:
            wvs = []
            refl = []
            trans = []
            fnames = []
            for i, f in enumerate(spec_fnames):
                spectra = np.loadtxt(f)
                wvs.append(spectra[:, 0])
                refl.append(spectra[:, 1])
                fnames.append(f.split("/")[-1])
                try:
                    trans.append(spectra[:, 2])
                except (IndexError, AttributeError):
                    trans = None

            self.wvs = np.array(wvs).T
            self.refl = np.array(refl).T
            if type(trans) == list:
                self.trans = np.array(trans).T
            else:
                self.trans = trans
            self.fnames = fnames


class DEM:
    """Create a DEM for an eradiate simulation"""

    def __init__(
        self,
        terrain_path_mesh,
        terrain_mesh_spectra,
        background_dem_path_mesh,
        background_dem_spectra,
        centre_offsets=np.array([0.0, 0.0, 0.0]),
    ):
        """
        Creates a DEM for eradiate simulations

        Parameters:
        -----------
        terrain_path_mesh        : str
                                   Path to the directory containing the mesh file for the
                                   local terrain
        terrain_mesh_spectra     : str
                                   Path to the file containing the spectral information for
                                   the local terrain
        background_dem_path_mesh : str
                                   Path to the file containing the background DEM for
                                   the wider area
        background_dem_spectra   : str
                                   Path to the directory containing the spectra for the
                                   background DEM for the wider area
        centre_offsets           : np.arr, float
                                   This puts the 0., 0. (x and y) at the centre of the terrain.
                                   This is carried through to the canopy class when you include
                                   the DEM.
        """
        self.canopy = Canopy()
        self.background_mesh = o3d.io.read_triangle_mesh(background_dem_path_mesh)
        self.terrain_mesh = o3d.io.read_triangle_mesh(terrain_path_mesh)
        self.background_dem_path_mesh = background_dem_path_mesh
        self.centre_offsets = centre_offsets

        # get the spectral information for the background
        self.background_dem_spectra = Spectra(background_dem_spectra)

    def get_max_z_extent(self, mesh, xmin, xmax, ymin, ymax):
        """Get the maximum height of the DEM within a region"""
        verts = np.asarray(mesh.vertices)
        mask = (
            (verts[:, 0] >= xmin)
            * (verts[:, 0] <= xmax)
            * (verts[:, 1] >= ymin)
            * (verts[:, 1] <= ymax)
        )
        vmax = np.max(verts[mask][:, 2])
        return vmax

    def get_meshes_bbox(self):
        """Get the min/max extent of the terrain surface"""
        bbox = self.terrain_mesh.get_axis_aligned_bounding_box()
        terrain = [
            bbox.min_bound[0],
            bbox.max_bound[0],
            bbox.min_bound[1],
            bbox.max_bound[1],
            bbox.min_bound[2],
            bbox.max_bound[2],
        ]
        return terrain

    def get_background_element(self):
        """Get eradiate DEM"""
        bbox = self.get_meshes_bbox()
        z_offset = self.get_max_z_extent(self.background_mesh, *bbox[:4])
        # background_dem_el, bsdfs = self.canopy.make_mesh_tree_elements(
        background_dem_el = self.canopy.make_mesh_tree_elements(
            [self.background_dem_path_mesh],
            "background_dem_mesh",
            self.background_dem_spectra.wvs,
            self.background_dem_spectra.refl,
            self.background_dem_spectra.trans,
        )
        instance_position = [
            np.array(self.centre_offsets) + np.array([0.0, 0.0, z_offset])
        ]
        background_dem_inst_el = self.canopy.make_instanced_canopy_element(
            background_dem_el, instance_position
        )
        bg_dem_inst = background_dem_inst_el[0]
        bg_dem_inst["centre_offsets"] = self.centre_offsets
        return bg_dem_inst


class DEMFromTiff:
    """Create DEM mesh from a tif file (e.g. Copernicus DEM)"""

    def __init__(self, tiffpath, resolution, outfile):
        """
        Create a DEM mesh file that can be used with eradiate

        Parameters:
        -----------
        tiffpath   : str
                     Path to the DEM in geotiff format
        resolution : int
                     Resolution in metres
        outfile    : str
                     Filename (including path) of the output ply file containing
                     the DEM mesh
        """
        if outfile[-4:] == ".ply":
            pass
        else:
            outfile += ".ply"
        self.outfile = outfile
        self.resolution = resolution
        self.dem_arr = rasterio.open(tiffpath).read()

        points = self.prepare_mesh()
        mesh = self.create_mesh(points)

        # write mesh to ply file
        o3d.io.write_triangle_mesh(outfile, mesh)

    @staticmethod
    def create_mesh(points):
        """
        Create mesh using Delaunay triangulation

        Parameters:
        -----------
        points : np.arr, float
                 n x 3 array of points to compute the Delaunay triangulation over. The first
                 and second give the x and y coordinates, while the final column gives the
                 DEM height
        """
        tri = Delaunay(points[:, :2])
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(points)
        mesh.triangles = o3d.utility.Vector3iVector(tri.simplices)
        return mesh

    def prepare_mesh(self):
        """Prepare the inputs into self.reate_mesh"""
        height, width = self.dem_arr.shape[1:]

        # Make centered x and y coordinates from DEM shape
        pixel_center = self.resolution / 2
        x_center = width * self.resolution / 2
        y_center = height * self.resolution / 2
        x = np.arange(0, width) * self.resolution + pixel_center - x_center
        y = np.arange(0, height) * self.resolution + pixel_center - y_center
        X, Y = np.meshgrid(x, y)

        # Make DEM point cloud
        points = np.column_stack((X.ravel(), Y.ravel(), self.dem_arr.ravel()))
        return points


class Canopy:
    """Create an eradiate canopy of 3D trees from mesh files"""

    def __init__(
        self,
        directory=None,
        meshes_dir=None,
        mesh_names=None,
        spectra_dir=None,
        canopy_pos=None,
        dem=None,
        size=None,
        padding=None,
        out_json=None,
        crossref=None,
    ):
        """
        Create an eradiate canopy of 3D trees from mesh files

        Fill in the kwargs to use the automated scene and file creation. For lower level
        control, leave the kwargs blank.

        Note that meshes and spectra files must match. For example, where there is a folder
        in the meshes directory called "trees" and inside there are individual meshes for:
        "tree1", "tree2", ... "treeN", then, where inidividual spectra are assigned to these,
        the corresponding directory in the spectra folder should be called "trees" and the
        files within this directory be called: "tree1", "tree2", ... "treeN" so it can be
        assigned appropriately. Where a single spectra exists in the directory, that will be
        used for all meshes of the same name.

        Parameters:
        -----------
        directory    : str
                       Directory of the scene files to be used
        meshes_dir   : str
                       Name of the meshes directory (relative to "directory")
        mesh_names   : list, str
                       List of strings containing the names of the meshes to be considered.
                       This is used for the spectra as well so must match.
        spectra_dir  : str
                       Name of the spectra directory (relative to "directory")

                       Note: the spectra files should be readable by np.loadtxt using the standard
                       delimiter and be stripped of headers/footers. This assumes that spectra
                       are n x 2 or n x 3 sized with wavelength, reflectance and transmittance
                       (if appropriate) being the two/three columns.
        canopy_pos   : np.arr, float
                       1 x 3 array giving the central location of canopy in the local
                       coordinates - typically [0., 0., 0.] when the positions in the mesh file
                       are to be kept exactly
        dem          : list (containing dictionary); DEM obj
                       DEM object which is derived from the make_instanced_canopy_element
                       in this class
        size         : tuple, int
                       Size of the background surface, the z value is automatically determined (X, Y, Z)
        padding      : int
                       Cloning around the scene. 0 for no cloning, 1 for once around itself, etc.
        out_json     : str
                       Filename (including directory) of the .json file to write to.
        crossref     : str or None (default)
                       Folder containing files which have two columns which relate the individual mesh files (first column) to the
                       spectra filename (no directory for either; second column). This is only needed when there is a
                       non-standard relationship between the meshes and spectra. You don't need this when you have
                       e.g. n meshes and n spectras which are in the same order. The file should not have header/footer
                       information and be comma delimited. The filenames should have the name of the meshes that they correspond to
                       at the start of the filename (e.g. "leaves_"...".txt").

        If the kwargs are used, a file containing the canopy scene is written to 'out_json'
        """
        # unless all arguments are given then assume lower level control is needed
        if (
            directory is not None
            and meshes_dir is not None
            and mesh_names is not None
            and spectra_dir is not None
            and canopy_pos is not None
            and dem is not None
            and size is not None
            and padding is not None
            and out_json is not None
        ):
            if directory[-1] != "/":
                directory += "/"
            if meshes_dir[-1] != "/":
                meshes_dir += "/"
            if spectra_dir[-1] != "/":
                spectra_dir += "/"

            if np.shape(canopy_pos) == (1, 3):
                pass
            else:
                raise IOError("'canopy_pos' must have a shape of n x 3")

            # loop over mesh directories, grab the spectra and create canopy elements
            self.mesh_names = mesh_names
            can_el = []
            bsdfs = {}
            for mesh in mesh_names:
                if mesh[-1] != "/":
                    mesh += "/"
                mesh_dir = directory + meshes_dir + mesh
                spec_dir = directory + spectra_dir + mesh

                spectra = Spectra(spec_dir)
                mesh_fnames = glob.glob(mesh_dir + "*.ply") + glob.glob(
                    mesh_dir + "*.obj"
                )

                # els, bsdf = self.make_mesh_tree_elements(mesh_fnames,
                els = self.make_mesh_tree_elements(
                    mesh_fnames,
                    mesh[:-1],
                    spectra.wvs,
                    spectra.refl,
                    spectra.fnames,
                    spectra.trans,
                    crossref,
                )
                inst_can_el = self.make_instanced_canopy_element(
                    els, canopy_pos, dem["instance_positions"][0]
                )[0]
                can_el.append(inst_can_el)

            # add DEM to canopy elements
            del dem["centre_offsets"]
            can_el.append(dem)
            can_scen = self.create_canopy(size, padding, can_el)

            # write to the scenario file
            jm = JsonManipulation()
            jm.write_json(can_scen, out_json)
        else:
            pass

    def spectra(self, wavelengths, values, stype="reflectance"):
        """
        Returns the completed dictionary for reflectance or transmittance

        Parameters:
        -----------
        wavelengths : np.arr, float or int
                      n x 1 numpy array containing the wavelength values
        values      : np.arr, float
                      n x 1 numpy array containing the spectra values
        stype       : str
                      The type of spectra it is (default='reflectance') -
                      can be either 'reflectance' or 'transmittance'
        """
        if values is not None:
            pass
        else:
            return None
        if stype == "reflectance" or stype == "transmittance":
            pass
        else:
            raise IOError(
                f"'{stype}' is not a recognised spectra type ('reflectance'"
                " or 'transmittance are the only options)"
            )
        if type(wavelengths) == np.ndarray:
            pass
        else:
            raise IOError(
                f"'wavelengths' should be of type 'np.ndarray', instead got {type(wavelengths)}"
            )
        if type(values) == np.ndarray:
            pass
        else:
            raise IOError(
                f"'values' should be of type 'np.ndarray', instead got {type(values)}"
            )
        if len(np.shape(wavelengths)) == 1:
            pass
        else:
            raise IOError("'wavelengths' should be 1-dimensional")
        if len(np.shape(values)) == 1:
            pass
        else:
            raise IOError("'values' should be 1-dimensional")
        if wavelengths.dtype == float or wavelengths.dtype == int:
            pass
        else:
            raise IOError("'wavelengths' should be an array of floats or ints")
        if values.dtype == float:
            pass
        else:
            raise IOError("'values' should be an array of floats")

        sdict = {
            "quantity": stype,
            "wavelengths": wavelengths,
            "wavelengths_units": "nanometers",
            "values": values,
            "type": "interpolated",
        }
        return sdict

    def spectra_wrapper(self, reflectance, transmittance=None):
        """
        Put spectra into the appropriate wrapper

        Parameters:
        -----------
        reflectance   : dict
                        dictionary output from self.spectra
        transmittance : dict, None
                        Optional (default=None), if given then dictionary output from self.spectra
        """
        if type(reflectance) == dict:
            pass
        else:
            raise IOError(
                f"'reflectance' should be a dict, instead given a '{type(reflectance)}'"
            )

        if transmittance is not None:
            if type(transmittance) == dict:
                pass
            else:
                raise IOError(
                    f"'transmittance' should be a dict, instead given a '{type(trasmittance)}'"
                )

            bsdf = {
                "type": "translucent",
                "reflectance": reflectance,
                "transmittance": transmittance,
            }
        else:
            bsdf = {"type": "lambertian", "reflectance": reflectance}
        return bsdf

    def make_mesh_tree_element(self, filename, idd, bsdf):
        """
        Method to construct a mesh tree element in correct dictionary form

        Parameters
        ----------
        filename      : str
                        Meshfile of the QSM (typically in .ply format)
        idd           : int
                        Identification number of the tree
        bsdf          : dict or str
                        Dictionary containing the reflectance/transmittance information or
                        string containing the id of the shared bsdf information

        Returns
        -------
        element       : dict
                        Eradiate compatible dictionary for a canopy element
        """
        # Define and populate element
        element = {
            "id": idd,
            "mesh_filename": pathlib.Path(os.path.abspath(filename)),
            "reflectance": bsdf["reflectance"],
        }
        try:
            element["transmittance"] = bsdf["transmittance"]
        except KeyError:
            pass

        return element

    def make_mesh_tree_elements(
        self,
        fnames,
        idd,
        wavelengths,
        reflectances,
        sfnames,
        transmittances=None,
        crossref=None,
    ):
        """
        Method to create a list of mesh tree elements based on a directory of files

        Parameters
        ----------
        fnames         : List[str]
                         List of strings containing the mesh files for the canopy elements
        idd            : str
                         Name of component (e.g. 'leaf', 'wood', ...)
        wavelengths    : np.arr, float
                         n x 1 array of wavelength values
        reflectances   : np.arr, float
                         n x 1 or n x len(fnames) array of reflectance values that match with the corresponding
                         mesh file (assumes the order of fnames and reflectances match)
        sfnames        : list[str]
                         List of strings containing the names of all the spectra files.
        transmittances : [optional] np.arr, float or None
                         n x 1 or n x len(fnames) array of transmittance values that match with the corresponding
                         mesh file (assumes the order of fnames and transmittances match)
        crossref       : [optional] str or None (default)
                         Folder with files containing two columns which relate the individual mesh files (first column) to the
                         spectra filename (no directory for either; second column). This is only needed when there is a
                         non-standard relationship between the meshes and spectra. You don't need this when you have
                         e.g. n meshes and n spectras which are in the same order. The file should not have header/footer
                         information and be comma delimited. The filenames should have the name of the meshes that they correspond to
                         at the start of the filename (e.g. "leaves_"...".txt").

        Returns
        -------
        elements : list[dict]
                   List of Eradiate compatible dictionaries representing mesh elements (mesh + spectra)
        """
        cref = self.crossref(crossref, idd)

        # Loop over all files and append in list
        elements = []
        for i, filename in enumerate(fnames):
            id_i = idd + str(i)
            fname = filename.split("/")[-1]
            ind = cref(fname, i, sfnames)
            try:
                refl = self.spectra(
                    wavelengths[:, ind], reflectances[:, ind], "reflectance"
                )
                try:
                    tran = self.spectra(
                        wavelengths[:, ind], transmittances[:, ind], "transmittance"
                    )
                except TypeError:
                    tran = None
            except (TypeError, IndexError):
                refl = self.spectra(wavelengths, reflectances, "reflectance")
                tran = self.spectra(wavelengths, transmittances, "transmittance")

            bsdf = self.spectra_wrapper(refl, tran)
            element = self.make_mesh_tree_element(filename, id_i, bsdf)
            elements.append(element)

        return elements

    def crossref(self, cref_dir, mesh_name):
        """
        Method to assign a spectra to a mesh file

        Parameters
        ----------
        cref_dir  : str
                    Path to the folder containing the crossref file. The file should be have two columns,
                    with the first giving the mesh filename without the directory, and the second giving
                    the spectral filename without the directory. The file should have no header/footer
                    information and be comma delimited.
        mesh_name : str
                    Name of the mesh which should also be in the filename of the crossref file.
        """

        def cref(fname, index, sfnames):
            return index

        if cref_dir:
            if cref_dir[:-1] == "/":
                pass
            else:
                cref_dir += "/"

            fpath = glob.glob(cref_dir + f"*{mesh_name}*")

            if len(fpath) == 1:
                filename = fpath[0]
            elif len(fpath) == 0:
                return cref
            else:
                raise IOError(
                    f"More than one crossref file was found in {cref_dir} for {mesh_name}"
                )

            crefop = np.loadtxt(filename, delimiter=",", dtype=str)
            if len(crefop.shape) == 1:
                crefop = np.array([crefop])

            def cref(fname, index, sfnames):
                index = np.where(crefop[:, 0] == fname)[0]
                if len(index) == 1:
                    pass
                elif len(index) == 0:
                    raise IOError(f"No entries for {fname} found in {filename}")
                else:
                    raise IOError(
                        f"More than one entry for {fname} found in {filename}"
                    )
                ind = int(index[0])
                return int(np.where(np.array(sfnames) == crefop[:, 1][ind])[0][0])

        return cref

    def make_instanced_canopy_element(
        self,
        mesh_tree_elements,
        instance_position,
        centre_offset=np.array([0.0, 0.0, 0.0]),
    ):
        """
        Method to create a canopy element

        Parameters
        ----------
        mesh_tree_elements : list[dict]
                             List of dictionaries containing Eradiate compatible tree elements
        instance_position  : np.arr, float
                             Position of the canopy centre
        centre_offsets     : np.arr, float (default = np.array([0.,0.,0.])
                             Offsets to be applied to the default position

        Returns
        -------
        canopy_element     : List[dict]
                             List with dictionary containing the canopy element
        """

        canopy_element = [
            {
                "instance_positions": instance_position + centre_offset,
                "type": "instanced",
                "canopy_element": {
                    "type": "mesh_tree",
                    "id": mesh_tree_elements[0]["id"][:-1],
                    "mesh_tree_elements": mesh_tree_elements,
                },
            }
        ]

        return canopy_element

    def create_canopy(self, size, padding, elements):
        """
        Method to create the canopy object

        Parameters
        ----------
        size     : tuple, int
                   Size of the background surface, the z value is automatically determined (X, Y, Z)
        padding  : int
                   Cloning around the scene. 0 for no cloning, 1 for once around itself, etc.
        elements : List of dicts
                   List of dictionaries containing the canopy elements and dem (each dictionary is typically
                   an output of self.instanced_canopy_element
        """
        canopy = {
            "type": "discrete_canopy",
            "size": size,
            "padding": padding,
            "instanced_canopy_elements": elements,
        }

        return canopy


class CreateExperiment:
    def __init__(
        self,
        canopy,
        background_surface,
        atmosphere,
        illumination,
        measure,
        result_fname=None,
    ):
        """
        Create Eradiate experiment which brings together all the simulation elements and
        runs the simulation

        Parameters
        ----------
        canopy             : list of dicts
                             List of dictionaries giving the canopy elements
        background_surface : dict
                             Dictionary giving the background surface. Typically this will
                             be None because the DEM is specified in the canopy dictionary
        atmosphere         : dict
                             Dictionary giving the atmospheric parameters
        illumination       : dict
                             Dictionary giving the illumination parameters
        measure            : dict
                             Dictionary giving the sensor parameters
        result_fname       : str or None
                             The filename of where to store the results of the simulation
        """
        exp = eradiate.experiments.CanopyAtmosphereExperiment(
            surface=background_surface,
            canopy=canopy,
            atmosphere=atmosphere,
            illumination=illumination,
            measures=measure,
        )

        self.result = eradiate.run(exp)

        if result_fname != None:
            try:
                self.result.to_netcdf(result_fname)
            except:
                # sort out the error handling here
                raise IOError(
                    f"{outfile} is probably not a valid filename for a netCDF file, "
                    "already exists, or something else"
                )

    def return_result(self):
        return self.result


class RunSimulations:
    def __init__(
        self,
        canopies,
        background_surfaces,
        atmospheres,
        illuminations,
        measures,
        project_dirs,
        results_dir="",
        order="all",
    ):
        """
        This class is designed to run many simulations at once

        Parameters:
        -----------
        canopies            : list of str
                              The canopy files to be simulated (json)
        background_surfaces : list of str
                              The background surface files to be used (json)
        atmospheres         : list of str
                              The atmosphere files to be used (json)
        illuminations       : list of str
                              The illumination files to be used (json)
        measures            : list of str
                              The measure files to be used (json)
        project_dirs        : list of str
                              Path to the top level of the forest scene (i.e. inside should be the meshes
                              and spectra)
        results_dir         : str, default: ""
                              Path to the directory for where the results will be saved. Default is the
                              current working folder or where the shell script is executed from.
        order               : str, 'all' or 'sequential', default: 'all'
                              Where multiple filenames are given for a specific input, this determines
                              whether the simulations are conducted for all permutations or sequentially.
                              Note that when 'sequential' is selected all inputs need to have the same
                              length.

        Note that name of the results files are created from the names of the scene and
        measurement files so make sure these are named accordingly so that the same name for
        different simulations doesn't occur.
        """

        if len(canopies) != len(project_dirs):
            raise IOError("Length of 'canopies' and 'project_dirs' must be the same")

        self.jm = JsonManipulation()
        components = [
            atmospheres,
            canopies,
            background_surfaces,
            illuminations,
            measures,
        ]

        if order == "all":
            self.simulations = list(itertools.product(*components))
            self.len_sims = len(self.simulations)

        elif order == "sequential":
            len_can = len(canopies)
            len_bac = len(background_surfaces)
            len_atm = len(atmospheres)
            len_ill = len(illuminations)
            len_mea = len(measures)
            len_all = np.array([len_can, len_bac, len_atm, len_ill, len_mea])
            itemnam = np.array(
                [
                    "canopies",
                    "background_surfaces",
                    "atmospheres",
                    "illuminations",
                    "measures",
                ]
            )

            # the number of items should be equal across all simulation components
            len_uni = len(np.unique(len_all))
            if len_uni == 1:
                pass
            else:
                raise IOError("All simulation components must be the same length")

            self.simulations = list(zip(*components))
            self.len_sims = len(self.simulations)

        else:
            raise IOError(
                f"'{order}' is not a valid option for kwarg 'order'. Options are either"
                "'all' or 'sequential'."
            )

        # match up the project directory to the scene after organisation
        for j, sim in enumerate(self.simulations):
            self.simulations[j] += tuple(project_dirs[canopies.index(sim[1])])

        self.result_fnames = self.get_results_fnames(results_dir)

    def run_simulation(
        self,
        atmosphere,
        canopy,
        background_surface,
        illumination,
        measure,
        result_fname,
    ):
        """
        Run simulation based on filenames for each component

        Parameters:
        -----------
        atmosphere         : str
                             Filename giving the atmospheric parameters in a json file
        canopy             : str
                             Filename giving the canopy parameters in a json file
        background_surface : str
                             Filename giving the background surface parameters in a json file
        illumination       : str
                             Filename giving the illumination parameters in a json file
        measure            : str
                             Filename giving the measurement parameters in a json file
        result_fname       : str
                             Filename of the simulation output.
        """
        atmos = self.jm.read_json(atmosphere)
        canop = self.jm.read_json(canopy)
        bgsur = self.jm.read_json(background_surface)
        illum = self.jm.read_json(illumination)
        measu = self.jm.read_json(measure)

        exp = eradiate.experiments.CanopyAtmosphereExperiment(
            surface=bgsur,
            canopy=canop,
            atmosphere=atmos,
            illumination=illum,
            measures=measu,
        )

        if result_fname[-3:] != ".nc":
            result_fname += ".nc"

        result = eradiate.run(exp)
        log_head = self.get_logfile_header()

        if type(result) != dict:
            result.attrs["atmosphere"] = atmosphere
            result.attrs["canopy"] = canopy
            result.attrs["background_surface"] = background_surface
            result.attrs["illumination"] = illumination
            result.attrs["measure"] = measure

            result.attrs["user"] = log_head[0]
            result.attrs["date_time"] = log_head[1]
            result.attrs["eradiate_version"] = log_head[2]
            result.attrs["branch"] = log_head[3]
            result.attrs["commit"] = log_head[4]
            result.attrs["git_url"] = log_head[5]
            result.to_netcdf(result_fname)
        else:
            bnames = [x["srf"].attrs["band"] for x in measu]
            ex = EradiateXarray(
                atmosphere=atmosphere,
                canopy=canopy,
                background_surface=background_surface,
                illumination=illumination,
                measure=measure,
                usr=log_head[0],
                time=log_head[1],
                eradiate_version=log_head[2],
                branch=log_head[3],
                commit=log_head[4],
                git_url=log_head[5],
            )
            ex.merge_eradiate_bands_with_padding(
                result.values(), result_fname, band_names=bnames
            )

    def run_simulations(self):
        """
        Run the simulations given to the instantiation of this class sequentially
        """
        for i in range(self.len_sims):
            self.run_simulation(
                self.simulations[i][0],
                self.simulations[i][1],
                self.simulations[i][2],
                self.simulations[i][3],
                self.simulations[i][4],
                self.result_fnames[i],
            )

    def create_python_slurm(
        self,
        atmosphere,
        canopy,
        background_surface,
        illumination,
        measure,
        res_name,
        project_dir=os.environ["ERADIATE_PROJECT_DIR"],
    ):
        """
        Create the python string to run the simulation (typically to be passed to slurm).

        N.B. all filenames are relative to where the simulations will be run.

        Parameters:
        -----------
        canopy             : str
                             Filename for the canopy dictionary
        background_surface : str
                             Filename for the background surface dictionary
        atmosphere         : str
                             Filename for the atmosphere dictionary
        illumination       : str
                             Filename for the illumination dictionary
        measure            : str
                             Filename for the measure dictionary
        res_name           : str
                             Name of the results file of the simulation
        project_dir        : str, default is the path set in os.environ["ERADIATE_PROJECT_DIR"]
                             Path to the top level directory of the scene (i.e. inside should be
                             the meshes and spectra folders
        """
        python_str = (
            f"python run_simulation.py {canopy} {background_surface} {atmosphere} "
            f"{illumination} {measure} {res_name} {project_dir}"
        )
        return python_str

    def slurm_output(self, slurm_outfile, ram_gb, time, queue_name, other_args):
        """
        Create a bash script which runs the simulations on a slurm compatible supercomputer

        Bases the job and log filenames from the result_fnames given to the class instantiation.
        Note that the log files this method is talking about will be the output of slurm, not
        the log file from the simulations. As a result, the slurm log files will have the format
        "<...>_log.log" whereas the regular log files will just be "<>.log".

        Parameters:
        -----------
        slurm_outfile : str
                        The filename of the bash script to launch the slurm processes
        ram_gb        : int
                        RAM required in MB
        time          : tuple, int
                        Tuple giving the hours, minutes and seconds of the simulations duration
        queue_name    : str
                        Name of the queue in which to launch the job
        other_args    : str
                        String giving the additional arguments that can be associated with the
                        job
        """
        core_str = (
            f'sbatch -p "{queue_name}" --time={time[0]:02d}:{time[1]:02d}:{time[2]:02d} '
            + f"--mem={ram_gb} --ntasks=1 {other_args} "
        )
        addi_str = '--output="{log_name}" --job-name="{job_name}" --wrap="{python_str}"'

        with open(slurm_outfile, "w") as outfile:
            for i in range(self.len_sims):
                python_str = self.create_python_slurm(
                    self.simulations[i][0],
                    self.simulations[i][1],
                    self.simulations[i][2],
                    self.simulations[i][3],
                    self.simulations[i][4],
                    self.result_fnames[i],
                )
                job_name = self.result_fnames[i].split("/")[-1][:-3]
                log_name = self.result_fnames[i][:-3] + "_log.log"
                addi = addi_str.format(
                    log_name=log_name, job_name=job_name, python_str=python_str
                )
                outfile.write(core_str + addi + "\n")

    def create_info_file(self):
        """
        Create a file containing a summary of the key information about the simulation for
        cross-referencing

        Whatever the simulation name is, the info file has a '.info' appended.
        """
        something = 0

    def get_logfile_header(self):
        usr = getpass.getuser()
        tme = time.strftime("%Y-%m-%d %H:%M:%S")
        evers = eradiate.__version__

        # get git details
        branch_query = subprocess.Popen(
            ["git", "branch"], text=True, stdout=subprocess.PIPE
        )
        branches = branch_query.stdout.read().split()
        branch = branches[branches.index("*") + 1]
        commit_query = subprocess.Popen(
            ["git", "rev-parse", "HEAD"], text=True, stdout=subprocess.PIPE
        )
        commit = commit_query.stdout.read()[:-1]
        git_url_query = subprocess.Popen(
            ["git", "config", "--get", "remote.origin.url"],
            text=True,
            stdout=subprocess.PIPE,
        )
        git_url_details = git_url_query.stdout.read()
        git_url = git_url_details.split("@")[1][:-1].replace(":", "/")

        return (usr, tme, evers, branch, commit, git_url)

    def get_results_fnames(self, results_dir):
        """
        Get the filenames of the simulations based on the input files

        Parameters:
        -----------
        results_dir : str
                      The path where the results files should be stored

        """
        rfnames = []
        for i in range(self.len_sims):
            rfname = (
                f"{self.simulations[i][1][:-5].split('/')[-1]}__{self.simulations[i][3][:-5].split('/')[-1]}__"
                f"{self.simulations[i][2][:-5].split('/')[-1]}__{self.simulations[i][0][:-5].split('/')[-1]}__"
                f"{self.simulations[i][4][:-5].split('/')[-1]}.nc"
            )
            rfnames.append(results_dir + rfname)
        return rfnames


class ParseSimulationInfo:
    def __init__(self, dirs, keywords=None, keywordvalues=None):
        """
        Parse through directories of simulation info files and search for simulations where
        particular keywords are present or where specific parameters have been set to specific
        values.

        Parameters:
        -----------
        dirs          : list of str
        keywords      : list of str
        keywordvalues : dict
                        Combinations of variables with their parameter values
        """
        something = 0


class EradiateXarray:
    def __init__(
        self,
        atmosphere=None,
        canopy=None,
        background_surface=None,
        illumination=None,
        measure=None,
        usr=None,
        time=None,
        eradiate_version=None,
        branch=None,
        commit=None,
        git_url=None,
    ):
        """
        Class to organise simulated data coming out of Eradiate and store it better, e.g.
        So multiple bands from a satellite sensor can be stored in a single dataset.
        """
        self.atmosphere = atmosphere
        self.canopy = canopy
        self.background_surface = background_surface
        self.illumination = illumination
        self.measure = measure
        self.usr = usr
        self.time = time
        self.eradiate_version = eradiate_version
        self.branch = branch
        self.commit = commit
        self.git_url = git_url

    @staticmethod
    def pad_along_axis(arr, target_len, axis=0, fill_value=np.nan, dtype=None):
        """
        Pad an array at the end of one axis until it reaches target_len.

        Parameters
        ----------
        arr : np.ndarray or xarray.DataArray
            Input array.
        target_len : int
            Desired size along the chosen axis.
        axis : int, default=0
            Axis to pad.
        fill_value : scalar, default=np.nan
            Value used for padding.
        dtype : numpy dtype, optional
            Output dtype. If None, inferred from input.

        Returns
        -------
        np.ndarray
            Padded array.
        """
        data = arr.values if hasattr(arr, "values") else np.asarray(arr)

        if dtype is None:
            if np.issubdtype(data.dtype, np.floating):
                dtype = data.dtype
            else:
                dtype = np.float64 if np.isnan(fill_value) else data.dtype

        current_len = data.shape[axis]

        if current_len > target_len:
            raise ValueError(
                f"Current length {current_len} exceeds target length {target_len} on axis {axis}"
            )

        if current_len == target_len:
            return data.astype(dtype, copy=False)

        pad_width = [(0, 0)] * data.ndim
        pad_width[axis] = (0, target_len - current_len)

        return np.pad(
            data.astype(dtype, copy=False),
            pad_width=pad_width,
            mode="constant",
            constant_values=fill_value,
        )

    @staticmethod
    def allclose_or_equal(a, b):
        a_val = a.values if hasattr(a, "values") else np.asarray(a)
        b_val = b.values if hasattr(b, "values") else np.asarray(b)

        if np.issubdtype(a_val.dtype, np.number) and np.issubdtype(
            b_val.dtype, np.number
        ):
            return np.allclose(a_val, b_val)
        return np.array_equal(a_val, b_val)

    def merge_eradiate_bands_with_padding(
        self,
        band_dsets,
        out_file,
        band_names=None,
        fill_value=np.nan,
        compression=False,
    ):
        """
        Merge a list of Eradiate-like xarray.Dataset objects into one Dataset
        with a 'band' dimension, padding spectral dimensions when needed.

        Parameters:
        -----------
        band_dsets    : list of Xarray Datasets
                        Output of Eradiate for simulated multi-band data
        out_file      : str
                        Name of the output netCDF file to create
        band_names    : list of str/int/float or None
                        Names of the bands to use
        fill_value    : float; int; nan; default nan
                        Set the fill_value for the where the spectral coords have
                        different lengths in each band. This method creates a common
                        spectral coordinate length.
        compression   : bool; default False
                        Determine whether the output netCDF file should be compressed


        Assumptions
        -----------
        Shared across bands:
            sza, saa, y_index, x_index, y, x, vza, vaa

        Variable per band:
            w, bin_wmin, bin_wmax, w_srf

        Variables:
            irradiance(sza, saa, w)
            radiance(w, y_index, x_index, saa, sza)
            radiance_srf(y_index, x_index, saa, sza)
        """
        if not band_dsets:
            raise ValueError("band_dsets is empty")

        band_dsets = list(band_dsets)
        ref = band_dsets[0]
        nbands = len(band_dsets)

        if band_names is None:
            band_names = np.arange(nbands)
        band_names = np.asarray(band_names)

        if len(band_names) != nbands:
            raise ValueError("band_names length must match number of datasets")

        shared_coords = ["sza", "saa", "y_index", "x_index", "y", "x", "vza", "vaa"]
        required_vars = ["irradiance", "radiance", "radiance_srf"]
        spectral_coords = ["w", "bin_wmin", "bin_wmax", "w_srf"]

        w_lengths = []

        for i, ds in enumerate(band_dsets):
            for name in required_vars:
                if name not in ds:
                    raise ValueError(f"Dataset {i} is missing data variable '{name}'")

            for name in shared_coords:
                if name not in ds.coords:
                    raise ValueError(f"Dataset {i} is missing coordinate '{name}'")
                if not self.allclose_or_equal(ds[name], ref[name]):
                    raise ValueError(
                        f"Dataset {i} has non-matching shared coordinate '{name}'"
                    )

            for name in spectral_coords:
                if name not in ds.coords:
                    raise ValueError(f"Dataset {i} is missing coordinate '{name}'")

            w_lengths.append(len(ds["w"]))

        nw_max = max(w_lengths)
        band = xr.DataArray(band_names, dims="band", name="band")

        radiance_srf = xr.concat(
            [ds["radiance_srf"] for ds in band_dsets],
            dim=band,
        )

        radiance_data = np.stack(
            [
                self.pad_along_axis(
                    ds["radiance"], nw_max, axis=0, fill_value=fill_value
                )
                for ds in band_dsets
            ],
            axis=0,
        )

        irradiance_data = np.stack(
            [
                self.pad_along_axis(
                    ds["irradiance"], nw_max, axis=-1, fill_value=fill_value
                )
                for ds in band_dsets
            ],
            axis=0,
        )

        w = xr.DataArray(
            np.stack(
                [
                    self.pad_along_axis(ds["w"], nw_max, axis=0, fill_value=fill_value)
                    for ds in band_dsets
                ],
                axis=0,
            ),
            dims=("band", "w_dim"),
            coords={"band": band_names},
            name="w",
        )

        bin_wmin = xr.DataArray(
            np.stack(
                [
                    self.pad_along_axis(
                        ds["bin_wmin"], nw_max, axis=0, fill_value=fill_value
                    )
                    for ds in band_dsets
                ],
                axis=0,
            ),
            dims=("band", "w_dim"),
            coords={"band": band_names},
            name="bin_wmin",
        )

        bin_wmax = xr.DataArray(
            np.stack(
                [
                    self.pad_along_axis(
                        ds["bin_wmax"], nw_max, axis=0, fill_value=fill_value
                    )
                    for ds in band_dsets
                ],
                axis=0,
            ),
            dims=("band", "w_dim"),
            coords={"band": band_names},
            name="bin_wmax",
        )

        w_srf = xr.DataArray(
            [float(ds["w_srf"].values) for ds in band_dsets],
            dims="band",
            coords={"band": band_names},
            name="w_srf",
        )

        radiance = xr.DataArray(
            radiance_data,
            dims=("band", "w_dim", "y_index", "x_index", "saa", "sza"),
            name="radiance",
        )

        irradiance = xr.DataArray(
            irradiance_data,
            dims=("band", "sza", "saa", "w_dim"),
            name="irradiance",
        )

        out = xr.Dataset(
            data_vars={
                "radiance_srf": radiance_srf,
                "radiance": radiance,
                "irradiance": irradiance,
                "w": w,
                "bin_wmin": bin_wmin,
                "bin_wmax": bin_wmax,
                "w_srf": w_srf,
            },
            coords={
                "band": band,
                "w_dim": np.arange(nw_max),
                "sza": ref["sza"],
                "saa": ref["saa"],
                "y_index": ref["y_index"],
                "x_index": ref["x_index"],
                "y": ref["y"],
                "x": ref["x"],
                "vza": ref["vza"],
                "vaa": ref["vaa"],
            },
            attrs=dict(ref.attrs),
        )

        out.attrs["band_count"] = nbands
        out.attrs["spectral_length_max"] = nw_max
        out.attrs["spectral_lengths"] = w_lengths
        out.attrs["fill_value"] = fill_value
        out.attrs["description"] = (
            "Merged Eradiate outputs with band-dependent spectral grids, "
            "padded to a common spectral length."
        )

        out.attrs["atmosphere"] = self.atmosphere
        out.attrs["canopy"] = self.canopy
        out.attrs["background_surface"] = self.background_surface
        out.attrs["illumination"] = self.illumination
        out.attrs["measure"] = self.measure

        out.attrs["user"] = self.usr
        out.attrs["date_time"] = self.time
        out.attrs["eradiate_version"] = self.eradiate_version
        out.attrs["branch"] = self.branch
        out.attrs["commit"] = self.commit
        out.attrs["git_url"] = self.git_url

        if compression:
            encoding = {var: {"zlib": True, "complevel": 4} for var in out.data_vars}
            out.to_netcdf(out_file, encoding=encoding, engine="netcdf4")
        else:
            out.to_netcdf(out_file)

        return out


class NumpyEncoder(json.JSONEncoder):
    """Encodes numpy data types for JSON - and now other types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return {"__type__": "np.int", "value": int(obj)}
        elif isinstance(obj, np.floating):
            return {"__type__": "np.float", "value": float(obj)}
        elif isinstance(obj, np.ndarray):
            return {"__type__": "np.ndarray", "value": obj.tolist()}
        elif isinstance(obj, pathlib.PosixPath):
            if "eradiate-v" in str(obj) and "/srf" in str(obj):
                cache_dir = pathlib.PosixPath(eradiate.asset_manager.install_dir)
                return {
                    "__type__": "pathlib.PosixPath",
                    "value": str(obj.relative_to(cache_dir)),
                    "category": "srf",
                }
            else:
                return {
                    "__type__": "pathlib.PosixPath",
                    "value": str(obj.relative_to(os.environ["ERADIATE_PROJECT_DIR"])),
                    "category": "other",
                }
        elif isinstance(obj, ureg.Quantity):
            if isinstance(obj.magnitude, (float, int)):
                return {
                    "__type__": "ureg.Quantity",
                    "value": float(obj.magnitude),
                    "value2": str(obj.units),
                    "original": "number",
                }
            elif isinstance(obj.magnitude, (tuple, list)):
                return {
                    "__type__": "ureg.Quantity",
                    "value": obj.magnitude,
                    "value2": str(obj.units),
                    "original": "tuple/list",
                }
            elif isinstance(obj.magnitude, np.ndarray):
                return {
                    "__type__": "ureg.Quantity",
                    "value": obj.magnitude.tolist(),
                    "value2": str(obj.units),
                    "original": "array",
                }
        elif isinstance(obj, eradiate.scenes.spectra.SolarIrradianceSpectrum):
            return {
                "__type__": "SolarIrradianceSpectrum",
                "value": "SolarIrradianceSpectrum",
            }
        return super().default(obj)


def numpy_object_hook(dct):
    """Decodes numpy types from JSON - adapted to decode other types"""
    t = dct.get("__type__")
    atm = Atmosphere()
    try:
        if dct["category"] == "srf":
            return eradiate.data.fresolver.load_dataset(dct["path"])
    except KeyError:
        pass
    try:
        return atm.atmosphere_checks(dct)
    except KeyError:
        pass
    if t == "np.int":
        return np.int64(dct["value"])
    elif t == "np.float":
        return np.float64(dct["value"])
    elif t == "np.ndarray":
        return np.array(dct["value"])
    elif t == "pathlib.PosixPath":
        if dct["category"] == "srf":
            am = pathlib.Path(eradiate.asset_manager.install_dir)
            return am / pathlib.PosixPath(dct["value"])
        elif dct["category"] == "other":
            return pathlib.Path(os.environ["ERADIATE_PROJECT_DIR"] + dct["value"])
        else:
            raise IOError("I don't know how you got here")
    elif t == "ureg.Quantity":
        if dct["original"] == "number" or dct["original"] == "tuple/list":
            return dct["value"] * ureg.Quantity(dct["value2"])
        elif dct["original"] == "array":
            return np.array(dct["value"]) * ureg.Quantity(dct["value2"])
    elif t == "SolarIrradianceSpectrum":
        return eradiate.scenes.spectra.SolarIrradianceSpectrum()
    return dct


def split_fname(fname):
    return fname.rsplit(".", 1)


class JsonManipulation:
    def __init__(self):
        something = 0

    def read_json(self, fname):
        """
        Read a .json file

        Parameters:
        -----------
        fname : str
                Filename to read
        """
        return json.loads(open(fname).read(), object_hook=numpy_object_hook)

    def write_json(self, element, fname):
        """
        Write to a .json file

        Parameters:
        -----------
        element : dict
                  Dictionary containing Eradiate scene element in eradiate
                  format
        fname   : str
                  Filename for the output .json file

        """
        with open(fname, "w") as f:
            json.dump(element, f, ensure_ascii=False, cls=NumpyEncoder)

    def change_json_fname(
        self, original_fname, override_name_change, str_prefix=None, pars=None
    ):
        """
        Change json filename to the specific parameters used

        Parameters:
        -----------
        original_fname : str
                         The original json filename
        override_name_change : bool
                               True or False depending on whether the overide
                               should happen
        str_prefix           : str
                               The prefix to follow the end of the
                               original_fname but before the .json and pars
        pars                 : list or None
                               List of parameter values to add (each should be
                               short). If None then just the string prefix is
                               used.

        Returns              : str
                               The new filename (e.g. if original_fname is
                               "scenario.json" and the pars are [1, 2, 3] this
                               will return "scenario_1_2_3.json"
        """
        if not override_name_change:
            if str_prefix:
                oj = split_fname(original_fname)
                oname = oj[0] + f"_{str_prefix}"
                try:
                    for par in pars:
                        oname += f"_{par}"
                except TypeError:
                    pass
                oname += ".json"
                return oname
            else:
                raise IOError("You need to specify 'str_prefix' to use this")
        else:
            return original_fname
