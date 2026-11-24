import logging

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay, KDTree

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

"""
Functions used for interpolation of data from WEatherGenerator output 
to obs points for the MetNor verif tool
"""


def convert_coordinates(coords: np.typing.NDArray) -> np.typing.NDArray:
    """
    Convert lat-lon coordinates to cartesian coordinates in a unit box
    """

    xyz_coords = np.empty((coords.shape[0], 3), dtype="float32")

    xyz_coords[:, 0] = np.cos(np.pi * coords[:, 0] / 180.0) * np.cos(np.pi * coords[:, 1] / 180.0)
    xyz_coords[:, 1] = np.cos(np.pi * coords[:, 0] / 180.0) * np.sin(np.pi * coords[:, 1] / 180.0)
    xyz_coords[:, 2] = np.sin(np.pi * coords[:, 0] / 180.0)

    return xyz_coords


def normalise(x: np.typing.NDArray) -> np.typing.NDArray:
    """
    Normalise an array by dividing by the sum of its elements.
    """
    return x[:] / np.sum(x[:])


class VerifInterpolator:
    """
    Interpolator class that's either a wrapper for scipys LinearNDInterpolator
    or uses the handmade approximate 2D linear interpolator
    """


class Verif2DInterpolator(VerifInterpolator):
    """
    Class that does approximate 2D interpolation
    """

    def __init__(self, grid_points: np.typing.NDArray, obs_points: np.typing.NDArray):
        """
        Initialise the class and store gridpoints
        """

        grid_xyz = convert_coordinates(grid_points)
        obs_xyz = convert_coordinates(obs_points)

        self.indices = np.empty((obs_points.shape[0], 5), dtype="float32")
        tree = KDTree(grid_xyz)
        _, self.indices = tree.query(obs_xyz, k=5)

        self.weights = np.empty((obs_points.shape[0], 3), dtype="float32")
        self.compute_weights(grid_xyz, obs_xyz)

    def compute_weights(self, grid_xyz: np.typing.NDArray, obs_xyz: np.typing.NDArray):
        """
        Compute the weights of the three nearest grid points
        by computing the barycentric coordinates,
        assuming that the observations are close enough to the plane through the grid points.
        """

        eps = 0.01

        for i, (obs, indix) in enumerate(zip(obs_xyz, self.indices, strict=True)):
            ab = grid_xyz[indix[1]] - grid_xyz[indix[0]]
            ac = grid_xyz[indix[2]] - grid_xyz[indix[0]]
            bc = grid_xyz[indix[2]] - grid_xyz[indix[1]]
            ap = obs - grid_xyz[indix[0]]
            bp = obs - grid_xyz[indix[1]]

            area_tot = np.linalg.norm(np.cross(ab, ac))
            self.weights[i, 0] = np.linalg.norm(np.cross(bc, bp))
            self.weights[i, 1] = np.linalg.norm(np.cross(ac, ap))
            self.weights[i, 2] = np.linalg.norm(np.cross(ab, ap))

            if 1 - area_tot / np.sum(self.weights[i, :]) < eps:
                continue

            indix[2] = indix[3]

            ac = grid_xyz[indix[2]] - grid_xyz[indix[0]]
            bc = grid_xyz[indix[2]] - grid_xyz[indix[1]]

            area_tot = np.linalg.norm(np.cross(ab, ac))
            self.weights[i, 0] = np.linalg.norm(np.cross(bc, bp))
            self.weights[i, 1] = np.linalg.norm(np.cross(ac, ap))

            if 1 - area_tot / np.sum(self.weights[i, :]) < eps:
                continue

            indix[2] = indix[4]

            ac = grid_xyz[indix[2]] - grid_xyz[indix[0]]
            bc = grid_xyz[indix[2]] - grid_xyz[indix[1]]

            self.weights[i, 0] = np.linalg.norm(np.cross(bc, bp))
            self.weights[i, 1] = np.linalg.norm(np.cross(ac, ap))

        self.weights = self.weights / self.weights.sum(axis=1)[:, np.newaxis]

    def interpolate(
        self, values: np.typing.NDArray, intmap: np.typing.NDArray = None
    ) -> np.typing.NDArray:
        """
        Interpolate values to points
        """

        wvalues = np.empty((self.weights.shape[0]), dtype="float32")

        if intmap is None:
            wvalues[:] = (
                self.weights[:, 0] * values[self.indices[:, 0]]
                + self.weights[:, 1] * values[self.indices[:, 1]]
                + self.weights[:, 2] * values[self.indices[:, 2]]
            )
        else:
            wvalues[:] = (
                self.weights[:, 0] * values[intmap[self.indices[:, 0]]]
                + self.weights[:, 1] * values[intmap[self.indices[:, 1]]]
                + self.weights[:, 2] * values[intmap[self.indices[:, 2]]]
            )

        return wvalues


class VerifLatLonInterpolator(VerifInterpolator):
    """
    Class that does approximate 2D interpolation
    """

    def __init__(self, grid_points, obs_points):
        """
        Initialise the class and store gridpoints
        """

        self.obs_points = obs_points
        self.triangulation = Delaunay(grid_points)

    def interpolate(
        self, values: np.typing.NDArray, intmap: np.typing.NDArray = None
    ) -> np.typing.NDArray:
        """
        Interpolate values to points
        """

        newvalues = np.empty_like(values)

        if intmap is None:
            newvalues = values
        else:
            for i in range(len(values)):
                newvalues[i] = values[intmap[i]]

        interpolator = LinearNDInterpolator(self.triangulation, newvalues)

        return interpolator(self.obs_points).astype(np.float32)


class VerifNearestInterpolator(VerifInterpolator):
    """
    Class that does approximate 2D interpolation
    """

    def __init__(self, grid_points: np.typing.NDArray, obs_points: np.typing.NDArray):
        """
        Initialise the class and store gridpoints
        """

        grid_xyz = convert_coordinates(grid_points)
        obs_xyz = convert_coordinates(obs_points)

        tree = KDTree(grid_xyz)
        _, self.indices = tree.query(obs_xyz, k=1)

    def interpolate(
        self, values: np.typing.NDArray, intmap: np.typing.NDArray = None
    ) -> np.typing.NDArray:
        """
        Interpolate values to points
        """

        wvalues = np.empty((self.indices.shape[0]), dtype="float32")

        if intmap is None:
            wvalues[:] = values[self.indices[:]]
        else:
            wvalues[:] = values[intmap[self.indices[:]]]

        return wvalues


class InterpolatorFactory:
    def __init__(self, method: str):
        valid_methods = ("2d", "lat_lon", "nearest")

        if method not in valid_methods:
            raise Exception(f"{method} is not a valid method.")

        self.method = method

    def get_interpolator(
        self, zarr_coords: np.typing.NDArray, obs_coords: np.typing.NDArray
    ) -> VerifInterpolator:
        if self.method == "2d":
            _logger.info("2D interpolation")
            return Verif2DInterpolator(zarr_coords, obs_coords)

        elif self.method == "lat_lon":
            _logger.info("lat-lon interpolation")
            return VerifLatLonInterpolator(zarr_coords, obs_coords)

        elif self.method == "nearest":
            _logger.info("nearest neighbour interpolation")
            return VerifNearestInterpolator(zarr_coords, obs_coords)
