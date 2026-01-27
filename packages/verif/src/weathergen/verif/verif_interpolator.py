import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay, KDTree


def convert_coordinates(coords):
    """
    Convert lat-lon coordinates to cartesian coordinates in a unit box
    """

    xyz_coords = np.ndarray((coords.shape[0], 3), dtype="float32")

    xyz_coords[:, 0] = np.cos(np.pi * coords[:, 0] / 180.0) * np.cos(np.pi * coords[:, 1] / 180.0)
    xyz_coords[:, 1] = np.cos(np.pi * coords[:, 0] / 180.0) * np.sin(np.pi * coords[:, 1] / 180.0)
    xyz_coords[:, 2] = np.sin(np.pi * coords[:, 0] / 180.0)

    return xyz_coords


def normalise(x):
    return x[:] / np.sum(x[:])


class Verif_interpolator:
    """
    Interpolator class that's either a wrapper for scipys LinearNDInterpolator
    or uses the handmade approximate 2D linear interpolator
    """


class Verif_2D_interpolator(Verif_interpolator):
    """
    Class that does approximate 2D interpolation
    """

    def __init__(self, grid_points, obs_points):
        """
        Initialise the class and store gridpoints
        """

        grid_xyz = convert_coordinates(grid_points)
        obs_xyz = convert_coordinates(obs_points)

        self.indices = np.ndarray((obs_points.shape[0], 5), dtype="float32")
        tree = KDTree(grid_xyz)
        _, self.indices = tree.query(obs_xyz, k=5)

        self.weights = np.ndarray((obs_points.shape[0], 3), dtype="float32")
        self.compute_weights(grid_xyz, obs_xyz)

    def compute_weights(self, grid_xyz, obs_xyz):
        """
        Compute the weights of the three nearest grid points
        by computing the barycentric coordinates,
        assuming that the observations are close enough to the plane through the grid points.
        """

        eps = 0.01

        for i, (obs, indix) in enumerate(zip(obs_xyz, self.indices, strict=True)):
            AB = grid_xyz[indix[1]] - grid_xyz[indix[0]]
            AC = grid_xyz[indix[2]] - grid_xyz[indix[0]]
            BC = grid_xyz[indix[2]] - grid_xyz[indix[1]]
            AP = obs - grid_xyz[indix[0]]
            BP = obs - grid_xyz[indix[1]]

            area_tot = np.linalg.norm(np.cross(AB, AC))
            self.weights[i, 0] = np.linalg.norm(np.cross(BC, BP))
            self.weights[i, 1] = np.linalg.norm(np.cross(AC, AP))
            self.weights[i, 2] = np.linalg.norm(np.cross(AB, AP))

            if 1 - area_tot / np.sum(self.weights[i, :]) < eps:
                continue

            indix[2] = indix[3]

            AC = grid_xyz[indix[2]] - grid_xyz[indix[0]]
            BC = grid_xyz[indix[2]] - grid_xyz[indix[1]]

            area_tot = np.linalg.norm(np.cross(AB, AC))
            self.weights[i, 0] = np.linalg.norm(np.cross(BC, BP))
            self.weights[i, 1] = np.linalg.norm(np.cross(AC, AP))

            if 1 - area_tot / np.sum(self.weights[i, :]) < eps:
                continue

            indix[2] = indix[4]

            AC = grid_xyz[indix[2]] - grid_xyz[indix[0]]
            BC = grid_xyz[indix[2]] - grid_xyz[indix[1]]

            self.weights[i, 0] = np.linalg.norm(np.cross(BC, BP))
            self.weights[i, 1] = np.linalg.norm(np.cross(AC, AP))

        self.weights = self.weights / self.weights.sum(axis=1)[:, np.newaxis]

    def interpolate(self, values, intmap=None):
        """
        Interpolate values to points
        """

        wvalues = np.ndarray((self.weights.shape[0]), dtype="float32")

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


class Verif_lat_lon_interpolator(Verif_interpolator):
    """
    Class that does approximate 2D interpolation
    """

    def __init__(self, grid_points, obs_points):
        """
        Initialise the class and store gridpoints
        """

        self.obs_points = obs_points
        self.triangulation = Delaunay(grid_points)

    def interpolate(self, values, intmap=None):
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


class Verif_nearest_interpolator(Verif_interpolator):
    """
    Class that does approximate 2D interpolation
    """

    def __init__(self, grid_points, obs_points):
        """
        Initialise the class and store gridpoints
        """

        grid_xyz = convert_coordinates(grid_points)
        obs_xyz = convert_coordinates(obs_points)

        tree = KDTree(grid_xyz)
        _, self.indices = tree.query(obs_xyz, k=1)

    def interpolate(self, values, intmap=None):
        """
        Interpolate values to points
        """

        wvalues = np.ndarray((self.indices.shape), dtype="float32")

        if intmap is None:
            wvalues[:] = values[self.indices[:]]
        else:
            wvalues[:] = values[intmap[self.indices[:]]]

        return wvalues


class Interpolator_factory:
    def __init__(self, method: str):
        valid_methods = ("2d", "lat_lon", "nearest")

        if method not in valid_methods:
            raise Exception(f"{method} is not a valid method.")

        self.method = method

    def get_interpolator(
        self, zarr_coords: np.ndarray, obs_coords: np.ndarray
    ) -> Verif_interpolator:
        if self.method == "2d":
            print("2D interpolation")
            return Verif_2D_interpolator(zarr_coords, obs_coords)

        elif self.method == "lat_lon":
            print("lat-lon interpolation")
            return Verif_lat_lon_interpolator(zarr_coords, obs_coords)

        elif self.method == "nearest":
            print("nearest neighbour interpolation")
            return Verif_nearest_interpolator(zarr_coords, obs_coords)
