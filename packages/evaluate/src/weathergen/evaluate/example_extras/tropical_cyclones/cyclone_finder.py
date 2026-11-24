from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from scipy.cluster.hierarchy import DisjointSet
from scipy.ndimage import gaussian_laplace, maximum_filter
from skimage.feature import peak_local_max
from sklearn.metrics.pairwise import haversine_distances
from tqdm import tqdm


@dataclass(order=True, frozen=True)
class Cyclone:
    wind: float
    pressure: float
    lon: float
    lat: float
    ID: str | None = None
    time: np.datetime64 | None = None

    def dist_to(self, other: "Cyclone") -> float:
        r_earth = 6371.0
        p1 = [np.deg2rad(deg) for deg in (self.lat, self.lon)]
        p2 = [np.deg2rad(deg) for deg in (other.lat, other.lon)]
        angle = haversine_distances(X=np.array(p1).reshape(1, -1), Y=np.array(p2).reshape(1, -1))
        return r_earth * angle

    def match(self, cyclones: list["Cyclone"], maxdist_km: float = 3000) -> "Cyclone":
        """
        Select the closest from a set of other cyclones
        """
        dists = [self.dist_to(other) for other in cyclones]
        if min(dists) < maxdist_km:
            return cyclones[np.argmin(dists)]
        else:
            return None


class CycloneFinder:
    def __init__(
        self,
        sigma: float = 2,
        th_laplace: float = 30,
        th_pressure: float = 101000,
        th_wind: float = 10,
        min_distance: float = 5,
    ):
        """
        Try finding cyclones with simple blob detection
        plus some heuristic filter criteria
        Attributes
        ----------
        sigma: Gauss standard deviation. The zeros of the laplace filter
               are at sqrt(2)*sigma distance from the center
        th_laplace: minimum value of the filtered field
        th_pressure: maxmimum pressure value
        th_wind: minimum wind speed
        min_distance: minimum distance between peaks in number of gridpoints
        """
        self.sigma = sigma
        self.th_laplace = th_laplace
        self.th_pressure = th_pressure
        self.th_wind = th_wind
        self.min_distance = min_distance

    def filter(self, image):
        return gaussian_laplace(image, sigma=self.sigma)

    def mask(self, pressure, windmax):
        pressuremask = (pressure < self.th_pressure).values
        windmask = windmax > self.th_wind
        return pressuremask & windmask

    def find_cyclones(self, pressure, wind, windmaxsize=5, timestamp=None) -> list["Cyclone"]:
        # apply the LoG filter to pressure
        filtered = self.filter(pressure)
        # find candidate maxima
        candidates = peak_local_max(
            filtered, threshold_abs=self.th_laplace, min_distance=self.min_distance
        )
        # apply mask
        windmax = maximum_filter(wind.values, size=windmaxsize)
        mask = self.mask(pressure, windmax)[candidates[:, 0], candidates[:, 1]]
        cyclones = candidates[mask, :]
        res = [
            Cyclone(
                lon=pressure.longitude.values[y],
                lat=pressure.latitude.values[x],
                wind=windmax[x, y],
                pressure=pressure.values[x, y],
                time=timestamp,
            )
            for x, y in zip(cyclones[:, 0], cyclones[:, 1], strict=False)
        ]
        return res


def track_cyclones(timesteps: list[list["Cyclone"]], merge_distance_km: float = 300) -> DisjointSet:
    """
    Takes a list of lists of cyclones, each top level entry representing one timestep,
    returns a DisjointSet where each entry represents a track.
    """
    tracks = DisjointSet()
    prev_step = []

    for step in tqdm(timesteps):
        # Add all storms from this timestep
        for storm in step:
            tracks.add(storm)

        # Build all candidate matches (prev → curr)
        candidates = []
        for s_prev in prev_step:
            for s_curr in step:
                d = s_prev.dist_to(s_curr)
                if d <= merge_distance_km:
                    candidates.append((d, s_prev, s_curr))

        # Sort by distance (closest first)
        candidates.sort(key=lambda x: x[0])

        # Keep track of which storms have already been matched
        used_prev = set()
        used_curr = set()

        # Greedy matching: closest pairs first
        for _dist, s_prev, s_curr in candidates:
            if s_prev not in used_prev and s_curr not in used_curr:
                tracks.merge(s_prev, s_curr)
                used_prev.add(s_prev)
                used_curr.add(s_curr)

        prev_step = step

    return tracks


def track2pandas(track: list["Cyclone"]) -> pd.DataFrame:
    return pd.DataFrame([storm.__dict__ for storm in track]).set_index("time").sort_index()


def cyclones_in_ds(ds: xr.Dataset, finder: "CycloneFinder", time: np.datetime64) -> list["Cyclone"]:
    """
    Find cyclones in a dataset containing at least msl, u10, v10,
    at a given timestep, using a given CycloneFinder.
    """
    ds_t = ds.sel(valid_time=time)
    msl = ds_t.msl
    v = np.sqrt(ds_t.u10**2 + ds_t.v10**2)
    return finder.find_cyclones(pressure=msl, wind=v, timestamp=time)


def track_error(track1: pd.DataFrame, track2: pd.DataFrame) -> pd.DataFrame:
    """
    Given two tracks as pd.DataFrames, compute their distance in km.
    At timesteps where one track is missing, the result is NaN.
    """
    r_earth = 6371.0
    coords = [np.deg2rad(x.loc[:, ["lat", "lon"]]) for x in track1.align(track2, join="inner")]
    angle = haversine_distances(X=coords[0].values, Y=coords[1].values)
    distance = pd.DataFrame({"distance": r_earth * np.diag(angle)}, index=coords[0].index)
    all_idx = track1.index.union(track2.index)
    distance = distance.reindex(all_idx)

    return distance


def wrap_lon(ds: xr.Dataset) -> xr.Dataset:
    "Convert longitude from 0...360 to -180...180"
    ds["longitude"] = (ds["longitude"] + 180) % 360 - 180
    ds = ds.sortby("longitude")
    return ds
