import pandas as pd
import numpy as np
import xarray as xr
from typing import List
from tqdm import tqdm
from scipy.ndimage import gaussian_laplace, maximum_filter
from scipy.cluster.hierarchy import DisjointSet
from skimage.feature import peak_local_max
from dataclasses import dataclass
from sklearn.metrics.pairwise import haversine_distances

@dataclass(order=True, frozen=True)
class cyclone:
    wind: float
    pressure: float
    lon: float
    lat: float
    ID: str | None = None
    time: np.datetime64 | None = None
    
    def dist_to(self, other: "cyclone") -> float:
        R = 6371.0
        p1 = [ np.deg2rad(deg) for deg in (self.lat, self.lon) ]
        p2 = [ np.deg2rad(deg) for deg in (other.lat, other.lon) ]
        angle = haversine_distances(
            X = np.array(p1).reshape(1,-1),
            Y = np.array(p2).reshape(1,-1)
        )
        return R*angle

    def match(self, cyclones, maxdist=3000) -> "cyclone":
        dists = [self.dist_to(other) for other in cyclones]
        if min(dists) < maxdist:
            return cyclones[np.argmin(dists)]
        else:
            return None

class cyclone_finder():
    def __init__(self, sigma: float = 2, th_LoG: float = 30, th_pressure: float = 101000, th_wind: float = 10, min_distance: float = 5):
        '''
        Try finding cyclones with simple blob detection 
        plus some heuristic filter criteria
        Attributes
        ----------
        sigma: Gauss standard deviation. The zeros of the laplace filter
               are at sqrt(2)*sigma distance from the center
        th_LoG: minimum value of the filtered field
        th_pressure: maxmimum pressure value
        th_wind: minimum wind speed
        min_distance: minimum distance between peaks in number of gridpoints
        '''
        self.sigma = sigma
        self.th_LoG = th_LoG
        self.th_pressure = th_pressure
        self.th_wind = th_wind
        self.min_distance = min_distance
        
    def filter(self, image):
        return gaussian_laplace(image, sigma=self.sigma)

    def mask(self, pressure, windmax):
        pressuremask = (pressure<self.th_pressure).values
        windmask = windmax > self.th_wind
        return pressuremask & windmask
    
    def find_cyclones(self,pressure, wind, windmaxsize=5, timestamp=None):
        # apply the LoG filter to pressure
        filtered = self.filter(pressure)
        # find candidate maxima
        candidates = peak_local_max(
            filtered, 
            threshold_abs=self.th_LoG, 
            min_distance=self.min_distance
        )
        # apply mask
        windmax = maximum_filter(wind.values, size=windmaxsize)
        mask = self.mask(pressure, windmax)[candidates[:,0],candidates[:,1]]
        cyclones = candidates[mask,:]
        res = [
            cyclone(
                lon = pressure.longitude.values[y],
                lat = pressure.latitude.values[x],
                wind = windmax[x,y],
                pressure= pressure.values[x,y],
                time = timestamp
            )
            for x,y in zip(cyclones[:,0],cyclones[:,1])
        ]
        return res

def track_cyclones(timesteps: List[List["cyclone"]], merge_distance_km: float = 300):
    '''
    Takes a list of lists of cyclones, each top level entry representing one timestep,
    returns a DisjointSet where each entry represents a track. 
    '''
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
        for dist, s_prev, s_curr in candidates:
            if s_prev not in used_prev and s_curr not in used_curr:
                tracks.merge(s_prev, s_curr)
                used_prev.add(s_prev)
                used_curr.add(s_curr)

        prev_step = step

    return tracks

def track2pandas(track: List["cyclone"]):
    return pd.DataFrame(
        [ storm.__dict__ for storm in track ]
    ).set_index("time").sort_index()    

def cyclones_in_ds(ds, finder, time):
    ds_t = ds.sel(valid_time=time)
    msl = ds_t.msl
    V = np.sqrt(ds_t.u10**2 + ds_t.v10**2)
    return finder.find_cyclones(pressure=msl, wind=V, timestamp=time)

def track_error(tr1, tr2):
    R = 6371.0
    coords = [
        np.deg2rad(x.loc[:,["lat","lon"]])
        for x in tr1.align(tr2, join="inner")
    ]
    angle = haversine_distances(
        X = coords[0].values, Y=coords[1].values
    )
    distance = pd.DataFrame({"distance":R*np.diag(angle)}, 
                            index = coords[0].index)
    all_idx = tr1.index.union(tr2.index)
    distance = distance.reindex(all_idx)

    return distance


def wrap_lon(ds):
    ds["longitude"] = (ds["longitude"] +180) % 360 - 180
    ds = ds.sortby("longitude")
    return ds
