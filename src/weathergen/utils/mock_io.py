import dask.array as da
import numpy as np
import xarray as xr


def _random_datetimes_next_24h(n, seed=None):
    """
    Generate an array of `n` random numpy.datetime64[ns] values uniformly distributed
    between now and 24 hours from now.
    """
    if seed is not None:
        np.random.seed(seed)

    now = np.datetime64("now", "ns")
    end = now + np.timedelta64(24, "h")

    # Represent times as integers (ns since epoch)
    now_int = now.astype("int64")
    end_int = end.astype("int64")

    # Sample n random integers in [now_int, end_int)
    rand_ints = np.random.randint(now_int, end_int, size=n)

    # Convert back to datetime64
    return list(rand_ints.astype("datetime64[ns]"))


class MockDataChunk:
    def __init__(self, sample, stream, forecast_step, datapoints=10, channels=1, ens_size=1):
        # dims: sample, stream, forecast_time, valid_time, ipoint, variable, ens
        shape = (sample, 1, 1, datapoints, channels, ens_size)
        sampler = np.arange(np.prod(shape)).reshape(shape)
        darr = da.from_array(sampler, chunks=tuple(max(1, s // 2) for s in shape))
        self._xr = xr.DataArray(
            darr,  # 1
            dims=["sample", "stream", "forecast_step", "ipoint", "channel", "ens"],
            coords={
                "sample": [sample],
                "stream": [stream],
                "forecast_step": [forecast_step],
                "valid_time": ("ipoint", _random_datetimes_next_24h(datapoints)),
                "ipoint": np.arange(datapoints),
                "channel": [f"var{i}" for i in range(channels)],
                "ens": np.arange(ens_size),
                # lat/lon for spatial points; here simply indexed
                "lat": ("ipoint", np.linspace(-90, 90, datapoints)),
                "lon": ("ipoint", np.linspace(-180, 180, datapoints)),
            },
        )

    def as_xarray(self):
        return self._xr


class MockIO:
    def __init__(self, config):
        self.config = config
        self.streams = ["ERA5", "GFS", "ECMWF"]
        self.samples = list(range(3))
        self.forecast_steps = [0, 6, 12]

    def get_data(self, sample: int, stream: str, forecast_step: int):
        assert sample in self.samples, f"Unknown sample: {sample}"
        assert stream in self.streams, f"Unknown stream: {stream}"
        assert forecast_step in self.forecast_steps, f"Unknown step: {forecast_step}"

        class DataBundle:
            def __init__(self, sbj):
                self.source = sbj
                self.target = sbj
                self.prediction = sbj

        chunk = MockDataChunk(sample, stream, forecast_step, datapoints=10, channels=1, ens_size=1)
        return DataBundle(chunk)


# Example usage in tests
if __name__ == "__main__":
    io = MockIO(config={"dummy": True})
    assert io.streams == ["ERA5", "GFS", "ECMWF"]
    db = io.get_data(1, "ERA5", 6)
    da = db.prediction.as_xarray()
    print(da)
