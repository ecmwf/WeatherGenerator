from datetime import datetime

import numpy as np
import zarr


class AtmorepDataset:
    def __init__(
        self,
        filename: str,
        start: datetime | int,
        end: datetime | int,
        len_hrs: int,
        step_hrs: int | None = None,
        normalize: bool = True,
        select: list[str] | None = None,
    ):
        format_str = "%Y%m%d%H%M%S"
        if type(start) is int:
            start = datetime.strptime(str(start), format_str)

        if type(end) is int:
            end = datetime.strptime(str(end), format_str)

        self.normalize = normalize
        self.filename = filename
        self.z = zarr.open(filename, mode="r")

        self.lats, self.lons = np.meshgrid(np.array(self.z["lats"]), np.array(self.z["lons"]))
        self.lats = self.lats.flatten()
        self.lons = self.lons.flatten()
        # Reshape lats and lons to be in shape (1, len_hrs, size_lat * size_lon), ready to added to data
        self.lats = np.expand_dims(np.stack((self.lats,) * len_hrs, axis=1).T, 0)
        self.lons = np.expand_dims(np.stack((self.lons,) * len_hrs, axis=1).T, 0)

        self.time = np.array(self.z["time"], dtype=np.datetime64)
        self.start_idx = np.searchsorted(self.time, start)
        self.end_idx = np.searchsorted(self.time, end)

        assert self.end_idx > self.start_idx, (
            f"Abort: Final index of {self.end_idx} is the same of larger than start index {self.start_idx}"
        )

        self.colnames = ["lat", "lon"] + list(self.z.attrs["fields"])
        self.len_hrs = len_hrs
        # Ignore step_hrs, idk how it supposed to work
        self.step_hrs = 1

        self.selected_colnames = self.colnames[2:]
        self.selected_cols_idx = np.arange(len(self.selected_colnames))
        self.data = self.z["data"]

        self.properties = {
            "obs_id": 0,
            "means": np.zeros(len(self.colnames), dtype=np.float32),
            "vars": np.ones(len(self.colnames), dtype=np.float32),
        }

        if select:
            self.select(select)

    def select(self, cols_list: list[str]) -> None:
        """
        Allow user to specify which columns they want to access.
        Get functions only returned for these specified columns.
        """
        self.selected_colnames = cols_list
        self.selected_cols_idx = np.array([self.colnames.index(item) for item in cols_list])

    def __len__(self):
        return self.end_idx - self.start_idx - self.len_hrs

    def __getitem__(self, idx: int) -> tuple:
        start_row = self.start_idx + idx
        end_row = start_row + self.len_hrs

        data = self.data.oindex[start_row:end_row, :, 0, :, :]
        datetimes = np.tile(self.time[start_row:end_row], data.shape[-1] * data.shape[-2])

        data = np.reshape(data, (data.shape[1], data.shape[0], -1))
        data = np.concatenate([self.lats, self.lons, data], 0).T
        data = np.reshape(data, (-1, data.shape[-1]))

        return (data.astype(np.float32), datetimes)

    def time_window(self, idx: int) -> tuple[np.datetime64, np.datetime64]:
        start_row = self.start_idx + idx
        end_row = start_row + self.len_hrs
        return (self.time[start_row], self.time[end_row])
