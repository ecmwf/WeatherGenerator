from datetime import datetime

import numpy as np
import zarr

class UnstructuredDataset():
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
        self.len_hrs = len_hrs

        format_str = "%Y%m%d%H%M%S"
        if type(start) is int:
            start = datetime.strptime(str(start), format_str)
        start = np.datetime64(start).astype('datetime64[D]')
        
        if type(end) is int:
            end = datetime.strptime(str(end), format_str)
        end = np.datetime64(end).astype('datetime64[D]')

        self.normalize = normalize
        self.filename = filename
        self.z = zarr.open( filename, mode="r")
        self.mesh_size = self.z.data.attrs['nod2']

        self.time = self.z['dates']
        self.len = (end - start).astype('timedelta64[D]').astype(int) - self.len_hrs
        begin = self.time[0][0].astype('datetime64[D]')

        self.start_idx = (start - begin).astype('timedelta64[D]').astype(int) * self.mesh_size
        self.end_idx = ((end - begin).astype('timedelta64[D]').astype(int) + 1) * self.mesh_size - 1
        
        assert self.end_idx > self.start_idx, (
            f"Abort: Final index of {self.end_idx} is the same of larger than start index {self.start_idx}"
        )

        self.colnames = list(self.z.data.attrs['colnames'])
        # Ignore step_hrs, idk how it supposed to work 
        self.step_hrs = 1

        self.data = self.z["data"]

        self.properties = { 'obs_id' : self.z.data.attrs["obs_id"],
                        'means' : self.z.data.attrs["means"],
                        'vars' : self.z.data.attrs['vars'], 
        }

        if select:
            self.select(select)
        else:
            self.select(self.colnames)


    def select(self, cols_list: list[str]) -> None:
        """
        Allow user to specify which columns they want to access.
        Get functions only returned for these specified columns.
        """

        self.selected_colnames = cols_list
        self.selected_cols_idx = np.array(
            [self.colnames.index(item) for item in cols_list]
        )

    def __len__(self):
        return self.len
    
    def __getitem__( self, idx: int) -> tuple :
        start_row = self.start_idx + idx * self.mesh_size
        end_row = start_row + self.len_hrs * self.mesh_size
        data = self.data.oindex[start_row:end_row, :]
        data[:, [0,1]] = data[:, [1,0]]
        datetimes = np.squeeze(self.time[start_row:end_row])
        
        return (data, datetimes)
    
    def time_window(self, idx: int) -> tuple[np.datetime64, np.datetime64]:
        start_row = self.start_idx + idx * self.mesh_size
        end_row = start_row + self.len_hrs * self.mesh_size

        return (self.time[start_row, 0], self.time[end_row, 0])
    

if __name__ == "__main__":
    from weathergen.datasets.regular_dataset import RegularDataset
    import time

    u = UnstructuredDataset("/work/ab0995/a270088/Kacper/weathergenertor/coupled_sst_sss_ssh_yearly", 196001010000, 201901010000, 6)

    # print(r.__dict__)
    t = time.process_time()

    print(len(u))
    print(u[8909][0].shape)
    print(u.time_window(8909))
    elapsed_time = time.process_time() - t
    print(f"Unstructured: {elapsed_time}")
