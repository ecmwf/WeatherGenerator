import numpy as np
from pathlib import Path
from weathergen.datasets.data_reader_synop import DataReaderSynop
from weathergen.datasets.data_reader_base import TimeWindowHandler

def test_reader():
    test_file = Path("/p/project1/hclimrep/hauer1/synop/data/datasetv1/2019.nc")
    
    tw_handler = TimeWindowHandler(
        t_start=np.datetime64("2019-03-01T00:00:00"),
        t_end=np.datetime64("2019-03-01T06:00:00"),
        t_window_len_hours=6,
        t_window_step_hours=6,
    )
    
    stream_info = {
        "name": "GermanReal",
        "type": "station",
        "filenames": [str(test_file)],
        "source": ["TT_10"],
        "target": ["TT_10"],
        "geoinfos": ["height"],
        "latitude_name": "latitude",
        "longitude_name": "longitude",
        "height_name": "height",
    }
    
    reader = DataReaderSynop(tw_handler, test_file, stream_info)
    print("Reader initialized successfully.")
    
    rd = reader._get(0, reader.source_idx)
    print(f"Data shape: {rd.data.shape}")
    print(f"Coords shape: {rd.coords.shape}")
    print(f"Geoinfos shape: {rd.geoinfos.shape}")
    
    print("Test finished!")

if __name__ == "__main__":
    test_reader()
