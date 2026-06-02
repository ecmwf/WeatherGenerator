import sys
import os
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from weathergen.common.config import load_streams
from weathergen.datasets.masking import Masker

try:
    conf_path = "config/config_era5_georing_avhrr_forecast_random_inputs.yml"
    cfg = OmegaConf.load(conf_path)
    streams_dir = Path(cfg.streams_directory)
    streams_list = load_streams(streams_dir)
    
    masker = Masker(healpix_level=cfg.healpix_level, stage='train', streams=streams_list, mode_cfg=cfg.training_config)
    masker.rng = np.random.default_rng(0)
    
    streams_dict = {s.name: s for s in streams_list}
    target_streams = ["ERA5_in", "ERA5", "METEOSAT_SEVIRI_IR"]
    num_cells = 12 * (4**cfg.healpix_level)
    
    for stream_name in target_streams:
        if stream_name not in streams_dict:
            print(f"Stream {stream_name} not found.")
            continue
        
        stream = streams_dict[stream_name]
        source_masks, target_masks, metadata = masker.build_samples_for_stream('masking', num_cells, stream)
        
        print(f"Stream: {stream_name}")
        # source_masks.masks should be the attribute
        print(f"  Source masks kept cells: {[m.sum() for m in source_masks.masks]}")
        print(f"  Target masks kept cells: {[m.sum() for m in target_masks.masks]}")
        print(f"  Metadata: {metadata}")
        
        effective_cfg = masker._effective_masking_cfgs.get(stream_name)
        print(f"  Effective masking cfg: {effective_cfg}")
        print("-" * 20)

except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
