import sys
from pathlib import Path
from omegaconf import OmegaConf
from weathergen.common.config import load_streams

try:
    conf_path = "config/config_era5_georing_avhrr_forecast_random_inputs.yml"
    cfg = OmegaConf.load(conf_path)
    streams_dir = Path(cfg.streams_directory)
    print(f"Loading streams from: {streams_dir}")
    streams = load_streams(streams_dir)
    print("Available streams:")
    for name in streams.keys():
        print(f" - {name}")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
