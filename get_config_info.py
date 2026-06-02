import os
from omegaconf import OmegaConf, DictConfig

try:
    conf_path = "config/config_era5_georing_avhrr_forecast_random_inputs.yml"
    cfg = OmegaConf.load(conf_path)
    print(f"streams_directory: {cfg.get('streams_directory')}")
    
    physical_mse = cfg.get("training_config", {}).get("losses", {}).get("physical", {}).get("loss_fcts", {}).get("mse", {})
    print(f"target_source_correspondence: {physical_mse.get('target_source_correspondence')}")

    streams_dir = "config/streams/era5_georing_avhrr_forecast_random_inputs"
    for f in sorted(os.listdir(streams_dir)):
        if f.endswith(".yml") or f.endswith(".yaml"):
            scat = OmegaConf.load(os.path.join(streams_dir, f))
            
            # The structure in era5.yml is { "ERA5_in": { ... } }
            # But it could also be a list or have a "streams" key.
            if isinstance(scat, list):
                items = scat
            elif isinstance(scat, DictConfig):
                if "streams" in scat:
                    items = scat.streams
                else:
                    # It might be many keys where each key is a stream
                    items = []
                    for k, v in scat.items():
                        if isinstance(v, (dict, DictConfig)) and ("type" in v or "filenames" in v):
                            # This looks like a stream definition
                            stream_info = dict(v)
                            stream_info["name"] = k
                            items.append(stream_info)
                        else:
                            # Fallback if it is just a flat dict representing one stream
                            pass
                    
                    if not items and ("type" in scat or "filenames" in scat):
                         items = [scat]
            else:
                items = []
            
            for s in items:
                name = s.get("name")
                mo = s.get("masking_override")
                print(f"stream: {name}, masking_override: {mo}")
except Exception as e:
    import traceback
    traceback.print_exc()
    exit(1)
