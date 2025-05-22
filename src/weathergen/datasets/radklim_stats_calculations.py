import json
import xarray as xr
import dask
from dask.diagnostics import ProgressBar
from pathlib import Path
import argparse

def compute_radklim_global_stats(
    data_root,
    out_json_path,
    variables=["RR"],
    fname_prefix="RW_2017.002_",
    chunking={"time": 24}, 
    use_progressbar=True, 
    scheduler="single-threaded" 
):
    """
    Computes global mean and std for specified variables from Radklim NetCDF files.
    
    Parameters
    ----------
    data_root : str or Path
        Root path to Radklim NetCDF directory structure (organized by year).
    out_json_path : str or Path
        Path to save output JSON with "mean" and "std".
    variables : list of str
        Variables to compute stats for.
    fname_prefix : str
        Filename prefix to match NetCDF files.
    chunking : dict
        Chunk sizes passed to xarray.open_mfdataset.
    use_progressbar : bool
        If True, show Dask progress bar during compute.
    scheduler : str
        Dask scheduler to use ("threads", "processes", or "distributed").
    """
    data_root = Path(data_root)
    all_nc_files = sorted(data_root.rglob(f"{fname_prefix}*.nc"))

    if not all_nc_files:
        raise FileNotFoundError(f"No files found under {data_root} with prefix {fname_prefix}")

    # Open all NetCDFs as a single virtual dataset with chunking
    ds = xr.open_mfdataset(
        all_nc_files,
        combine="by_coords", # worked faster compared to nested
        engine="netcdf4", 
        chunks=chunking,
        parallel=False,
    )

    # Sanity check on variable presence
    for var in variables:
        if var not in ds:
            raise ValueError(f"Variable '{var}' not found in dataset.")

    stats = {"mean": [], "std": []}

    # # Double check for NaN values
    # compute_fn = lambda d: {
    # "mean": d.mean(dim=["time", "y", "x"], skipna=True),
    # "std": d.std(dim=["time", "y", "x"], skipna=True)
    # }

    for var in variables:
        data = ds[var]
        mean = data.mean(dim=["time", "y", "x"], skipna=True)
        std = data.std(dim=["time", "y", "x"], skipna=True)

        # Can be elemenated
        if use_progressbar and scheduler != "distributed":
            with ProgressBar():
                mean_val, std_val = dask.compute(mean, std, scheduler=scheduler)
        else:
            mean_val, std_val = dask.compute(mean, std, scheduler=scheduler)

        stats["mean"].append(float(mean_val))
        stats["std"].append(float(std_val))

    # Save to JSON
    out_json_path = Path(out_json_path)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_json_path, "w") as f:
        json.dump(stats, f, indent=2)

    ds.close()
    print(f"Saved stats to {out_json_path}")
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_json_path", type=str, required=True)
    parser.add_argument("--variables", nargs="+", required=True)

    args = parser.parse_args()

    compute_radklim_global_stats(
        data_root=Path(args.data_root),
        out_json_path=args.out_json_path,
        variables=args.variables,
    )

