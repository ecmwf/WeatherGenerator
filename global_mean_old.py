import zarr
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    model_id = "wdnqgwzl"
    zarr_file = f"/e/scratch/weatherai/shared_work/results/{model_id}/validation_chkpt00000_rank0000.zarr"
    
    var = "2t"

    store = zarr.storage.LocalStore(zarr_file)

    ds = zarr.open(store=store)

    target_val = []
    pred_val = []

    for i in range(1,1000):
        target_mean = np.mean(ds[f"0/ERA5/{i}/target/data"][:,3])
        pred_mean = np.mean(ds[f"0/ERA5/{i}/prediction/data"][:,3])
        target_val.append(target_mean)
        pred_val.append(pred_mean)

    target_numpy = np.array(target_val)
    pred_numpy = np.array(pred_val)

    lead_time = np.arange(1,1000)
    plt.figure()
    plt.plot(lead_time*6, target_numpy, color="black", label="target")
    plt.plot(lead_time*6, pred_numpy, color="red", linestyle="-", label="pred")

    plt.xlabel("lead time")
    plt.ylabel("2t")
    plt.legend()
    plt.grid(True)      # optional
    plt.tight_layout()  # optional

    plt.savefig(f"pred_vs_target_{model_id}_{var}.png")

    

    
