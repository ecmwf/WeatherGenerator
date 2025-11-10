import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import tqdm


def load_checkpoint(run_id: str, epoch: int) -> dict[str, torch.Tensor]:
    chkpt = torch.load(f"models/{run_id}/{run_id}_epoch{str(epoch).zfill(5)}.chkpt")
    fe_keys = [key for key in list(chkpt.keys()) if "fe" in key]
    fe_chkpt = {fe_key: chkpt[fe_key] for fe_key in fe_keys}
    return fe_keys, fe_chkpt


def get_layer_and_name(key: str) -> [int, str]:
    key_split = key.split(".")
    layer = int(key_split[1])
    name = ".".join(key_split[2:])
    return layer, name


def prepare_weights_and_eigenvalues(
    w_dict: dict[str, torch.Tensor],
) -> [dict[str, list], dict[str, list]]:
    # Compute eigenvectors of each layer. Set to [0, 0] if no matrix.
    e_dict = {
        key: (w_dict[key].svd().S.cpu().numpy()) if len(w_dict[key].shape) > 1 else [0, 0]
        for key in w_dict
    }
    # Flatten all weights
    w_dict = {key: w_dict[key].flatten().cpu().numpy() for key in w_dict}
    return w_dict, e_dict


def plot_results(
    w_dict: dict[str, torch.Tensor], epoch: int, layers: int, run_id: str, plot_dir: str
):
    w_dict, e_dict = prepare_weights_and_eigenvalues(w_dict=w_dict)
    fig, axs = plt.subplots(2, 1, figsize=(len(w_dict.keys()), 5), sharex=True)
    axs[0].boxplot(w_dict.values(), tick_labels=w_dict.keys())
    axs[1].violinplot(e_dict.values())
    axs[0].grid()
    axs[1].grid()
    axs[0].set_title("Weight distribution")
    axs[1].set_title("Singular value distribution")
    plt.xticks(rotation=45, ha="right")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = plot_dir / Path(f"w-dist_{run_id}_epoch{str(epoch).zfill(3)}.png")
    fig.savefig(plot_path, bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    run_id = "vso7p6dt"
    epochs = [2, 4, 8, 16, 32, 63]
    plot_dir = Path("plots", "w_dist", run_id)

    for epoch in tqdm.tqdm(epochs, desc="Processing epoch"):
        fe_keys, fe_w_dict = load_checkpoint(run_id=run_id, epoch=epoch)

        #
        # Option 1: All layers in one plot
        plot_results(w_dict=fe_w_dict, epoch=epoch, layers=15, run_id=run_id, plot_dir=plot_dir)

        #
        # Option 2: One plot per layer
        # layer = -1  # init
        # for fe_key in fe_keys:
        #     l, name = get_layer_and_name(key=fe_key)

        #     if layer != l:
        #         if layer != -1:
        #             plot_results(w_dict=w_per_layer, epoch=epoch, layers=15, run_id=run_id, plot_dir=plot_dir)
        #         # Reset layer dict for new layer
        #         layer = l
        #         w_per_layer = dict()

        #     w_per_layer[name] = fe_w_dict[fe_key]

        #     print(fe_key)

        # plot_results(w_dict=w_per_layer, epoch=epoch, layer=l)
