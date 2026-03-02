import os
import argparse
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import load_dataframe, save_dataframe
from typing import List, Dict
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_configs(config_pool_path: str) -> pd.DataFrame:
    """
    Load the dataset configuration dataframe for the given config pool path.
    """
    df = load_dataframe(config_pool_path)
    return df


def get_list_configs_from_dir(split_dir: str) -> List[str]:
    """
    Get a list of config IDs from the split directory.
    """
    config_ids = []
    for root, _, files in os.walk(split_dir):
        for file in files:
            if "_" in file:
                config_id = os.path.splitext(file)[0].split("_")[-1]
                config_ids.append(config_id)
    return config_ids


def select_configs_by_list(df: pd.DataFrame, config_ids: List[str]) -> pd.DataFrame:
    """
    Select configurations from the dataframe based on a list of config IDs.

    Args:
        df (pd.DataFrame): The dataframe containing configuration data.
        config_ids (list): A list of configuration IDs to select.

    Returns:
        pd.DataFrame: A dataframe containing only the selected configurations.
    """
    config_ids = [str(cid) for cid in config_ids]  # sanitize input
    selected_configs = df[df["Config"].isin(config_ids)]
    return selected_configs


def save_split_dfs(split_dfs, out_dir):
    """
    Save the split dataframes to the specified output directory.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for split, df in split_dfs.items():
        out_path = os.path.join(out_dir, f"{split}_configs")
        save_dataframe(df=df, out_path=out_path, stringify=True)


def get_split_dfs(
    config_pool_path: str, split_dir: str, save: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Get the configuration dataframes for each split (train, test, predict) from the split directory.
    """
    df = load_configs(config_pool_path)
    split_dfs = {}

    for split in ["train", "test", "predict"]:
        split_path = os.path.join(split_dir, split)
        if not os.path.exists(split_path):
            continue
        config_ids = get_list_configs_from_dir(split_path)
        split_dfs[split] = select_configs_by_list(df, config_ids)
    if save:
        save_split_dfs(split_dfs, split_dir)
    return split_dfs


def plot_pca(split_dfs: Dict[str, pd.DataFrame], param_cols: List[str], out_path=None):
    """
    Plot PCA of the configuration parameters for each split.

    Args:
        split_dfs (Dict[str, pd.DataFrame]): Dictionary of split dataframes.
        param_cols (List[str]): List of parameter columns to use for PCA.
        out_path (str, optional): Path to save the PCA plot. If None, will show the plot.
    """
    # Concatenate all splits for PCA
    all_dfs = []
    labels = []
    for split, df in split_dfs.items():
        # If columns contain arrays, flatten them into separate columns
        split_df = df[param_cols].dropna()
        # Expand array columns into multiple columns if needed
        for col in param_cols:
            if (
                split_df[col]
                .apply(lambda x: isinstance(x, (list, tuple, np.ndarray)))
                .any()
            ):
                # Convert each array in the column to a DataFrame and join
                expanded = pd.DataFrame(split_df[col].tolist(), index=split_df.index)
                expanded.columns = [f"{col}_{i}" for i in range(expanded.shape[1])]
                split_df = split_df.drop(columns=[col]).join(expanded)
        all_dfs.append(split_df)
        labels.extend([split] * len(split_df))
    X = pd.concat(all_dfs, ignore_index=True)

    # Fit PCA on all param columns (can be >2)
    pca = PCA(n_components=min(len(X.columns), 3))
    X_pca = pca.fit_transform(X)

    # Plot first two or three principal components for visualization
    if X_pca.shape[1] > 2:

        # 3D plot of the first three principal components
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        for split in split_dfs:
            idx = [i for i, l in enumerate(labels) if l == split]
            ax.scatter(
                X_pca[idx, 0], X_pca[idx, 1], X_pca[idx, 2], label=split, alpha=0.7
            )
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_zlabel("PCA 3")
        ax.set_title(f"3D PCA of Config Parameters ({', '.join(param_cols)}) by Split")
        ax.legend()

        # 2D subplots for each combination of PCA components
        n_components = X_pca.shape[1]
        pairs = list(itertools.combinations(range(n_components), 2))
        fig2, axs = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5))
        if len(pairs) == 1:
            axs = [axs]
        for ax, (i, j) in zip(axs, pairs):
            for split in split_dfs:
                idx = [k for k, l in enumerate(labels) if l == split]
                ax.scatter(X_pca[idx, i], X_pca[idx, j], label=split, alpha=0.7)
            ax.set_xlabel(f"PCA {i+1}")
            ax.set_ylabel(f"PCA {j+1}")
            ax.set_title(f"PCA {i+1} vs PCA {j+1}")
        axs[0].legend()
        if out_path:
            if out_path.endswith(".png"):
                out_path = out_path[:-4]
            fig2.savefig(out_path + "_2d.png")
        plt.close(fig2)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        for split in split_dfs:
            idx = [i for i, l in enumerate(labels) if l == split]
            ax.scatter(X_pca[idx, 0], X_pca[idx, 1], label=split, alpha=0.7)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title(f"PCA of Config Parameters ({', '.join(param_cols)}) by Split")
        ax.legend()
    if out_path:
        if out_path.endswith(".png"):
            out_path = out_path[:-4]
        fig.savefig(out_path + ".png")
    plt.close(fig)


def _parser():
    parser = argparse.ArgumentParser(
        description=(
            "Given a path to the split directories (should contain train, test, predict subdirectories), "
            "generates the config dataframes for each split set. Also possibility to plot these "
            "configs with regards to one another to visualize similarity using PCA."
        )
    )
    parser.add_argument(
        "-p",
        "--config_pool_path",
        type=str,
        required=True,
        help="Path to the config pool file (must be pickle file, ie .pkl).",
    )
    parser.add_argument(
        "-d",
        "--split_dir",
        type=str,
        required=True,
        help="Path to the directory containing the split (train, test, predict) subdirectories.",
    )
    parser.add_argument(
        "--plot_pca",
        action="store_true",
        help="If set, plots PCA of the configuration parameters.",
    )
    parser.add_argument(
        "--param_cols",
        type=str,
        nargs="+",
        default=["x_objects", "y_objects"],
        help="List of parameter columns to use for PCA (space-separated).",
    )
    parser.add_argument(
        "--plot_save_path",
        type=str,
        default=None,
        help="Path to save the PCA plot (if --plot_pca is set).",
    )
    return parser.parse_args()


def main_postprocess_splits(args=None):
    if args is None:
        args = _parser()

    split_dfs = get_split_dfs(args.config_pool_path, args.split_dir)
    print("Split sets:")
    for split, df in split_dfs.items():
        print(f" - {split}: {df.shape[0]} configs")

    if args.plot_pca:
        plot_pca(
            split_dfs=split_dfs,
            param_cols=args.param_cols,
            out_path=args.plot_save_path,
        )


if __name__ == "__main__":
    main_postprocess_splits()
