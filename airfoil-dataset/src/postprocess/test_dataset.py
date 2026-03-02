import argparse
import math
import os
import re

import matplotlib.pyplot as plt
import meshio
import numpy as np
from tqdm import tqdm


def check_velocity_threshold(xdmf_file: str, threshold: float, amount: float = 0.25):
    """
    Check if the maximum absolute velocity in the XDMF file exceeds a certain threshold.
    Analyze the last `amount` % of timesteps.

    Args:
        xdmf_file (str): Path to the XDMF file.
        threshold (float): Velocity threshold.
        amount (float): Percentage of timesteps to analyze from the end of the file. (default is 0.25)
    Returns:
        tuple: A tuple containing a boolean indicating if the check passed,
               and the maximum velocity found.
    """
    reader = meshio.xdmf.TimeSeriesReader(xdmf_file)
    max_velocity = 0
    point, cell = reader.read_points_cells()
    for k in range(
        int((1 - amount) * reader.num_steps), reader.num_steps
    ):  # Skip the first 75% timesteps
        _, point_data, _, _ = reader.read_data(k)
        velocity = point_data["Vitesse"]
        max_velocity = max(max_velocity, np.max(np.abs(velocity)))
        if max_velocity > threshold:
            return False, max_velocity
    return True, max_velocity


def get_number_of_timesteps(xdmf_file):
    """Get the number of timesteps in the XDMF file, not considering
    initial timestep as timestep of trajectory (n-1)."""
    reader = meshio.xdmf.TimeSeriesReader(xdmf_file)
    return reader.num_steps - 1


def check_dataset_quality(
    directory: str,
    threshold: float,
    expected_timesteps: int = 600,
    amount: float = 0.25,
):
    """Check dataset quality based on the following criteria:
        1. Check if the maximum velocity is below a certain threshold
        2. Check if the number of timesteps is as expected
        3. Calculate the average number of mesh elements

    Args:
        directory (str): Directory containing .xdmf files.
        threshold (float): Velocity threshold.
        expected_timesteps (int): Expected number of timesteps.
        amount (float): Percentage of timesteps to analyze from the end of the file.

    Returns:
        list: A list of failed simulations with corresponding max velocity,
              and the average number of mesh elements."""
    failed_simulations = []

    files = [file for file in os.listdir(directory) if file.endswith(".xdmf")]
    for file in tqdm(files, desc="Checking dataset quality"):
        xdmf_file = os.path.join(directory, file)

        velocity_check, max_velocity = check_velocity_threshold(
            xdmf_file=xdmf_file, threshold=threshold, amount=amount
        )
        if not velocity_check:
            failed_simulations.append((file, max_velocity))
            continue

        if get_number_of_timesteps(xdmf_file) != expected_timesteps:
            failed_simulations.append((file, max_velocity))
            continue

    return failed_simulations


def compute_average_velocity_per_timestep(xdmf_file: str) -> list:
    """
    Read an XDMF TimeSeries and compute the average velocity magnitude
    over all mesh points for each timestep. Returns a list of averages
    in timestep order (0..T-1).
    """
    reader = meshio.xdmf.TimeSeriesReader(xdmf_file)
    averages = []
    reader.read_points_cells()
    for k in range(reader.num_steps):
        _, point_data, _, _ = reader.read_data(k)
        vel = point_data["Vitesse"]
        # compute magnitude per node then mean over mesh
        mags = np.linalg.norm(vel, axis=1)
        averages.append(float(np.mean(mags)))
    return averages


def _numeric_key(filename: str):
    """Sort key that sorts numeric stems as numbers when possible, otherwise lexicographically."""
    stem = os.path.splitext(filename)[0]
    m = re.match(r"^\s*0*(\d+)\s*$", stem)
    if m:
        return (0, int(m.group(1)))
    # try any leading number
    m2 = re.match(r"^\s*(\d+)", stem)
    if m2:
        return (0, int(m2.group(1)))
    return (1, stem)


def plot_average_velocity_evolution(directory: str, out_dir: str | None = None):
    """
    For each XDMF in `directory` (sorted 0,1,2,...), compute average velocity per timestep
    and save figures containing 5x5 subplots. One figure per 25 configurations.
    Subplot title is the configuration filename (without path).
    """

    files = [f for f in os.listdir(directory) if f.endswith(".xdmf")]
    if not files:
        return
    files_sorted = sorted(files, key=_numeric_key)
    if out_dir is None:
        out_dir = os.path.join(directory, "plots")
    os.makedirs(out_dir, exist_ok=True)

    per_config_averages = []
    for f in tqdm(files_sorted, desc="Computing average velocities"):
        path = os.path.join(directory, f)
        try:
            avgs = compute_average_velocity_per_timestep(path)
        except Exception:
            avgs = []
        per_config_averages.append((f, avgs))

    configs_per_page = 25
    total = len(per_config_averages)
    pages = math.ceil(total / configs_per_page)

    for page in range(pages):
        fig, axes = plt.subplots(5, 5, figsize=(16, 12), constrained_layout=True)
        axes = axes.flatten()
        start = page * configs_per_page
        end = min(total, start + configs_per_page)
        for idx_ax in range(configs_per_page):
            ax = axes[idx_ax]
            cfg_index = start + idx_ax
            if cfg_index < end:
                name, avgs = per_config_averages[cfg_index]
                # config label is the filename stem split by "_" and take last segment
                cfg_label = os.path.splitext(name)[0].split("_")[-1]
                if avgs:
                    ax.plot(
                        range(10, len(avgs)), avgs[10:], color="darkblue", linewidth=1
                    )
                    ax.set_xlabel("timestep")
                    ax.set_ylabel("avg |v|")
                else:
                    ax.text(0.5, 0.5, "read error or empty", ha="center", va="center")
                ax.set_title(cfg_label)
            else:
                ax.axis("off")
        out_path = os.path.join(out_dir, f"avg_velocity_evolution_page_{page:03d}.png")
        fig.suptitle(
            f"Average velocity evolution (configs {start}..{end-1})", fontsize=16
        )
        fig.savefig(out_path, dpi=300)
        plt.close(fig)


def _parser():
    parser = argparse.ArgumentParser(description="Check dataset quality.")
    parser.add_argument(
        "directory", type=str, help="Directory containing .xdmf/.h5 files"
    )
    parser.add_argument(
        "-v", "--threshold", type=float, default=10, help="Velocity threshold"
    )
    parser.add_argument(
        "-n",
        "--expected_timesteps",
        type=int,
        default=600,
        help="Expected number of timesteps",
    )
    parser.add_argument(
        "-a",
        "--amount",
        type=float,
        default=0.25,
        help="Percentage of timesteps to analyze from the end of the file (default is 0.25)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help=(
            "Compute and save average-velocity-vs-time plots (5x5 subplots per figure) "
            "into a 'plots' folder inside the dataset directory."
        ),
    )

    return parser.parse_args()


def main_test(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = _parser()
    failed_simulations = check_dataset_quality(
        directory=args.directory,
        threshold=args.threshold,
        expected_timesteps=args.expected_timesteps,
        amount=args.amount,
    )
    total_fails = len(failed_simulations)
    print(f"{total_fails} failed simulations:")
    for sim, velocity in failed_simulations:
        xdmf_file = os.path.join(args.directory, sim)
        num_timesteps = get_number_of_timesteps(xdmf_file)
        print(
            f"  - {sim}: max velocity = {velocity:.2f}, total timesteps = {num_timesteps}"
        )
    if args.plot:
        plot_average_velocity_evolution(args.directory)


if __name__ == "__main__":
    main_test(args=None)
