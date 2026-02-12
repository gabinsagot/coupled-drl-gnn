import os
from typing import List, Optional, Tuple

from cycler import cycler
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


# Define plot style:
scientific_style = {
    # Figure
    "figure.figsize": (8, 6),  # Adjust the figure size
    # Font
    "font.family": "serif",  # Use a serif font
    "font.serif": ["cmr10", "Computer Modern Serif", "DejaVu Serif"],
    "font.size": 14,  # Set the font size
    # Axes
    "axes.labelsize": 14,  # Label font size
    "axes.titlesize": 16,  # Title font size
    "axes.prop_cycle": cycler(
        "color",
        [
            "darkblue",
            "darkgreen",
            "darkred",
            "darkorange",
            "darkorchid",
            "darksalmon",
            "darkmagenta",
            "darkcyan",
            "darkgray",
            "saddlebrown",
        ],
    ),
    "axes.grid": True,  # Show grid lines
    "axes.formatter.use_mathtext": True,
    # Set x axis
    "xtick.direction": "in",
    "xtick.major.size": 3,
    "xtick.major.width": 0.5,
    "xtick.minor.size": 1.5,
    "xtick.minor.width": 0.5,
    "xtick.minor.visible": True,
    "xtick.top": True,
    "xtick.labelsize": 8,
    # Set y axis
    "ytick.direction": "in",
    "ytick.major.size": 3,
    "ytick.major.width": 0.5,
    "ytick.minor.size": 1.5,
    "ytick.minor.width": 0.5,
    "ytick.minor.visible": True,
    "ytick.right": True,
    "ytick.labelsize": 8,
    # Lines
    "lines.linewidth": 1,  # Line width
    "lines.markersize": 5,  # Marker size
    # Grid
    "grid.linestyle": "--",  # Grid line style
    "grid.linewidth": 0.3,
    "grid.alpha": 0.4,
    "grid.color": "gray",
    # Legend
    "legend.fontsize": 8,  # Legend font size
    "legend.frameon": True,  # Show legend frame
    "legend.framealpha": 0.6,  # Legend frame opacity
    "legend.edgecolor": "gray",  # Legend frame color
    "legend.loc": "best",  # Legend location
    # Savefig
    "savefig.dpi": 300,  # Set DPI for saved figures
    "savefig.bbox": "tight",  # Adjust bounding box when saving
    "savefig.pad_inches": 0.05,
    # Text
    "text.usetex": False,  # Use LaTeX for text rendering
    # 'text.latex.preamble' : r'\usepackage{amsmath}',  # LaTeX preamble
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",
}


def L2_norm(x: list) -> float:
    return np.sqrt(np.sum(np.square(x)))


def distance_actions(action1: list, action2: list) -> float:
    return L2_norm(np.array(action1) - np.array(action2))


def _create_pbo_avg_dat_file(reward_files_dir: str, number_runs: int):
    """Create pbo_avg.dat file by aggregating data from individual pbo_bst_X files."""
    reward_data = []
    for run in range(number_runs):
        reward_file = reward_files_dir + "/pbo_bst_" + str(run)
        try:
            with open(reward_file, "r") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        except FileNotFoundError:
            raise FileNotFoundError(f"Reward file {reward_file} not found.")
        for idx, ln in enumerate(lines):
            parts = ln.split()
            episode = float(parts[0])
            bst_reward = float(parts[2])
            if len(reward_data) <= idx:
                reward_data.append([episode, []])
            reward_data[idx][1].append(bst_reward)
    with open(reward_files_dir + "/pbo_avg.dat", "w") as f:
        for data in reward_data:
            episode = data[0]
            bst_rewards = data[1]
            if not bst_rewards:
                continue
            avg = np.mean(bst_rewards)
            std = np.std(bst_rewards)
            f.write(f"{episode} {avg:.6f} {avg + std:.6f} {avg - std:.6f}\n")


def _read_pbo_output_reward_data(
    reward_files_dir: str,
    number_runs: int = 1,
    scale_reward: float = 1,
    specific_run_id: int | None = None,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Read PBO output reward files and return avg and best arrays.

    Args:
        reward_files_dir (str): Directory containing reward files.
        number_runs (int): Number of runs to read.
        scale_reward (float): Scaling factor for rewards.
        specific_run_id (int|None): specific id of run to read, will ignore number_runs.

    Returns:
        Tuple containing:
            - Eaxis (np.ndarray): Episode axis values.
            - Ravg_avg (np.ndarray): Average reward over runs.
            - Ravg_p (np.ndarray): Average reward plus std deviation over runs.
            - Ravg_m (np.ndarray): Average reward minus std deviation over runs.
            - Rbest (np.ndarray): Best reward over runs.
            - Rbest_p (np.ndarray): Best reward plus std deviation over runs.
            - Rbest_m (np.ndarray): Best reward minus std deviation over runs.
    """
    Eaxes, Rs = [], []
    number_runs_start = specific_run_id if specific_run_id is not None else 0
    number_runs_end = (
        number_runs_start + 1 if specific_run_id is not None else number_runs
    )
    for run in range(number_runs_start, number_runs_end):
        reward_file = reward_files_dir + "/pbo_avg_" + str(run)
        try:
            with open(reward_file, "r") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        except FileNotFoundError:
            raise FileNotFoundError(f"Reward file {reward_file} not found.")
        Eaxis, R = [], []
        for ln in lines:
            parts = ln.split()
            Eaxis.append(float(parts[0]))
            R.append(-float(parts[1]) * scale_reward)
        Rs.append(R)
        Eaxes.append(Eaxis)
    Eaxis = np.array(Eaxes[0])
    Ravgs = np.array(Rs)
    Ravg_avg = np.mean(Ravgs, axis=0)
    Ravg_std = np.std(Ravgs, axis=0)
    Ravg_p = Ravg_avg + Ravg_std
    Ravg_m = Ravg_avg - Ravg_std
    Rbest, Rbest_p, Rbest_m = [], [], []

    if specific_run_id is not None:
        try:
            best_reward_file = reward_files_dir + f"/pbo_bst_{specific_run_id}"
            with open(best_reward_file, "r") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        except FileNotFoundError:
            raise FileNotFoundError(f"Best reward file {best_reward_file} not found.")
        for ln in lines:
            parts = ln.split()
            Rbest.append(-float(parts[2]) * scale_reward)
        Rbest_p = Rbest_m = Rbest.copy()
    else:
        try:
            with open(reward_files_dir + "/pbo_avg.dat") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        except FileNotFoundError:
            try:
                _create_pbo_avg_dat_file(reward_files_dir, number_runs)
                with open(reward_files_dir + "/pbo_avg.dat") as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Reward file {reward_files_dir + '/pbo_avg.dat'} was not found "
                    "and could not be created because of missing data."
                )
        for ln in lines:
            parts = ln.split()
            Rbest.append(-float(parts[1]) * scale_reward)
            Rbest_p.append(-float(parts[2]) * scale_reward)
            Rbest_m.append(-float(parts[3]) * scale_reward)
    Rbest = np.array(Rbest)
    Rbest_p = np.array(Rbest_p)
    Rbest_m = np.array(Rbest_m)
    return (Eaxis, Ravg_avg, Ravg_p, Ravg_m, Rbest, Rbest_p, Rbest_m)


def _read_processed_reward(
    reward_file: str,
    nactions: int,
    scale_reward: float,
    scale_actions: float,
    discretize_actions_set: None | np.ndarray = None,
) -> Tuple[List[float], List[float], List[float], List[List[float]], List[List[float]]]:
    """
    Read processed reward file and return data arrays.

    Args:
        reward_file (str): Path to the processed reward file.
        nactions (int): Number of actions.
        scale_reward (float): Scaling factor for rewards.
        scale_actions (float): Scaling factor for actions.
        discretize_actions_set (None|np.ndarray): Set for discretizing actions if passed

    Returns:
        Tuple containing:
            - Eaxis (List[float]): Episode axis values.
            - R (List[float]): Reward values.
            - MR (List[float]): Moving average reward values.
            - A (List[List[float]]): Action values.
            - MA (List[List[float]]): Moving average action values.
    """
    with open(reward_file, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    Eaxis, R, MR = [], [], []
    A = [[] for _ in range(nactions)]
    MA = [[] for _ in range(nactions)]
    for ln in lines:
        parts = ln.split()
        Eaxis.append(float(parts[0]))
        R.append(-float(parts[1]) * scale_reward)
        for j in range(nactions):
            if discretize_actions_set is not None:
                action_value = float(parts[2 + j]) * scale_actions
                closest_value = discretize_actions_set[
                    np.argmin(np.abs(discretize_actions_set - action_value))
                ]
                A[j].append(closest_value)
            else:
                A[j].append(float(parts[2 + j]) * scale_actions)
        MR.append(-float(parts[2 + nactions]) * scale_reward)
        for j in range(nactions):
            MA[j].append(float(parts[2 + nactions + 1 + j]) * scale_actions)
    return Eaxis, R, MR, A, MA


def _compute_markers(
    markers: List[float],
    env: int,
    R: List[float],
    A: List[List[float]],
    nactions: int,
) -> Tuple[List[str] | None, List[float] | None, List[List[float]] | None]:
    """Compute marker names and their corresponding reward and action values.
    Args:
        markers (List[float]): List of episode numbers for markers.
        env (int): Number of environments/individuals per episode/generation.
        R (List[float]): List of rewards.
        A (List[List[float]]): List of actions.
        nactions (int): Number of actions.
    Returns:
        Tuple containing:
            - List of marker names.
            - List of reward values for markers.
            - List of action values for markers.
    """
    if not markers:
        return None, None, None
    markers_name = ["A", "B", "C", "D", "E", "F", "G", "H"][: len(markers)]
    idxs = [int(m * env) for m in markers]
    markersY = [R[i] for i in idxs]
    markersA = [[A[j][i] for i in idxs] for j in range(nactions)]
    return markers_name, markersY, markersA


def _make_grid(
    nactions: int, actions_figsize: Tuple[float, float]
) -> Tuple[plt.Figure, np.ndarray, int, int]:
    """Create a grid of subplots for actions."""
    ncols = max(int(np.ceil(np.sqrt(nactions))), 3)
    nrows = int(np.ceil(nactions / ncols))
    figs, axs = plt.subplots(nrows, ncols, figsize=actions_figsize)
    axs = axs.flatten()
    return figs, axs, nrows, ncols


def _constrain_range(data: list, user_range: tuple):
    """Constrain data range based on user input or data min/max."""
    if user_range:
        return user_range[0], user_range[1]
    mn = min(data)
    mx = max(data)
    return (1.1 * mn, 0) if mx <= 0 else (mn * 0.9, mx * 1.1)


def _plot_reward_convergence(
    Eaxis: List[float],
    R: List[float],
    MR: List[float],
    markers: List[float],
    markers_name: List[str],
    markersY: List[float],
    reward_figsize: Tuple[float, float],
    episodes: int,
    Rmin: float,
    Rmax: float,
    save: bool,
    show: bool,
    reward_file: str,
    moving_avg_plot_start: int,
):
    """
    Plot reward convergence over episodes.

    Args:
        Eaxis (List[float]): Episode axis values.
        R (List[float]): Reward values.
        MR (List[float]): Moving average reward values.
        markers (List[float]): Episode numbers for plotting markers.
        markers_name (List[str]): Names of the markers.
        markersY (List[float]): Reward values for the markers.
        reward_figsize (Tuple[float, float]): Figure size for reward plot.
        episodes (int): Maximum number of episodes/generations.
        Rmin (float): Minimum reward value for y-axis.
        Rmax (float): Maximum reward value for y-axis.
        save (bool): Whether to save the figure.
        show (bool): Whether to show the figure.
        reward_file (str): Path to the processed reward file.
        moving_avg_plot_start (int): Episode number to start plotting moving average.
    """
    fig, ax = plt.subplots(figsize=reward_figsize)
    ax.plot(Eaxis, R, color="lightgray", linewidth=0.75)
    ax.plot(
        Eaxis[moving_avg_plot_start:],
        MR[moving_avg_plot_start:],
        color="black",
        linewidth=1.75,
    )
    if markers:
        ax.scatter(
            markers, markersY, s=40, facecolors="none", edgecolors="darkred", zorder=2
        )
        for i, label in enumerate(markers_name):
            ax.annotate(
                label,
                (markers[i], markersY[i]),
                textcoords="offset points",
                xytext=(2, -7.5),
                ha="left",
                va="top",
                color="darkred",
                fontsize=9,
            )
    ax.grid(False)
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Episode")
    ax.set_ylabel("$r$", rotation=0)
    ax.set_xlim((0, episodes))
    ax.set_xticks(np.arange(0, episodes + 1, step=10))
    # ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_ylim((Rmin, Rmax))
    ax.tick_params(
        labelright=False,
        labelleft=True,
        labeltop=True,
        left=True,
        right=False,
        top=True,
        labelbottom=False,
        bottom=False,
    )
    fig.tight_layout()
    if save:
        fig.savefig(f"{reward_file.split('.')[0]}_reward.png")
    if show:
        plt.show()
    plt.close(fig)


def _plot_reward_convergence_multi(
    Eaxis: np.ndarray,
    Ravg_avg: np.ndarray,
    Ravg_p: np.ndarray,
    Ravg_m: np.ndarray,
    Rbest: np.ndarray,
    Rbest_p: np.ndarray,
    Rbest_m: np.ndarray,
    reward_figsize: Tuple[float, float],
    Rmin: float | None,
    Rmax: float | None,
    reward_files_dir: str,
    save: bool,
    show: bool,
):
    """
    Plot reward convergence over episodes.

    Args:
        Eaxis (np.ndarray): Episode axis values, shape (episodes,).
        Ravg_avg (np.ndarray): Average reward over runs, shape (episodes,).
        Ravg_p (np.ndarray): Average reward plus std deviation over runs, shape (episodes,).
        Ravg_m (np.ndarray): Average reward minus std deviation over runs, shape (episodes,).
        Rbest (np.ndarray): Best reward over runs, shape (episodes,).
        Rbest_p (np.ndarray): Best reward plus std deviation over runs, shape (episodes,).
        Rbest_m (np.ndarray): Best reward minus std deviation over runs, shape (episodes,).
        reward_figsize (Tuple[float, float]): Figure size for reward plot.
        Rmin (float): Minimum reward value for y-axis.
        Rmax (float): Maximum reward value for y-axis.
        save (bool): Whether to save the figure.
        show (bool): Whether to show the figure.
    """
    fig, ax = plt.subplots(figsize=reward_figsize)
    # best
    ax.plot(Eaxis, Rbest, color="darkred", linewidth=1.25, label="best", alpha=0.85)
    ax.fill_between(
        Eaxis, Rbest_p, Rbest_m, facecolor="darkred", edgecolor=None, alpha=0.4
    )
    # avg
    ax.plot(Eaxis, Ravg_avg, color="black", linewidth=1.25, label="avg", alpha=0.85)
    ax.fill_between(
        Eaxis, Ravg_p, Ravg_m, facecolor="darkgray", edgecolor=None, alpha=0.4
    )
    ax.grid(False)
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Episode")
    ax.set_ylabel("$r$", rotation=0)
    ax.legend(loc="lower right")
    ax.set_xlim((Eaxis[0], Eaxis[-1]))
    ax.set_xticks(np.arange(0, len(Eaxis) + 1, step=10))
    # ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    if Rmin is None or Rmax is None:
        Rmin = min(min(Ravg_m), min(Rbest_m)) * 1.1
        Rmax = max([max(Ravg_p), max(Rbest_p), 0.02])
    ax.set_ylim((Rmin, Rmax))
    ax.tick_params(
        labelright=False,
        labelleft=True,
        labeltop=True,
        left=True,
        right=False,
        top=True,
        labelbottom=False,
        bottom=False,
    )
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(reward_files_dir, "avg_reward.png"))
    if show:
        plt.show()
    plt.close(fig)


def _plot_reward_convergence_multi_individual(
    Eaxis: np.ndarray,
    Ravg_avg: np.ndarray,
    Ravg_best: np.ndarray,
    reward_figsize: Tuple[float, float],
    Rmin: float | None,
    Rmax: float | None,
    reward_files_dir: str,
    save: bool,
    show: bool,
):
    """
    Plot reward convergence over episodes for each run on same plot

    Args:
        Eaxis (np.ndarray): Episode axis values, shape (episodes,num_runs).
        Ravg_avg (np.ndarray): Average reward over runs, shape (episodes,num_runs).
        Ravg_best (np.ndarray): Best reward over runs, shape (episodes,num_runs).
        reward_figsize (Tuple[float, float]): Figure size for reward plot.
        Rmin (float): Minimum reward value for y-axis.
        Rmax (float): Maximum reward value for y-axis.
        save (bool): Whether to save the figure.
        show (bool): Whether to show the figure.
    """
    fig, ax = plt.subplots(figsize=reward_figsize)
    colors = scientific_style["axes.prop_cycle"].by_key()["color"]
    for run in range(Ravg_avg.shape[1]):
        # avg
        ax.plot(
            Eaxis[:, run],
            Ravg_avg[:, run],
            linewidth=1,
            color=colors[run % len(colors)],
            label=f"run {run+1}",
            alpha=0.6,
        )
        # best
        ax.plot(
            Eaxis[:, run],
            Ravg_best[:, run],
            linewidth=1,
            linestyle="--",
            color=colors[run % len(colors)],
            label=f"best {run+1}",
            alpha=0.6,
        )
    ax.grid(False)
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Episode")
    ax.set_ylabel("$r$", rotation=0)
    ax.legend(loc="lower right", ncols=Ravg_avg.shape[1])
    ax.set_xlim((Eaxis[0, 0], Eaxis[-1, 0]))
    ax.set_xticks(np.arange(0, Eaxis.shape[0] + 1, step=10))
    # ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    if Rmin is None or Rmax is None:
        Rmin = Ravg_avg.min() * 1.1
        Rmax = max(Ravg_avg.max(), 0.02)
    ax.set_ylim((Rmin, Rmax))
    ax.tick_params(
        labelright=False,
        labelleft=True,
        labeltop=True,
        left=True,
        right=False,
        top=True,
        labelbottom=False,
        bottom=False,
    )
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(reward_files_dir, "avg_reward_runs.png"))
    if show:
        plt.show()
    plt.close(fig)


def _plot_actions_convergence(
    Eaxis: List[float],
    A: List[List[float]],
    MA: List[List[float]],
    markers: List[float],
    markers_name: List[str],
    markersA: List[float],
    nactions: int,
    actions_figsize: Tuple[float, float],
    episodes: int,
    action_range: Optional[Tuple[float, float]],
    discrete: bool,
    save: bool,
    show: bool,
    reward_file: str,
    moving_avg_plot_start: int,
):
    """
    Plot actions convergence over episodes.

    Args:
        Eaxis (List[float]): Episode axis values.
        A (List[List[float]]): Action values.
        MA (List[List[float]]): Moving average action values.
        markers (List[float]): Episode numbers for plotting markers.
        markers_name (List[str]): Names of the markers.
        markersA (List[float]): Action values for the markers.
        nactions (int): Number of actions.
        actions_figsize (Tuple[float, float]): Figure size for actions plot.
        episodes (int): Maximum number of episodes/generations.
        action_range (Optional[Tuple[float, float]]): Range of actions to plot.
        discrete (bool): Whether actions are discrete (plots as scatter if True).
        save (bool): Whether to save the figure.
        show (bool): Whether to show the figure.
        reward_file (str): Path to the processed reward file.
        moving_avg_plot_start (int): Episode number to start plotting moving average.
    """
    figs, axs, nrows, ncols = _make_grid(nactions, actions_figsize)
    for j in range(nactions):
        if isinstance(A[j], (list, np.ndarray)):
            if discrete:
                axs[j].scatter(Eaxis, A[j], color="lightgray", s=10, zorder=1)
            else:
                axs[j].plot(Eaxis, A[j], color="lightgray", linewidth=0.5)
        axs[j].plot(
            Eaxis[moving_avg_plot_start:],
            MA[j][moving_avg_plot_start:],
            color="black",
            linewidth=1,
        )
        if markers:
            axs[j].scatter(
                markers,
                markersA[j],
                s=30,
                facecolors="none",
                edgecolors="darkred",
                zorder=2,
            )
            for i, label in enumerate(markers_name):
                axs[j].annotate(
                    label,
                    (markers[i], markersA[j][i]),
                    textcoords="offset points",
                    xytext=(2, 2.75),
                    ha="center",
                    va="bottom",
                    color="darkred",
                    fontsize=10,
                )
        axs[j].grid(False)
        if action_range:
            action_min, action_max = action_range[0], action_range[1]
        else:
            amin = min(A[j])
            amax = max(A[j])
            action_min, action_max = (1.1 * amin if amin < 0 else 0.9 * amin), (
                1.1 * amax if amax > 0 else 0.9 * amax
            )
        axs[j].set(ylim=(action_min * 1.1, action_max * 1.1), xlim=(0, episodes))
        axs[j].set_xticks(np.arange(0, episodes + 1, step=10))
        axs[j].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ticks = np.arange(action_min, action_max, 5)
        if 0 not in ticks:
            ticks = np.insert(ticks, np.searchsorted(ticks, 0), 0)
        axs[j].set_yticks(ticks)
        axs[j].label_outer()
    for j in range(nactions, len(axs)):
        figs.delaxes(axs[j])
    bottom_row_start = ncols * (nrows - 1)
    middle_col = ncols // 2
    bottom_middle_idx = bottom_row_start + middle_col
    if bottom_middle_idx < nactions:
        axs[bottom_middle_idx].set_xlabel("Episode")
    for j in range(nactions):
        axs[j].set_ylabel(
            rf"$\theta_{{{j+1}}}$", rotation=0, labelpad=10 if j % ncols != 0 else 0
        )
    figs.tight_layout()
    if save:
        figs.savefig(f"{reward_file.split('.')[0]}_actions.png")
    if show:
        plt.show()
    plt.close(figs)


def _plot_actions_convergence_multi(
    Eaxis: List[float],
    A: List[List[List[float]]],
    MA: List[List[List[float]]],
    nactions: int,
    actions_figsize: Tuple[float, float],
    episodes: int,
    action_range: Optional[Tuple[float, float]],
    save: bool,
    show: bool,
    reward_files: List[str],
    moving_avg_plot_start: int,
):
    """
    Plot actions convergence over episodes.

    Args:
        Eaxis (List[float]): Episode axis values.
        A (List[List[List[float]]]): Action values.
        MA (List[List[List[float]]]): Moving average action values.
        nactions (int): Number of actions.
        actions_figsize (Tuple[float, float]): Figure size for actions plot.
        episodes (int): Maximum number of episodes/generations.
        action_range (Optional[Tuple[float, float]]): Range of actions to plot.
        discrete (bool): Whether actions are discrete (plots as scatter if True).
        save (bool): Whether to save the figure.
        show (bool): Whether to show the figure.
        reward_files (List[str]): Paths to the processed reward files.
        moving_avg_plot_start (int): Episode number to start plotting moving average.
    """
    figs, axs, nrows, ncols = _make_grid(nactions, actions_figsize)
    for run in range(len(reward_files)):
        for j in range(nactions):
            axs[j].plot(
                Eaxis[moving_avg_plot_start:],
                MA[run][j][moving_avg_plot_start:],
                linewidth=1.5,
                label="run " + str(run + 1),
                alpha=0.7,
            )
            axs[j].grid(False)
            if action_range:
                action_min, action_max = action_range[0], action_range[1]
            else:
                amin = min(A[run][j])
                amax = max(A[run][j])
                action_min, action_max = (1.1 * amin if amin < 0 else 0.9 * amin), (
                    1.1 * amax if amax > 0 else 0.9 * amax
                )
            axs[j].set(ylim=(action_min * 1.1, action_max * 1.1), xlim=(0, episodes))
            axs[j].set_xticks(np.arange(0, episodes + 1, step=10))
            # axs[j].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ticks = np.arange(action_min, action_max, 5)
            if 0 not in ticks:
                ticks = np.insert(ticks, np.searchsorted(ticks, 0), 0)
            axs[j].set_yticks(ticks)
            axs[j].label_outer()
    for j in range(nactions, len(axs)):
        figs.delaxes(axs[j])
    bottom_row_start = ncols * (nrows - 1)
    middle_col = ncols // 2
    bottom_middle_idx = bottom_row_start + middle_col
    if bottom_middle_idx < nactions:
        axs[bottom_middle_idx].set_xlabel("Episode")
    for j in range(nactions):
        if j == 0:
            axs[j].legend()
        axs[j].set_ylabel(
            rf"$\theta_{{{j+1}}}$", rotation=0, labelpad=10 if j % ncols != 0 else 0
        )
    figs.tight_layout()
    if save:
        figs.savefig(os.path.join(os.path.dirname(reward_files[0]), "avg_actions.png"))
    if show:
        plt.show()
    plt.close(figs)


def _compute_uniques_and_distances(
    A: List[List[float]] | np.ndarray, R: List[float] | np.ndarray
) -> Tuple[List[Tuple[float, float, List[float]]], List[List[float]], List[float]]:
    """
    Compute unique actions, their corresponding rewards, and distances from the best action.

    Args:
        A (List[List[float]] | np.ndarray): List of actions.
        R (List[float] | np.ndarray): List of rewards.
    Returns:
        Tuple containing:
            - List of tuples (reward, distance, action) sorted by distance.
            - List of unique actions.
            - List of corresponding rewards for unique actions.
    """
    A_T = np.array(A).T.tolist()
    A_uniques, indices = np.unique(A_T, axis=0, return_index=True)
    R_corresp_uniques = [R[i] for i in indices]
    best_action = A_uniques[np.argmax(R_corresp_uniques)]
    A_distances = [distance_actions(best_action, action) for action in A_uniques]
    if np.ptp(A_distances) == 0:
        A_distances = np.zeros_like(A_distances)
    else:
        A_distances = (A_distances - np.min(A_distances)) / (
            np.max(A_distances) - np.min(A_distances)
        )
    reward_distance_pairs = list(zip(R_corresp_uniques, A_distances, A_uniques))
    reward_distance_pairs.sort(key=lambda x: x[1])
    return reward_distance_pairs, A_uniques, R_corresp_uniques


def _plot_reward_stability(
    reward_distance_pairs: List[Tuple[float, float, List[float]]],
    Rmin: float,
    Rmax: float,
    reward_figsize: tuple,
    save: bool,
    show: bool,
    reward_file: str,
):
    """
    Plot reward stability showing the relationship between distance from best action
    and reward.

    Args:
        reward_distance_pairs (List[Tuple[float, float, List[float]]]): List of tuples containing
            (reward, distance, action) sorted by distance.
        Rmin (float): Minimum reward value for y-axis.
        Rmax (float): Maximum reward value for y-axis.
        reward_figsize (tuple): Figure size for reward stability plot.
        save (bool): Whether to save the figure.
        show (bool): Whether to show the figure.
        reward_file (str): Path to the processed reward file.
    """
    distances = [p[1] for p in reward_distance_pairs]
    rewards = [p[0] for p in reward_distance_pairs]
    sorted_idx = np.argsort(distances)
    sorted_distances = np.array(distances)[sorted_idx]
    sorted_rewards = np.array(rewards)[sorted_idx]
    window_size = min(10, max(1, len(sorted_rewards)))
    if len(sorted_rewards) >= window_size:
        mean_rewards = np.convolve(
            sorted_rewards, np.ones(window_size) / window_size, mode="valid"
        )
        mean_distances = sorted_distances[: len(mean_rewards)]
        smoothed_mean_rewards = gaussian_filter1d(mean_rewards, sigma=2)
    else:
        mean_distances = sorted_distances
        smoothed_mean_rewards = sorted_rewards
    fig_stability, ax_stability = plt.subplots(figsize=reward_figsize)
    ax_stability.plot(
        mean_distances,
        smoothed_mean_rewards,
        color="black",
        linewidth=1,
        label="mean density",
    )
    ax_stability.scatter(distances, rewards, c="lightgray", s=5)
    ax_stability.legend()
    ax_stability.set_xlabel(r"$\| \theta_i - \theta^* \|_2$")
    ax_stability.set_ylabel("$r$", rotation=0)
    ax_stability.set_ylim((Rmin, Rmax))
    ax_stability.grid(False)
    fig_stability.tight_layout()
    if save:
        fig_stability.savefig(f"{reward_file.split('.')[0]}_reward_stability.png")
    if show:
        plt.show()
    plt.close(fig_stability)


def _plot_action_map(
    reward_distance_pairs: List[Tuple[float, float, List[float]]],
    A_uniques: List[List[float]],
    R_corresp_uniques: List[float],
    A: List[List[float]],
    R: List[float],
    nactions: int,
    action_range: List[float] | None,
    actions_figsize: tuple,
    Rmin: float,
    Rmax: float,
    save: bool,
    show: bool,
    reward_file: str,
    markers: List[float] | None,
    markers_name: List[str] | None,
    markersA: List[float] | None,
    markersY: List[float] | None,
):
    """
    Plot action map showing the relationship between each action and the reward.

    Args:
        reward_distance_pairs (List[Tuple[float, float, List[float]]]): List of tuples containing
            (reward, distance, action) sorted by distance.
        A_uniques (List[List[float]]): Unique actions.
        R_corresp_uniques (List[float]): Corresponding rewards for unique actions.
        A (List[List[float]]): All actions.
        R (List[float]): All rewards.
        nactions (int): Number of actions.
        action_range (List[float] | None): Range of actions to plot.
        actions_figsize (tuple): Figure size for actions plot.
        Rmin (float): Minimum reward value for y-axis.
        Rmax (float): Maximum reward value for y-axis.
        save (bool): Whether to save the figure.
        show (bool): Whether to show the figure.
        reward_file (str): Path to the processed reward file.
        markers (List[float] | None): Episode numbers for plotting markers.
        markers_name (List[str] | None): Names of the markers.
        markersA (List[float] | None): Action values for the markers.
        markersY (List[float] | None): Reward values for the markers.
    """
    figs_map, axs_map, _, _ = _make_grid(nactions, actions_figsize)
    marker_best = reward_distance_pairs[0]
    A_uniques_T = np.array(A_uniques).T.tolist()
    for j in range(nactions):
        unique_actions_j = np.unique(A_uniques_T[j])
        avg_rewards, min_rewards, max_rewards = [], [], []
        for val in unique_actions_j:
            rr = [
                R_corresp_uniques[i]
                for i in range(len(A_uniques_T[j]))
                if A_uniques_T[j][i] == val
            ]
            avg_rewards.append(np.mean(rr))
            min_rewards.append(np.min(rr))
            max_rewards.append(np.max(rr))
        axs_map[j].plot(unique_actions_j, avg_rewards, color="black", linewidth=1)
        axs_map[j].fill_between(
            unique_actions_j, min_rewards, max_rewards, color="lightgray", alpha=0.3
        )
        axs_map[j].scatter(A_uniques_T[j], R_corresp_uniques, color="lightgray", s=2)
        axs_map[j].scatter(
            marker_best[2][j],
            marker_best[0],
            marker="+",
            s=30,
            linewidths=1,
            color="darkred",
            label=r"$\theta^*$",
        )
        axs_map[j].set_xlabel(rf"$\theta_{j+1}$")
        axs_map[j].set_ylabel(r"$r$", rotation=0)
        if markers:
            axs_map[j].scatter(
                markersA[j],
                markersY,
                s=30,
                facecolors="none",
                edgecolors="darkred",
                zorder=2,
            )
            for i, label in enumerate(markers_name):
                axs_map[j].annotate(
                    label,
                    (markersA[j][i], markersY[i]),
                    textcoords="offset points",
                    xytext=(2, 2.75),
                    ha="center",
                    va="bottom",
                    color="darkred",
                    fontsize=10,
                )
        axs_map[j].grid(False)
        if action_range:
            action_min, action_max = action_range[0], action_range[1]
        else:
            amin = min(A[j])
            amax = max(A[j])
            action_min, action_max = (1.1 * amin if amin < 0 else 0.9 * amin), (
                1.1 * amax if amax > 0 else 0.9 * amax
            )
        axs_map[j].set(xlim=(action_min * 1.1, action_max * 1.1), ylim=(Rmin, Rmax))
        axs_map[j].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ticks = np.arange(
            np.ceil(action_min / 5) * 5, np.floor(action_max / 5) * 5 + 1, 5
        )
        if 0 not in ticks:
            ticks = np.insert(ticks, np.searchsorted(ticks, 0), 0)
        axs_map[j].set_xticks(ticks)
        if j == 0:
            axs_map[j].legend()
    for j in range(nactions, len(axs_map)):
        figs_map.delaxes(axs_map[j])
    for j in range(nactions):
        axs_map[j].set_xlabel(rf"$\theta_{{{j+1}}}$")
    figs_map.tight_layout()
    if save:
        figs_map.savefig(f"{reward_file.split('.')[0]}_actions_map.png")
    if show:
        plt.show()
    plt.close(figs_map)


def plot_pbo(
    reward_file: str,
    markers: List[float] | None = None,
    env: int = 8,
    episodes: int = 100,
    nactions: int = 6,
    scale_reward: float = 1,
    scale_actions: float = 1,
    discretize_actions_set: np.ndarray | None = None,
    reward_figsize: tuple = (4, 2.5),
    actions_figsize: tuple = (8, 3.05),
    reward_range: List[float] | None = None,
    action_range: List[float] | None = [-30, 30],
    discrete: bool = False,
    plot_reward_stability: bool = True,
    plot_action_map: bool = True,
    save: bool = False,
    show: bool = False,
    moving_avg_plot_start: int = 0,
):
    """
    Plot the PBO results including reward convergence, actions convergence,
    reward stability, and action map.

    Args:
        reward_file (str): Path to the processed reward file.
        markers (List[float] | None): Episode numbers for plotting markers.
        env (int): Number of environments/individuals per episode/generation.
        episodes (int): Maximum number of episodes/generations.
        nactions (int): Number of actions.
        scale_reward (float): Scaling factor for reward values.
        scale_actions (float): Scaling factor for action values.
        discretize_actions_set (np.ndarray | None): Set of discrete actions to remap to.
        reward_figsize (tuple): Figure size for reward plot.
        actions_figsize (tuple): Figure size for actions plot.
        reward_range (List[float] | None): Axis range of rewards to plot.
        action_range (List[float] | None): Range of actions to plot.
        discrete (bool): Whether to plot discrete actions as scatters.
        plot_reward_stability (bool): Whether to plot reward stability.
        plot_action_map (bool): Whether to plot action map.
        save (bool): Whether to save the figures.
        show (bool): Whether to show the figures.
        moving_avg_plot_start (int): Episode number to start plotting moving average.
    """
    Eaxis, R, MR, A, MA = _read_processed_reward(
        reward_file=reward_file,
        nactions=nactions,
        scale_reward=scale_reward,
        scale_actions=scale_actions,
        discretize_actions_set=discretize_actions_set,
    )
    markers_name, markersY, markersA = _compute_markers(
        markers=markers, env=env, R=R, A=A, nactions=nactions
    )
    Rmin, Rmax = reward_range if reward_range else (1.1 * min(R), 0)
    _plot_reward_convergence(
        Eaxis=Eaxis,
        R=R,
        MR=MR,
        markers=markers,
        markers_name=markers_name,
        markersY=markersY,
        reward_figsize=reward_figsize,
        episodes=episodes,
        Rmin=Rmin,
        Rmax=Rmax,
        save=save,
        show=show,
        reward_file=reward_file,
        moving_avg_plot_start=moving_avg_plot_start,
    )
    # actions convergence
    _plot_actions_convergence(
        Eaxis=Eaxis,
        A=A,
        MA=MA,
        markers=markers,
        markers_name=markers_name,
        markersA=markersA,
        nactions=nactions,
        actions_figsize=actions_figsize,
        episodes=episodes,
        action_range=action_range,
        discrete=discrete,
        save=save,
        show=show,
        reward_file=reward_file,
        moving_avg_plot_start=moving_avg_plot_start,
    )
    reward_distance_pairs = None
    if plot_reward_stability or plot_action_map:
        reward_distance_pairs, A_uniques, R_corresp_uniques = (
            _compute_uniques_and_distances(A=A, R=R)
        )
    if plot_reward_stability and reward_distance_pairs:
        _plot_reward_stability(
            reward_distance_pairs=reward_distance_pairs,
            Rmin=Rmin,
            Rmax=Rmax,
            reward_figsize=reward_figsize,
            save=save,
            show=show,
            reward_file=reward_file,
        )
    if plot_action_map and reward_distance_pairs:
        _plot_action_map(
            reward_distance_pairs=reward_distance_pairs,
            A_uniques=A_uniques,
            R_corresp_uniques=R_corresp_uniques,
            A=A,
            R=R,
            nactions=nactions,
            action_range=action_range,
            actions_figsize=actions_figsize,
            Rmin=Rmin,
            Rmax=Rmax,
            save=save,
            show=show,
            reward_file=reward_file,
            markers=markers,
            markers_name=markers_name,
            markersA=markersA,
            markersY=markersY,
        )


def plot_pbo_multi(
    reward_files_dir: str,
    processed_files_prefix: str,
    scale_reward: float = 1,
    scale_actions: float = 1,
    discretize_actions_set: np.ndarray | None = None,
    nactions: int = 6,
    reward_figsize: tuple = (4, 2.5),
    actions_figsize: tuple = (8, 3.05),
    reward_range: List[float] | None = None,
    action_range: List[float] | None = [-30, 30],
    save: bool = False,
    show: bool = False,
):
    """
    Plot the PBO results including reward convergence and actions convergence
    for multiple runs.

    Args:
        reward_files_dir (str): Directory containing pbo and processed reward files.
        processed_files_prefix (str): Prefix for processed reward file names.
        scale_reward (float): Scaling factor for reward values.
        scale_actions (float): Scaling factor for action values.
        discretize_actions_set (np.ndarray | None): Set of discrete actions to remap to.
        nactions (int): Number of actions.
        reward_figsize (tuple): Figure size for reward plot.
        actions_figsize (tuple): Figure size for actions plot.
        reward_range (List[float] | None): Axis range of rewards to plot.
        action_range (List[float] | None): Range of actions to plot.
        discrete (bool): Whether to plot discrete actions as scatters.
        save (bool): Whether to save the figures.
        show (bool): Whether to show the figures.
    """
    # load data
    processed_files = [
        f
        for f in os.listdir(reward_files_dir)
        if f.startswith(processed_files_prefix) and f.endswith(".txt")
    ]
    processed_files.sort()
    # gather data
    processed_files_str = "\n\t".join(processed_files)
    print(f"Found {len(processed_files)} processed runs:\n\t{processed_files_str}")
    if len(processed_files) == 0:
        print("\tExiting.")
        return
    # avg of avg data
    Eaxis, Ravg_avg, Ravg_p, Ravg_m, Rbest, Rbest_p, Rbest_m = (
        _read_pbo_output_reward_data(
            reward_files_dir=reward_files_dir,
            number_runs=len(processed_files),
            scale_reward=scale_reward,
        )
    )
    # gather each run avg data
    Eaxis_ind, Ravg_avg_ind, Ravg_best_ind = [], [], []
    for individual_run in range(len(processed_files)):
        eaxis_ind_, ravg_avg_ind_, _, _, ravg_best_ind_, _, _ = (
            _read_pbo_output_reward_data(
                reward_files_dir=reward_files_dir,
                scale_reward=scale_reward,
                specific_run_id=individual_run,
            )
        )
        Eaxis_ind.append(eaxis_ind_)
        Ravg_avg_ind.append(ravg_avg_ind_)
        Ravg_best_ind.append(ravg_best_ind_)
    Eaxis_ind = np.array(Eaxis_ind).T
    Ravg_avg_ind = np.array(Ravg_avg_ind).T
    Ravg_best_ind = np.array(Ravg_best_ind).T

    episodes = Eaxis.shape[0]
    Eaxis_all, A_all, MA_all = [], [], []
    for f in processed_files:
        reward_file = os.path.join(reward_files_dir, f)
        Eaxis_, _, _, A_, MA_ = _read_processed_reward(
            reward_file=reward_file,
            nactions=nactions,
            scale_reward=scale_reward,
            scale_actions=scale_actions,
            discretize_actions_set=discretize_actions_set,
        )
        Eaxis_all.append(Eaxis_)
        A_all.append(A_)
        MA_all.append(MA_)
    # plot
    _plot_reward_convergence_multi(
        Eaxis=Eaxis,
        Ravg_avg=Ravg_avg,
        Ravg_p=Ravg_p,
        Ravg_m=Ravg_m,
        Rbest=Rbest,
        Rbest_p=Rbest_p,
        Rbest_m=Rbest_m,
        reward_figsize=reward_figsize,
        Rmin=reward_range[0] if reward_range else None,
        Rmax=reward_range[1] if reward_range else None,
        reward_files_dir=reward_files_dir,
        save=save,
        show=show,
    )
    _plot_reward_convergence_multi_individual(
        Eaxis=Eaxis_ind,
        Ravg_avg=Ravg_avg_ind,
        Ravg_best=Ravg_best_ind,
        reward_figsize=reward_figsize,
        Rmin=reward_range[0] if reward_range else None,
        Rmax=reward_range[1] if reward_range else None,
        reward_files_dir=reward_files_dir,
        save=save,
        show=show,
    )
    _plot_actions_convergence_multi(
        Eaxis=Eaxis_all[0],
        A=A_all,
        MA=MA_all,
        nactions=nactions,
        actions_figsize=actions_figsize,
        episodes=episodes,
        action_range=action_range,
        save=save,
        show=show,
        reward_files=processed_files,
        moving_avg_plot_start=0,
    )


def _parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot PBO results (single run or multi-run).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # single run subparser (maps to plot_pbo)
    p_single = subparsers.add_parser(
        "single",
        help="Plot a single processed reward file (uses plot_pbo).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_single.add_argument("reward_file", type=str, help="Processed reward file path.")
    p_single.add_argument(
        "--markers",
        type=float,
        nargs="+",
        default=None,
        required=False,
        help="Episode numbers (ex:32.75 for env 6 of ep 32 if 8 env/episode) for plotting markers, separate by space.",
    )
    p_single.add_argument(
        "--nactions", type=int, default=6, help="Number of actions (default: 6)."
    )
    p_single.add_argument(
        "--nep",
        type=int,
        default=100,
        help="Maximum number of episodes/generations.",
    )
    p_single.add_argument(
        "--nenv",
        type=int,
        default=8,
        help="Number of environments/individuals per episode/generation.",
    )
    p_single.add_argument(
        "--save", action="store_true", help="Saves the figures if passed."
    )
    p_single.add_argument(
        "--show", action="store_true", help="Shows the figures if passed."
    )
    p_single.add_argument(
        "--actionrange",
        type=float,
        nargs=2,
        required=False,
        default=[-30, 30],
        help="Range of actions to plot, separate by space character (min,max of yaxis).",
    )
    p_single.add_argument(
        "--discrete_actions_step",
        type=float,
        default=None,
        required=False,
        help=(
            "Step size to create discrete action set for remapping continuous actions to discrete ones. "
            "If provided, actions will be remapped to the closest values in a set created from discretizing "
            "the action range with the given step size. If not provided, actions are treated as continuous."
        ),
    )
    p_single.add_argument(
        "--rewardrange",
        type=float,
        nargs=2,
        required=False,
        help=(
            "Axis range of rewards to plot, separate by space character (min,max of yaxis). "
            "Defaults to auto range based on data."
        ),
    )
    p_single.add_argument(
        "--discrete",
        action="store_true",
        help="Discrete actions plotted as scatters in action plot if passed.",
    )
    p_single.add_argument(
        "--figsize_reward",
        type=float,
        nargs=2,
        default=(6, 3.125),
        help="Figure size for reward plot, separate by space character (width,height).",
    )
    p_single.add_argument(
        "--figsize_actions",
        type=float,
        nargs=2,
        default=(8, 3.05),
        help="Figure size for actions plot, separate by space character (width,height).",
    )
    p_single.add_argument(
        "--scale_reward",
        type=float,
        default=1,
        help="Scaling factor for reward values.",
    )
    p_single.add_argument(
        "--scale_actions",
        type=float,
        default=1,
        help="Scaling factor for action values.",
    )
    p_single.add_argument(
        "--plot_reward_stability",
        action="store_true",
        help="Plot reward stability if passed.",
    )
    p_single.add_argument(
        "--plot_action_map",
        action="store_true",
        help="Plot action map if passed.",
    )
    p_single.add_argument(
        "--moving_avg_start",
        type=int,
        default=0,
        help="Env number from which to start plotting moving average reward and actions.",
    )

    # multi-run subparser (maps to plot_pbo_multi)
    p_multi = subparsers.add_parser(
        "multi",
        help="Plot multiple runs from a directory (uses plot_pbo_multi).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_multi.add_argument(
        "reward_files_dir",
        type=str,
        help="Directory containing pbo/processed reward files.",
    )
    p_multi.add_argument(
        "processed_files_prefix",
        type=str,
        help="Prefix for processed reward file names (used to discover files).",
    )
    p_multi.add_argument(
        "--scale_reward",
        type=float,
        default=1,
        help="Scaling factor for reward values.",
    )
    p_multi.add_argument(
        "--scale_actions",
        type=float,
        default=1,
        help="Scaling factor for action values.",
    )
    p_multi.add_argument(
        "--discrete_actions_step",
        type=float,
        default=None,
        required=False,
        help=(
            "Step size to create discrete action set for remapping continuous actions to discrete ones. "
            "If provided, actions will be remapped to the closest values in a set created from discretizing "
            "the action range with the given step size. If not provided, actions are treated as continuous."
        ),
    )
    p_multi.add_argument("--nactions", type=int, default=6, help="Number of actions.")
    p_multi.add_argument(
        "--figsize_reward",
        type=float,
        nargs=2,
        default=(6, 3.125),
        help="Figure size for reward plot, separate by space character (width,height).",
    )
    p_multi.add_argument(
        "--figsize_actions",
        type=float,
        nargs=2,
        default=(8, 3.05),
        help="Figure size for actions plot, separate by space character (width,height).",
    )
    p_multi.add_argument(
        "--rewardrange",
        type=float,
        nargs=2,
        required=False,
        help="Axis range of rewards to plot, separate by space character (min,max of yaxis).",
    )
    p_multi.add_argument(
        "--actionrange",
        type=float,
        nargs=2,
        required=False,
        default=[-30, 30],
        help="Range of actions to plot, separate by space character (min,max of yaxis).",
    )
    p_multi.add_argument(
        "--save", action="store_true", help="Saves the figures if passed."
    )
    p_multi.add_argument(
        "--show", action="store_true", help="Shows the figures if passed."
    )

    return parser.parse_args()


def main():
    # Parser
    args = _parser()

    # fig Parameters
    plt.style.use(scientific_style)

    # discrete actions set
    discretize_actions_set = None
    if args.discrete_actions_step:
        discretize_actions_set = np.linspace(
            args.actionrange[0],
            args.actionrange[1],
            int(
                (args.actionrange[1] - args.actionrange[0])
                / args.discrete_actions_step,
            )
            + 1,
        )

    if args.command == "multi":
        # plot
        plot_pbo_multi(
            reward_files_dir=args.reward_files_dir,
            processed_files_prefix=args.processed_files_prefix,
            scale_reward=args.scale_reward,
            scale_actions=args.scale_actions,
            discretize_actions_set=discretize_actions_set,
            nactions=args.nactions,
            reward_figsize=args.figsize_reward,
            actions_figsize=args.figsize_actions,
            reward_range=args.rewardrange,
            action_range=args.actionrange,
            save=args.save,
            show=args.show,
        )
        return
    elif args.command == "single":
        if not args.reward_file.endswith(".txt"):  # check reward has been processed
            raise ValueError("Reward file provided has not been processed.")
        # plot
        plot_pbo(
            reward_file=args.reward_file,
            markers=args.markers,
            env=args.nenv,
            episodes=args.nep,
            nactions=args.nactions,
            scale_reward=args.scale_reward,
            scale_actions=args.scale_actions,
            discretize_actions_set=discretize_actions_set,
            reward_figsize=args.figsize_reward,
            actions_figsize=args.figsize_actions,
            reward_range=args.rewardrange,
            action_range=args.actionrange,
            discrete=args.discrete,
            plot_reward_stability=args.plot_reward_stability,
            plot_action_map=args.plot_action_map,
            save=args.save,
            show=args.show,
            moving_avg_plot_start=args.moving_avg_start,
        )


if __name__ == "__main__":
    main()
