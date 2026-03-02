import argparse
import os
import json
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.style.use("seaborn-v0_8-poster")
plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
        "axes.labelsize": 24,
        "axes.titlesize": 32,
        "legend.fontsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    }
)


def plot_config(
    config_name: str,
    pool_path: str | None,
    configs_pool: pd.DataFrame | None,
    meta_path: str,
    ax: None | plt.Axes = None,
    save_path: str = None,
    show: bool = False,
    opacity: float = 1,
    color: np.ndarray | None = None,
    add_title: bool = True,
) -> plt.Axes:
    """
    Plot a single configuration from the pool of configurations.

    Args:
    config_name (str): Name of the configuration to plot.
    pool_path (str | None): Path to pool of configurations (if None, will load from pool path in case params).
    meta_path (str): Path to parameters of the case (meta json file path).
    ax (None | plt.Axes): Axis object to plot the configuration. If None, it will create a new axis.
    save_path (str): Path to directory where a "config_name.png" file will be saved.
    show (bool): If True, it will show the plot.

    Returns:
    Axes(plt.Axes) object with the plot. If ax is not None, it will return the same ax object.
    """
    # general params
    with open(meta_path, "r") as fp:
        case_params = json.load(fp)
    case = case_params["case"]
    if pool_path is not None:
        if pool_path.endswith(".txt"):
            pool_path = pool_path.replace(".txt", ".pkl")
        configs_pool = pd.read_pickle(pool_path)
    elif pool_path is None and configs_pool is None:
        pool_path = case_params["configs_pool_path"]
        if pool_path.endswith(".txt"):
            pool_path = pool_path.replace(".txt", ".pkl")
        configs_pool = pd.read_pickle(pool_path)
    elif pool_path is None and configs_pool is not None:
        pass
    else:
        raise ValueError(
            "Either pool_path or configs_pool should be provided. Not both!"
        )

    # plot params
    fill_obj = True
    line_width_dom = 6
    color_area = "darkred"

    if ax is None:
        geometry_params = case_params["geometry_parameters"]
        domain_size = (
            case_params["domain_parameters"]["dx"],
            case_params["domain_parameters"]["dy"],
        )
        domain_origin = (
            case_params["domain_parameters"]["origin_x"],
            case_params["domain_parameters"]["origin_y"],
        )
        # create figure and axis for the case
        fig, ax = plt.subplots(1, 1, figsize=(domain_size[0], domain_size[1]), dpi=150)
        ax.spines["top"].set_linewidth(line_width_dom)
        ax.spines["right"].set_linewidth(line_width_dom)
        ax.spines["bottom"].set_linewidth(line_width_dom)
        ax.spines["left"].set_linewidth(line_width_dom)
        # set domain
        ax.set_xlim(domain_origin[0], domain_origin[0] + domain_size[0])
        ax.set_ylim(domain_origin[1], domain_origin[1] + domain_size[1])
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        # Plot area of object placements
        x_area = geometry_params["x_object"]
        y_area = geometry_params["y_object"]
        corners = [
            (x_area[0], y_area[0]),
            (x_area[0], y_area[1]),
            (x_area[1], y_area[1]),
            (x_area[1], y_area[0]),
        ]
        for i in range(len(corners)):
            x_start, y_start = corners[i]
            x_end, y_end = corners[(i + 1) % len(corners)]
            ax.plot(
                [x_start, x_end],
                [y_start, y_end],
                linestyle="dotted",
                linewidth=line_width_dom,
                color=color_area,
            )

    if case == "cylinders":
        config = configs_pool[configs_pool["Config"] == config_name]
        # plot
        radii = config["radius_objects"].values[0]
        x_coords = config["x_objects"].values[0]
        y_coords = config["y_objects"].values[0]
        for i in range(len(radii)):
            circle = plt.Circle(
                (float(x_coords[i]), float(y_coords[i])),
                float(radii[i]),
                color=color if color is not None else "black",
                fill=fill_obj,
                alpha=opacity,
                linewidth=0.75 * line_width_dom,
            )
            ax.add_artist(circle)
    elif case == "airfoil":
        raise NotImplementedError("Plotting for airfoil case is not implemented yet.")
    else:
        raise NotImplementedError("Plotting for other cases is not implemented yet.")
    if save_path is not None:
        file_name = f"{case}_{config_name}.png"
        if add_title:
            plt.title(f"{case} config {config_name}", fontsize=45, pad=30)
        plt.savefig(os.path.join(save_path, file_name))
        print(f"Saved plot of {config_name} at {os.path.join(save_path, file_name)}")
    if show:
        plt.show()
    return ax


def plot_all_configs(
    meta_path: str,
    pool_path: str | None,
    save_path: str = None,
    show=False,
    add_title: bool = True,
) -> None:
    """
    Plot all configurations from the pool of configurations.

    Args:
    configs_pool (Configs|None): Pool of configurations (will load Configs object from pool path in case params).
    case_params (dict): Parameters of the case (meta json file loaded as dict).
    save_path (str): Path of directory to save the plot.
    """
    # general params
    with open(meta_path, "r") as fp:
        case_params = json.load(fp)
    if pool_path is None:
        pool_path = case_params["configs_pool_path"]
        if pool_path.endswith(".txt"):
            pool_path = pool_path.replace(".txt", ".pkl")
    configs_pool = pd.read_pickle(pool_path)

    # set plot
    line_width_dom = 6
    color_area = "darkgrey"  # darkred

    geometry_params = case_params["geometry_parameters"]
    domain_size = (
        case_params["domain_parameters"]["dx"],
        case_params["domain_parameters"]["dy"],
    )
    domain_origin = (
        case_params["domain_parameters"]["origin_x"],
        case_params["domain_parameters"]["origin_y"],
    )
    # create figure and axis for the case
    fig, ax = plt.subplots(1, 1, figsize=(domain_size[0], domain_size[1]), dpi=150)
    ax.spines["top"].set_linewidth(line_width_dom)
    ax.spines["right"].set_linewidth(line_width_dom)
    ax.spines["bottom"].set_linewidth(line_width_dom)
    ax.spines["left"].set_linewidth(line_width_dom)
    # set domain
    ax.set_xlim(domain_origin[0], domain_origin[0] + domain_size[0])
    ax.set_ylim(domain_origin[1], domain_origin[1] + domain_size[1])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    # Plot area of object placements
    x_area = geometry_params["x_object"]
    y_area = geometry_params["y_object"]
    corners = [
        (x_area[0], y_area[0]),
        (x_area[0], y_area[1]),
        (x_area[1], y_area[1]),
        (x_area[1], y_area[0]),
    ]
    for i in range(len(corners)):
        x_start, y_start = corners[i]
        x_end, y_end = corners[(i + 1) % len(corners)]
        ax.plot(
            [x_start, x_end],
            [y_start, y_end],
            linestyle="dashed",
            linewidth=line_width_dom,
            color=color_area,
        )

    # color by radius (for cylinders case)
    case = case_params["case"]
    if (
        case == "cylinders"
        and case_params["geometry_parameters"]["number_cylinders"][1] < 2
    ):
        min_diam = case_params["geometry_parameters"]["radius"][0] * 2
        max_diam = case_params["geometry_parameters"]["radius"][1] * 2
        norm = plt.Normalize(vmin=min_diam, vmax=max_diam)
        cmap = cm.copper
        # cmap = cm.get_cmap("copper", 14)

        # Plot each config, coloring by radius
        opacity = 0.5 + min(0.5, 0.5 / (10 * len(configs_pool)))
        for idx, config in configs_pool.iterrows():
            radii = config["radius_objects"]
            diams = [r * 2 for r in radii]
            x_coords = config["x_objects"]
            y_coords = config["y_objects"]
            for i in range(len(radii)):
                color = cmap(norm(diams[i]))
                circle = plt.Circle(
                    (float(x_coords[i]), float(y_coords[i])),
                    float(radii[i]),
                    color=color,
                    fill=True,
                    alpha=opacity,
                    linewidth=0,  # No border
                )
                ax.add_artist(circle)
    else:
        # fallback: color by config index
        num_configs = len(configs_pool["Config"].values)
        colors = cm.copper(np.linspace(0, 1, num_configs))
        opacity = 0.5 + min(0.5 / (10 * num_configs), 0.5)
        for i, config in enumerate(configs_pool["Config"].values):
            plot_config(
                config_name=config,
                pool_path=None,
                configs_pool=configs_pool,
                meta_path=meta_path,
                ax=ax,
                save_path=None,
                show=False,
                opacity=opacity,
                color=colors[i],
            )

    if save_path is not None:
        file_name = f"{case_params['case']}_all_configs.png"
        if add_title:
            title = os.path.basename(pool_path).replace("_", " ").rsplit(".", 1)[0]
            plt.title(title, fontsize=45, pad=30)
        if (
            case == "cylinders"
            and case_params["geometry_parameters"]["number_cylinders"][1] < 2
        ):
            sm = cm.ScalarMappable(
                # cmap=cm.copper,
                cmap=cm.get_cmap("copper", 14),
                norm=plt.Normalize(vmin=min_diam, vmax=max_diam),
            )
            sm.set_array([])
            divider = make_axes_locatable(ax)
            # The height of the colorbar matches the axes, width is 1/8 of height
            cbar_ax = divider.append_axes("right", size="4%", pad=0.4)
            cbar = plt.colorbar(sm, cax=cbar_ax)
            # Set 4 ticks, formatted to 1 decimal
            ticks = np.linspace(min_diam, max_diam, 4)
            cbar.set_ticks(ticks)
            cbar.ax.set_yticklabels([f"{tick:.1f}" for tick in ticks])
            cbar.set_label("D", fontsize=60, rotation=0, labelpad=35)
            cbar.ax.tick_params(labelsize=50)
        else:
            sm = cm.ScalarMappable(
                cmap=cm.copper, norm=plt.Normalize(vmin=1, vmax=num_configs)
            )
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Configuration Index", fontsize=45)
            cbar.ax.tick_params(labelsize=45)
        plt.savefig(os.path.join(save_path, file_name))
        print(f"Saved plot of all configs at {os.path.join(save_path, file_name)}")
    if show:
        plt.show()
    plt.close(fig)
    return None


def _parser():
    parser = argparse.ArgumentParser(
        description="Plot configurations from a pool of configurations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "meta_path",
        type=str,
        help="Path to the case meta parameters json file.",
        default="./config/airfoil.json",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        help="Name of a single configuration to plot if desired.",
        required=False,
    )
    parser.add_argument(
        "--pool_path",
        type=str,
        help="Path to the pool of configurations to load if different from path in meta json.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Path to directory where the plots will be saved, if not passed, no figure saved.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="If passed, show the plots.",
        required=False,
    )
    parser.add_argument(
        "--no_title",
        action="store_true",
        help="If passed, do not add a title to the plot.",
        required=False,
    )
    return parser.parse_args()


def main_visualize():
    args = _parser()
    if args.config_name is not None:
        plot_config(
            config_name=args.config_name,
            pool_path=args.pool_path,
            configs_pool=None,
            meta_path=args.meta_path,
            save_path=args.save_path,
            show=args.show,
            add_title=not args.no_title,
        )
    else:
        plot_all_configs(
            meta_path=args.meta_path,
            pool_path=args.pool_path,
            save_path=args.save_path,
            show=args.show,
            add_title=not args.no_title,
        )
    return None


if __name__ == "__main__":
    main_visualize()
