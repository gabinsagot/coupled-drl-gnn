import argparse
import math
import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# mpl style
plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.formatter.useoffset": False,
        "axes.formatter.use_mathtext": True,
    }
)

# colors
COLORS = [
    "darkblue",
    "darkred",
    "darkgreen",
    "darkorange",
    "purple",
    "brown",
    "cyan",
    "magenta",
]


def _clean_object_label(obj: object) -> str:
    """convert object id to clean string (e.g., 0.0 -> "0")."""
    s = str(obj).strip()
    if s.endswith(".0"):
        # Simple cleanup for common float-to-int labels
        try:
            f = float(s)
            if f.is_integer():
                s = str(int(f))
        except ValueError:
            pass
    return s


def parse_objects_string(objects_str: str) -> Optional[List[str]]:
    """parse comma-separated object ids into cleaned strings."""
    if objects_str is None:
        return None
    return [_clean_object_label(obj.strip()) for obj in objects_str.split(",")]


def get_available_objects(df: pd.DataFrame) -> List[str]:
    """get object ids from 'Object' column, cleaned, excluding empty/total."""
    if "Object" in df.columns:
        unique_objects = df["Object"].dropna().unique()
        cleaned = [_clean_object_label(obj) for obj in unique_objects]
        # Filter out empty strings and NaN-like values
        objects = sorted(
            [s for s in cleaned if s and s.lower() != "nan" and s.lower() != "total"]
        )
        return objects if objects else ["0"]
    return ["0"]


def get_plottable_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """identify columns with data: returns (force_cols, moment_cols)."""
    force_cols = []
    moment_cols = []

    # Standard column names for forces and moments
    force_names = ["Fx", "Fy", "Fz"]
    moment_names = ["Mx", "My", "Mz"]

    for col in force_names:
        if col in df.columns and df[col].notna().any():
            force_cols.append(col)

    for col in moment_names:
        if col in df.columns and df[col].notna().any():
            moment_cols.append(col)

    return force_cols, moment_cols


def round_nearest(x: float, base: float = 0.5, mode: str = "nearest") -> float:
    """round to nearest base value."""
    if mode == "ceil":
        return base * math.ceil(float(x) / base)
    elif mode == "floor":
        return base * math.floor(float(x) / base)
    else:
        return base * round(float(x) / base)


def plot_forces_and_moments(
    csv_path: str,
    objects: Optional[List[str]] = None,
    out_path: str = ".",
    show: bool = False,
    save: bool = False,
    time_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
) -> None:
    """plot force/moment components from csv."""
    # load
    df = pd.read_csv(csv_path)

    if "Temps" not in df.columns:
        raise ValueError("CSV file must contain 'Temps' column")

    # Get available objects if not specified
    if objects is None:
        objects = get_available_objects(df)

    # Prepare a cleaned object label column for reliable filtering/legend
    if "Object" in df.columns:
        df["_ObjectLabel"] = df["Object"].apply(_clean_object_label)
        # Filter by objects if provided
        df = df[df["_ObjectLabel"].isin(objects) | df["Object"].isna()]

    print(f"plotting: {csv_path}")
    print(f"objects: {objects}")

    # Get plottable columns
    force_cols, moment_cols = get_plottable_columns(df)
    all_cols = force_cols + moment_cols

    if not all_cols:
        raise ValueError("No force/moment columns with data found in CSV")

    # layout: one subplot per component (rows)
    n_rows = len(all_cols)
    fig, axes = plt.subplots(n_rows, 1, figsize=(6.5, 3.5 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    time_data = df["Temps"]

    # time range for scaling
    if time_range is None:
        # use last 75% of data for scaling
        time_range = (time_data.quantile(0.25), time_data.max())

    # min/max per component across all objects
    limits = {}
    df_range = df[(df["Temps"] >= time_range[0]) & (df["Temps"] <= time_range[1])]
    for col in all_cols:
        valid_data = df_range[col].dropna()
        if y_range is not None:
            limits[col] = y_range
        elif len(valid_data) > 0:
            col_min = round_nearest(valid_data.min(), 0.5, mode="floor")
            col_max = round_nearest(valid_data.max(), 0.5, mode="ceil")
            limits[col] = (col_min, col_max)
        else:
            limits[col] = (0, 1)

    # color per object
    color_cycle = plt.rcParams.get("axes.prop_cycle").by_key().get("color", COLORS)
    obj_colors = {
        obj: color_cycle[i % len(color_cycle)] for i, obj in enumerate(objects)
    }

    # plot: one axis per component, series per object
    for row_idx, col_name in enumerate(all_cols):
        ax = axes[row_idx]
        any_plotted = False
        for obj in objects:
            if "_ObjectLabel" in df.columns:
                df_obj = df[df["_ObjectLabel"] == obj]
            else:
                df_obj = df
            valid_data = df_obj[col_name].dropna()
            if len(valid_data) > 0:
                ax.plot(
                    df_obj["Temps"],
                    df_obj[col_name],
                    linewidth=2,
                    color=obj_colors.get(obj, COLORS[0]),
                    label=f"Obj. {obj}",
                )
                any_plotted = True

        # axis labels (horizontal)
        labelpad = 15 if (df_range[col_name].dropna() >= 0).all() else 10
        ax.set_ylabel(col_name, rotation=0, labelpad=labelpad)
        if row_idx == len(all_cols) - 1:
            ax.set_xlabel("Time (s)")

        # y limits
        if col_name in limits:
            ax.set_ylim(limits[col_name])
        # x limits
        if time_range is not None:
            ax.set_xlim(time_range)

        # grid + formatting
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        if col_name in moment_cols:
            ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        # legend only on first plot
        if row_idx == 0 and any_plotted:
            ax.legend(loc="best")

    # figure title from filename (without extension), '_' -> ' '
    csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
    title_text = csv_basename.replace("_", " ")
    fig.suptitle(title_text, fontsize=18, y=0.995)
    fig.tight_layout()

    # save
    if save:
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        plot_filename = os.path.join(out_path, f"{csv_basename}_forces_moments.png")
        fig.savefig(plot_filename, dpi=150, bbox_inches="tight")
        print(f"plot saved to: {plot_filename}")

    # show
    if show:
        plt.show()
    else:
        plt.close(fig)


def _parser():
    """cli parser."""
    parser = argparse.ArgumentParser(
        description="Plot force and moment components from simulation CSV data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the CSV file containing simulation results (generated by process_simulation.py).",
    )

    parser.add_argument(
        "-o",
        "--objects",
        type=str,
        default=None,
        help=(
            "Comma-separated list of object IDs to plot (e.g., '0,1,2')."
            " If not provided, plots all objects."
        ),
    )

    parser.add_argument(
        "--out_path",
        type=str,
        default=".",
        help="Directory to save the plot. Defaults to current directory.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="display the plot (window).",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="save the plot to a file.",
    )

    parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        metavar=("xmin", "xmax"),
        default=None,
        help="set custom x-axis limits (time range).",
    )

    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        metavar=("ymin", "ymax"),
        default=None,
        help="set custom y-axis limits for all components.",
    )

    return parser.parse_args()


def main_plot_simu(args=None):
    """entry point."""
    if args is None:
        args = _parser()

    try:
        # parse objects
        objects = parse_objects_string(args.objects)

        # run plot
        plot_forces_and_moments(
            csv_path=args.csv_path,
            objects=objects,
            out_path=args.out_path,
            show=args.show,
            save=args.save,
            time_range=tuple(args.xlim) if args.xlim is not None else None,
            y_range=tuple(args.ylim) if args.ylim is not None else None,
        )
        print("done")

    except FileNotFoundError:
        print(f"error: csv file not found at {args.csv_path}")
    except Exception as e:
        print(f"error: {e}")
        raise


if __name__ == "__main__":
    main_plot_simu()
