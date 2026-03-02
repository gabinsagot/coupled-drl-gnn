import argparse
from postprocess.edit_xdmf_names import main_edit
from postprocess.reorder_xdmf import main_reorder
from postprocess.split_dataset import main_split
from postprocess.test_dataset import main_test
from postprocess.interpolate_dataset import main_interpolate
from postprocess.process_simulation import main_process_simulation
from postprocess.postprocess_splits import main_postprocess_splits
from postprocess.mesh_stats import main_mesh_stats
from postprocess.plot_simu import main_plot_simu


def subparser_edit(subparsers: argparse._SubParsersAction):
    """Add subparser for edit_xdmf"""
    edit_xdmf_parser = subparsers.add_parser(
        name="edit",
        help=(
            "Makes sure xdmf/h5 file pairs have matching names and pointers "
            "and updates them if necessary, can do this recursively too. "
            "Also renames fields in the xdmf/h5 pair if desired (old_name -> new_name)."
            "Can also drop fields in the xdmf files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    edit_xdmf_parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default="./",
        help="Root directory of the data (where the xdmf/h5 file pairs are).",
    )
    edit_xdmf_parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively search for xdmf files in the directory.",
    )
    edit_xdmf_parser.add_argument(
        "--rename_fields",
        type=str,
        default=None,
        help='JSON string representing the dictionary for renaming fields. Example: \'{"old_name": "new_name"}\'',
    )
    edit_xdmf_parser.add_argument(
        "--drop_fields",
        type=str,
        default=None,
        help="Comma-separated list of fields to drop in xdmfs. Example: AppartientObject,Pression",
    )


def subparser_reorder(subparsers: argparse._SubParsersAction):
    """Add subparser for reorder"""
    reorder_parser = subparsers.add_parser(
        name="reorder",
        help=(
            "Reorder the timesteps of xdmf/h5 file pairs in a directory, "
            "so that they are in order of time. Provides option to impose a new timestep."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    reorder_parser.add_argument(
        "path",
        type=str,
        help="Path to the directory containing the XDMF files.",
    )
    reorder_parser.add_argument(
        "--outpath",
        type=str,
        required=False,
        help=(
            "Path to the output directory containg the reordered XDMF files. "
            "If none provided, used input path (overwrites)."
        ),
    )
    reorder_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print more information.",
    )
    reorder_parser.add_argument(
        "-dt",
        "--timestep",
        type=float,
        default=1,
        help="Timestep between two frames.",
    )
    reorder_parser.add_argument(
        "--drop_firststep",
        action="store_true",
        help="Drop the first timestep of the xdmf you are reordering.",
    )


def subparser_split(subparsers: argparse._SubParsersAction):
    """Add subparser for splitting xdmf dataset"""
    split_parser = subparsers.add_parser(
        name="split",
        help="Split a dataset folder into train, test, and predict folders, according to a given ratio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    split_parser.add_argument(
        "input_dir", help="Directory containing the .xdmf and .h5 files."
    )
    split_parser.add_argument(
        "-train",
        "--train_dir",
        default="train",
        help="Directory name for training data.",
    )
    split_parser.add_argument(
        "-test",
        "--test_dir",
        default="test",
        help="Directory name for testing/validation data.",
    )
    split_parser.add_argument(
        "-predict",
        "--predict_dir",
        default="predict",
        help="Directory name for prediction data.",
    )
    split_parser.add_argument(
        "--copy", action="store_true", help="Copy files instead of moving them."
    )
    split_parser.add_argument(
        "-r",
        "--ratio",
        type=str,
        default="80,10,10",
        help=(
            "Train, test, predict split ratio as a comma-separated string "
            "of three integers (e.g., '80,10,10'). Must add up to 100."
        ),
    )
    split_parser.add_argument(
        "--reproduce",
        default=None,
        type=str,
        help=(
            "Reproduce a dataset split from a given directory "
            "(must include train/test/predict subdirectories). "
            "This will look for the same configuration IDs in the given directory "
            "and current directory, split the current directory in the same way. "
            "Non-matching IDs will be ignored.split to respect train/loss/split."
        ),
    )
    split_parser.add_argument(
        "--sorted",
        action="store_true",
        help="Sort files instead of shuffling before splitting.",
    )


def subparser_test(subparsers: argparse._SubParsersAction):
    """Add subparser for test_dataset"""
    test_parser = subparsers.add_parser(
        name="test",
        help="Tests the dataset quality by checking that velocity magnitude thresholds, and number of frames.",
    )
    test_parser.add_argument(
        "directory", type=str, help="Directory containing .xdmf/.h5 files"
    )
    test_parser.add_argument(
        "-v", "--threshold", type=float, default=10, help="Velocity threshold"
    )
    test_parser.add_argument(
        "-n",
        "--expected_timesteps",
        type=int,
        default=600,
        help="Expected number of timesteps",
    )
    test_parser.add_argument(
        "-a",
        "--amount",
        type=float,
        default=0.25,
        help="Percentage of timesteps to analyze from the end of the file (between 0 and 1)",
    )
    test_parser.add_argument(
        "--plot",
        action="store_true",
        help=(
            "Compute and save average-velocity-vs-time plots (5x5 subplots per figure) "
            "into a 'plots' folder inside the dataset directory."
        ),
    )


def subparser_interpolate(subparsers: argparse._SubParsersAction):
    """Add subparser for interpolate_dataset"""
    interpolate_parser = subparsers.add_parser(
        name="interpolate",
        help=(
            "Interpolates the values of the frames from one dataset onto another mesh resolution "
            "(e.g.: interpolate a finemesh dataset onto a coarse mesh). "
            "Provide the source and destination meshes."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    interpolate_parser.add_argument(
        "-fine",
        "--fine_directory",
        type=str,
        default="./",
        help="Directory containing the fine mesh files.",
    )
    interpolate_parser.add_argument(
        "-coarse",
        "--coarse_directory",
        type=str,
        default="./",
        help="Directory containing the coarse mesh files.",
    )
    interpolate_parser.add_argument(
        "-out",
        "--out_directory",
        type=str,
        default="./",
        help="Directory to save the interpolated meshes.",
    )
    interpolate_parser.add_argument(
        "--fields",
        type=str,
        nargs="+",
        default=["Vitesse", "Pression", "NodeType", "Reynolds"],
        help="Field names to interpolate (separate by spaces).",
    )
    interpolate_parser.add_argument(
        "--method",
        type=str,
        default="linear",
        help=("Interpolation method (choose from 'nearest','linear','cubic'). "),
    )


def subparser_simu(subparsers: argparse._SubParsersAction):
    """Add subparser for process_simulation"""
    simu_parser = subparsers.add_parser(
        name="simu",
        help="Process CFD simulation results (compress to xdmf, save signals to csv).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    simu_parser.add_argument(
        "simu_dir", type=str, help="Path to the simulation directory."
    )
    simu_parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        default=".",
        help="Path to save the processed results.",
    )
    simu_parser.add_argument(
        "--dim",
        type=int,
        default=2,
        choices=[2, 3],
        help="Dimension of the simulation (2 or 3).",
    )
    simu_parser.add_argument(
        "--vtu_start",
        type=int,
        default=0,
        help="Start time in seconds for vtus to compress.",
    )
    simu_parser.add_argument(
        "--dt", type=float, default=0.2, help="Time step of the simulation."
    )


def subparser_postsplits(subparsers: argparse._SubParsersAction):
    """Add subparser for postprocessing splits"""
    postsplits_parser = subparsers.add_parser(
        name="postsplits",
        help=(
            "Given a path to the split directories (should contain train, test, predict subdirectories), "
            "generates the config dataframes for each split set. Also possibility to plot these "
            "configs with regards to one another to visualize similarity using PCA."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    postsplits_parser.add_argument(
        "-p",
        "--config_pool_path",
        type=str,
        required=True,
        help="Path to the config pool file (must be pickle file, ie .pkl).",
    )
    postsplits_parser.add_argument(
        "-d",
        "--split_dir",
        type=str,
        required=True,
        help="Path to the directory containing the split (train, test, predict) subdirectories.",
    )
    postsplits_parser.add_argument(
        "--plot_pca",
        action="store_true",
        help="If set, plots PCA of the configuration parameters.",
    )
    postsplits_parser.add_argument(
        "--param_cols",
        type=str,
        nargs="+",
        default=["x_objects", "y_objects"],
        help="List of parameter columns to use for PCA (space-separated).",
    )
    postsplits_parser.add_argument(
        "--plot_save_path",
        type=str,
        default=None,
        help="Path to save the PCA plot (if --plot_pca is set).",
    )


def subparser_meshstats(subparsers: argparse._SubParsersAction):
    """Add subparser for mesh statistics"""
    meshstats_parser = subparsers.add_parser(
        name="meshstats",
        help=(
            "Analyze a dataset's XDMF files for retrieving mesh (num of nodes) statistics."
            "Outputs statistics to 'mesh_statistics.txt' in the given directory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    meshstats_parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default=".",
        help=(
            "Directory of dataset to search for .xdmf files. "
            "Will search into dataset splits (train/val/test) recursively, if any, "
            "This is also where the output statistics file will be saved. Defaults to current directory."
        ),
    )
    meshstats_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="If set, prints detailed processing information.",
    )


def subparser_plotsimu(subparsers: argparse._SubParsersAction):
    """add subparser for plotting simulation forces and moments"""
    plotsimu_parser = subparsers.add_parser(
        name="plot",
        help="plot force and moment components from simulation csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    plotsimu_parser.add_argument(
        "csv_path",
        type=str,
        help="path to the csv file containing simulation results.",
    )
    plotsimu_parser.add_argument(
        "-o",
        "--objects",
        type=str,
        default=None,
        help=(
            "comma-separated list of object ids to plot (e.g., '0,1,2'). "
            "if not provided, plots all objects except empty/total."
        ),
    )
    plotsimu_parser.add_argument(
        "--out_path",
        type=str,
        default=".",
        help="directory to save the plot (defaults to current directory).",
    )
    plotsimu_parser.add_argument(
        "--show",
        action="store_true",
        help="display the plot (window).",
    )
    plotsimu_parser.add_argument(
        "--save",
        action="store_true",
        help="save the plot to a file.",
    )

    plotsimu_parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        metavar=("xmin", "xmax"),
        default=None,
        help="set custom x-axis limits (time range).",
    )

    plotsimu_parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        metavar=("ymin", "ymax"),
        default=None,
        help="set custom y-axis limits for all components.",
    )


def _parser():
    parser = argparse.ArgumentParser(
        description="A set of commands to postprocess a dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # Add subparsers
    subparser_edit(subparsers)
    subparser_reorder(subparsers)
    subparser_split(subparsers)
    subparser_test(subparsers)
    subparser_interpolate(subparsers)
    subparser_simu(subparsers)
    subparser_postsplits(subparsers)
    subparser_meshstats(subparsers)
    subparser_plotsimu(subparsers)

    # Parse the arguments
    args = parser.parse_args()
    return args


def main_postprocess():
    args = _parser()
    if args.command == "edit":
        main_edit(args=args)
    elif args.command == "reorder":
        main_reorder(args=args)
    elif args.command == "split":
        args.ratio = list(map(lambda x: int(float(x) + 0.5), args.ratio.split(",")))
        main_split(args=args)
    elif args.command == "test":
        main_test(args=args)
    elif args.command == "interpolate":
        main_interpolate(args=args)
    elif args.command == "simu":
        main_process_simulation(args=args)
    elif args.command == "postsplits":
        main_postprocess_splits(args=args)
    elif args.command == "meshstats":
        main_mesh_stats(args=args)
    elif args.command == "plot":
        main_plot_simu(args=args)
    else:
        print("Unknown command. Please use --help to see available commands.")


if __name__ == "__main__":
    main_postprocess()
