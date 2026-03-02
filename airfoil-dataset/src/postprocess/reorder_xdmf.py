import os
import argparse
from tqdm import tqdm
from utils import meshes_to_xdmf, xdmf_to_meshes


def _parser():
    parser = argparse.ArgumentParser(
        description="Reorder the timesteps of an XDMF archive file."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to the directory containing the XDMF files.",
    )
    parser.add_argument(
        "--outpath",
        type=str,
        required=False,
        help=(
            "Path to the output directory containg the reordered XDMF files. "
            "If none provided, used input path (overwrites)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print more information.",
    )
    parser.add_argument(
        "-dt",
        "--timestep",
        type=float,
        default=1,
        help="Timestep between two frames.",
    )
    parser.add_argument(
        "--drop_firststep",
        action="store_true",
        help="Drop the first timestep of the xdmf you are reordering. (default: False)",
    )
    if not parser.parse_known_args()[0].outpath:
        confirm = (
            input(
                "No output path provided. This will overwrite the input XDMF files. "
                "Do you want to continue? (y/n): "
            )
            .strip()
            .lower()
        )
        if confirm != "y":
            print("Operation cancelled.")
            exit(0)
    return parser.parse_args()


def main_reorder(args: argparse.Namespace | None = None) -> None:
    """
    Main function to reorder the timesteps of XDMF files in a directory.
    """
    if args is None:
        args = _parser()
    path = args.path
    outpath = args.path if args.outpath is None else args.outpath
    verbose = args.verbose
    timestep = args.timestep
    drop_firststep = args.drop_firststep

    # Create the output directory if it doesn't exist
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # List all the XDMF files in the directory
    xdmf_files = [
        f
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f)) and f.endswith(".xdmf")
    ]

    # Loop through all the XDMF files
    print(f"Reordering XDMF files by timestep, in {path}...")
    for xdmf_file in tqdm(
        xdmf_files,
        desc="Reordering each XDMF file's content by timestep",
        disable=not verbose,
    ):
        xdmf_file_path = os.path.join(path, xdmf_file)
        meshes, times = xdmf_to_meshes(xdmf_file_path, verbose=False)

        # Sort the meshes by time
        sorted_time, sorted_mesh = zip(*sorted(zip(times, meshes)))
        sorted_time = list(sorted_time)
        sorted_mesh = list(sorted_mesh)

        # Write the sorted meshes to a new XDMF file
        out_xdmf_file_path = os.path.join(outpath, xdmf_file)
        meshes_to_xdmf(
            filename=out_xdmf_file_path,
            meshes=sorted_mesh,
            timestep=timestep,
            verbose=False,
            drop_firststep=drop_firststep,
        )


if __name__ == "__main__":
    main_reorder(args=None)
