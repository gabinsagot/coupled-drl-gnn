import argparse
import ast
from geometry.generate_airfoil import main_airfoil


def subparser_airfoil(subparsers: argparse._SubParsersAction):
    """
    Add the bluff body geometry generation subparser
    """
    airfoil_parser = subparsers.add_parser(
        "airfoil",
        help="Create airfoil object(s) and fluid domain",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    airfoil_parser.add_argument(
        "-f",
        "--dataset_config",
        type=str,
        default="./config/airfoil.json",
        help="Path to the dataset config file",
    )
    airfoil_parser.add_argument(
        "-naca",
        "--airfoil_type",
        type=str,
        default="NACA0010",
        help="Type of airfoil to use (default=NACA0010). ",
    )
    airfoil_parser.add_argument(
        "-npoints",
        "--points_per_surface",
        type=int,
        default=15,
        help="Number of points per surface (upper/lower) for all airfoils",
    )
    airfoil_parser.add_argument(
        "-c",
        "--chords",
        type=str,
        default="[1.0]",
        help="Chord length(s) for each airfoil (in brackets)",
    )
    airfoil_parser.add_argument(
        "-t",
        "--thicknesses",
        type=str,
        default="[0.5]",
        help="Thickness(es) for each airfoil (in brackets)",
    )
    airfoil_parser.add_argument(
        "-aoa",
        "--angles",
        type=str,
        default="[0.0]",
        help="Angle(s) of attack for each airfoil (in brackets, degrees)",
    )
    airfoil_parser.add_argument(
        "-x",
        "--centersx",
        type=str,
        default="[0.0]",
        help="X coordinate(s) of the leading point(s) of each airfoil (in brackets)",
    )
    airfoil_parser.add_argument(
        "-y",
        "--centersy",
        type=str,
        default="[0.0]",
        help="Y coordinate(s) of the leading point(s) of each airfoil (in brackets)",
    )
    airfoil_parser.add_argument(
        "-n",
        "--number_airfoils",
        type=int,
        default=1,
        help="Number of airfoils to create.",
    )
    airfoil_parser.add_argument(
        "--dim",
        type=int,
        default=2,
        help="Which dimension for geometry and meshing (2D or 3D), provide int 2 or 3",
    )
    airfoil_parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="./cfd_bank/cfd_airfoil",
        help="Path to cfd directory, where the geometry will be created.",
    )
    airfoil_parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean up all outputs, keeping only .t mesh files",
    )
    airfoil_parser.add_argument(
        "--output_dir",
        type=str,
        default="meshes_output",
        help="Output directory name for .t mesh files",
    )


def _parser():
    """
    Create the main parser for the geometry generation
    """
    parser = argparse.ArgumentParser(
        description="Create geometries and fluid domains using GMSH",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="commands",
    )
    subparser_airfoil(subparsers)

    args = parser.parse_args()

    return args


def main_geometry():
    """
    Main function to create a geometry object and fluid domain using GMSH.
    """

    args = _parser()

    if args.commands == "airfoil":
        args.chords = ast.literal_eval(args.chords)
        args.thicknesses = ast.literal_eval(args.thicknesses)
        args.angles = ast.literal_eval(args.angles)
        args.centersx = ast.literal_eval(args.centersx)
        args.centersy = ast.literal_eval(args.centersy)
        main_airfoil(args)
    else:
        print("Unknown command. Use -h for help.")


if __name__ == "__main__":
    main_geometry()
