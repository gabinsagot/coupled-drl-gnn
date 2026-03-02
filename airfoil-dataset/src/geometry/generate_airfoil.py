import argparse
import ast
import os

import gmsh

from convert.mesh import convert_gmsh_to_mtc
from geometries import Airfoil
from utils import (
    load_json_to_dict,
    move_meshes,
    remove_directory,
)


def _parser():
    parser = argparse.ArgumentParser(
        description="Create airfoil object(s) and fluid domain",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-f",
        "--dataset_config",
        type=str,
        default="./config/airfoil.json",
        help="Path to the dataset config file",
    )
    parser.add_argument(
        "-naca",
        "--airfoil_type",
        type=str,
        default="NACA0010",
        help="Type of airfoil to use (default=NACA0010). ",
    )
    parser.add_argument(
        "-npoints",
        "--points_per_surface",
        type=int,
        default=15,
        help="Number of points per surface (upper/lower) for all airfoils",
    )
    parser.add_argument(
        "-c",
        "--chords",
        type=str,
        default="[1.0]",
        help="Chord length(s) for each airfoil (in brackets)",
    )
    parser.add_argument(
        "-t",
        "--thicknesses",
        type=str,
        default="[0.5]",
        help="Thickness(es) for each airfoil (in brackets)",
    )
    parser.add_argument(
        "-aoa",
        "--angles",
        type=str,
        default="[0.0]",
        help="Angle(s) of attack for each airfoil (in brackets, degrees)",
    )
    parser.add_argument(
        "-x",
        "--centersx",
        type=str,
        default="[0.0]",
        help="X coordinate(s) of the leading point(s) of each airfoil (in brackets)",
    )
    parser.add_argument(
        "-y",
        "--centersy",
        type=str,
        default="[0.0]",
        help="Y coordinate(s) of the leading point(s) of each airfoil (in brackets)",
    )
    parser.add_argument(
        "-n",
        "--number_airfoils",
        type=int,
        default=1,
        help="Number of airfoils to create.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=2,
        help="Which dimension for geometry and meshing (2D or 3D), provide int 2 or 3",
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="./cfd_bank/cfd_airfoil",
        help="Path to cfd directory, where the geometry will be created.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean up all outputs, keeping only .t mesh files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="meshes_output",
        help="Output directory name for .t mesh files",
    )
    args = parser.parse_args()
    # reformat some args
    args.chords = ast.literal_eval(args.chords)
    args.thicknesses = ast.literal_eval(args.thicknesses)
    args.angles = ast.literal_eval(args.angles)
    args.centersx = ast.literal_eval(args.centersx)
    args.centersy = ast.literal_eval(args.centersy)
    return args


# EXECUTION


def main_airfoil(args: None | argparse.Namespace = None):
    """
    Main function to create airfoil object(s) and fluid domain using GMSH.
    """
    # Parse arguments
    if args is None:
        args = _parser()

    # create airfoil points
    print(f"\nLoading {args.airfoil_type} airfoil...")
    print(f"Chord length: {args.chords}")
    print(f"Thickness: {args.thicknesses}")

    # load meta dict or create
    try:
        meta_dict = load_json_to_dict(args.dataset_config)
    except FileNotFoundError:
        print(f"Warning: Config file not found at {args.dataset_config}")
        meta_dict = {
            "case": "airfoil_case",
            "domain_parameters": {
                "dx": 10.0,
                "dy": 10.0,
                "dz": 1.0,
                "origin_x": -5.0,
                "origin_y": -5.0,
                "origin_z": 0.0,
            },
            "cfd_parameters": {
                "mesh_adapt": False,
            },
        }
        print("Using default configuration")

    # parser sanity check
    if len(args.chords) != args.number_airfoils:
        raise ValueError(
            f"Number of chords ({len(args.chords)}) must match "
            f"number of airfoils ({args.number_airfoils})"
        )
    if len(args.thicknesses) != args.number_airfoils:
        raise ValueError(
            f"Number of thicknesses ({len(args.thicknesses)}) must match "
            f"number of airfoils ({args.number_airfoils})"
        )
    if len(args.angles) != args.number_airfoils:
        raise ValueError(
            f"Number of angles ({len(args.angles)}) must match "
            f"number of airfoils ({args.number_airfoils})"
        )
    if len(args.centersx) != args.number_airfoils:
        raise ValueError(
            f"Number of x-coordinates ({len(args.centersx)}) must match "
            f"number of airfoils ({args.number_airfoils})"
        )
    if len(args.centersy) != args.number_airfoils:
        raise ValueError(
            f"Number of y-coordinates ({len(args.centersy)}) must match "
            f"number of airfoils ({args.number_airfoils})"
        )

    # define geometry
    print(f"\nCreating {args.number_airfoils} airfoil(s)...")
    geometry = Airfoil(
        parameters_dict=meta_dict,
        airfoil_points_list=[args.airfoil_type] * args.number_airfoils,
        chords=args.chords,
        thicknesses=args.thicknesses,
        angles=args.angles,
        centers_x=args.centersx,
        centers_y=args.centersy,
        num_airfoils=args.number_airfoils,
        dim=args.dim,
        path=args.path,
    )
    geometry.auto_mesh_options()
    if meta_dict["cfd_parameters"].get("mesh_adapt", False):
        geometry.apply_box2params()

    # create domain
    print(f"\nCreating {args.dim}D domain...")
    _ = geometry.create_domain(save_mesh=True, dim_mesh=args.dim)

    # create object
    print(f"\nCreating {args.dim}D object...")
    _ = geometry.create_object(force_model="", save_mesh=True, dim_mesh=args.dim)

    # create individual objects
    all_airfoil_dict = geometry.create_each_object(save_mesh=True)

    # Finish
    print("\nAll models used during GMSH OCC instance:")
    print(gmsh.model.list())
    print("\nClosing GMSH instance!")
    geometry.finalize()

    # Convert and cleanup
    convert_gmsh_to_mtc(
        input=os.path.join(args.path, "object.msh"),
        output=os.path.join(args.path, "object.t"),
        verbose=False,
    )
    convert_gmsh_to_mtc(
        input=os.path.join(args.path, "domain.msh"),
        output=os.path.join(args.path, "domain.t"),
        verbose=False,
    )

    for airfoil in all_airfoil_dict:
        convert_gmsh_to_mtc(
            input=os.path.join(args.path, airfoil["model"] + ".msh"),
            output=os.path.join(args.path, airfoil["model"] + ".t"),
            verbose=False,
        )

    # Move output files
    move_meshes(
        output_directory=args.output_dir, extensions=[".t"], source_directory=args.path
    )
    move_meshes(
        output_directory=args.output_dir + "_GMSH",
        extensions=[".msh", ".geo_unrolled", ".vtk"],
        source_directory=args.path,
    )

    if args.clean:
        print("\nCleaning up...")
        remove_directory(args.output_dir + "_GMSH")

    print("\nDone!")


if __name__ == "__main__":
    main_airfoil(args=None)
