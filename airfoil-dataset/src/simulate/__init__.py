import argparse
import os
import ast
from geometries import Geometry, Airfoil
from utils import load_json_to_dict, move_meshes
from simulation import Simulation
from convert.mesh import convert_gmsh_to_mtc


def add_base_arguments(parser: argparse.ArgumentParser, case: str = "airfoil"):
    """
    Add base arguments to the parser
    """
    parser.add_argument(
        "-f",
        "--simu_config",
        type=str,
        default=f"./config/{case}.json",
        help=f"Path to the simulation config file (default=./config/{case}.json)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=2,
        help="Which dimension for geoemtry and meshing (2D or 3D), provide int 2 or 3",
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=f"./simu_{case}",
        help=f"Path and name of cfd directory to be simulated (default=./simu_{case})",
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help=(
            "If passed, a slurm script is created for you to submit the job (manually) "
            "with max(32,number_cores) in background. To obtain postprocess, "
            "run same command again but with the --postprocess flag. Otherwise runs in foreground."
        ),
    )
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help=("If passed, only the postprocessing will be run"),
    )


def subparser_airfoil(subparsers: argparse._SubParsersAction):
    """
    Add the airfoil simulation subparser
    """
    airfoil_parser = subparsers.add_parser(
        "airfoil",
        help="Create airfoil object(s) and fluid domain, and simulate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_base_arguments(airfoil_parser, case="airfoil")
    airfoil_parser.add_argument(
        "-naca_code",
        "--airfoil_type",
        type=str,
        default="NACA0010",
        help="NACA airfoil code to be created (e.g., NACA0010)",
    )
    airfoil_parser.add_argument(
        "-c",
        "--chords",
        type=str,
        default="[1.0]",
        help="Chord(s) of each airfoil (in brackets)",
    )
    airfoil_parser.add_argument(
        "-t",
        "--thicknesses",
        type=str,
        default="[1.0]",
        help="Thickness(es) of each airfoil (in brackets)",
    )
    airfoil_parser.add_argument(
        "-aoa",
        "--angles",
        type=str,
        default="[0.0]",
        help="Angle(s) of attack of each airfoil (in brackets)",
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
        help="Number of airfoil(s) to create.",
    )


def _parser():
    """
    Create the main parser for the simulation
    """
    parser = argparse.ArgumentParser(
        description="Simulate different geometries with CIMLIB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="commands",
    )
    subparser_airfoil(subparsers)

    args = parser.parse_args()

    return args


def format_command_line_args(
    args: argparse.Namespace, leave_out: list[str] = None
) -> str:
    """
    Format the command line arguments to be passed to the simu command

    Args:
        args (argparse.Namespace): The arguments to format
        leave_out (list[str], optional): List of arguments to leave out. Defaults to None
    Returns:
        str: The formatted command line arguments
    """
    command_line_args = ""
    if leave_out is None:
        leave_out = []
    for key, value in vars(args).items():
        if key not in leave_out:
            command_line_args += f" --{key} {value}"
    return command_line_args.strip()


def simulate(
    args: None | argparse.Namespace = None,
    geometry_class: Geometry = Airfoil,
    geometry_args: dict = {
        "airfoil_points_list": ["NACA0010"],
        "chords": [1.0],
        "thicknesses": [0.5],
        "angles": [0],
        "centers_x": [0.0],
        "centers_y": [0.0],
        "num_airfoils": 1,
    },
):
    """
    Main function to simulate a single airfoil configuration.
    """
    # Parse arguments
    if args is None:
        args = _parser()

    # meta and sanity checks
    meta_dict = load_json_to_dict(args.simu_config)
    if meta_dict["case"] != args.commands:
        raise ValueError(
            f"The provided config file {args.simu_config} is not for airfoil(s), please provide a valid config file."
        )

    # init simu
    print("Prepping simulation...")
    try:
        simulation = Simulation(
            meta_dict=meta_dict,
            simu_path=os.path.dirname(args.path),
            simu_name=args.path.split("/")[-1],
            save_path=args.path,
            number_cores=meta_dict["cfd_parameters"]["number_cores"],
            multigrid=meta_dict["graph_parameters"]["multigrid"],
        )
        if args.postprocess:
            print("Postprocessing simulation...")
            simulation.results_to_xdmf()
            simulation.results_to_csv()
            simulation.cleanup()
            return
        simulation.prep_simulation()
    except Exception as e:
        print(f"Error in prepping simulation: {e}")
        raise

    # geometry
    print("Creating simulation geometry...")
    try:
        geometry = geometry_class(
            parameters_dict=simulation.meta,
            dim=args.dim,
            path=simulation.simu_dir,
            **geometry_args,
        )
        geometry.auto_mesh_options()
        geometry.apply_box2params()
    except Exception as e:
        print(f"Error in creating initializing geometry: {e}")
        raise
    try:
        _ = geometry.create_domain(save_mesh=True, dim_mesh=2)
        _ = geometry.create_object(force_model="", save_mesh=True, dim_mesh=2)
        all_objects_dict = geometry.create_each_object(save_mesh=True)
        geometry.finalize()
        convert_gmsh_to_mtc(
            input=os.path.join(simulation.simu_dir, "object.msh"),
            output=os.path.join(simulation.simu_dir, "object.t"),
            verbose=False,
        )
        convert_gmsh_to_mtc(
            input=os.path.join(simulation.simu_dir, "domain.msh"),
            output=os.path.join(simulation.simu_dir, "domain.t"),
            verbose=False,
        )
        for obj in all_objects_dict:
            convert_gmsh_to_mtc(
                input=os.path.join(args.path, obj["model"] + ".msh"),
                output=os.path.join(args.path, obj["model"] + ".t"),
                verbose=False,
            )
        move_meshes(
            output_directory=os.path.join(simulation.simu_dir, "meshes"),
            extensions=[".t"],
            source_directory=simulation.simu_dir,
        )
        move_meshes(
            output_directory=os.path.join(simulation.simu_dir, "meshes_GMSH"),
            extensions=[".msh", ".geo_unrolled", ".vtk"],
            source_directory=simulation.simu_dir,
        )
    except Exception as e:
        print(f"Error in creating and handling meshes: {e}")
        raise
    try:
        objects_meshdict = {}  # create dict of objectname: rel mesh path
        for mesh_object in all_objects_dict:
            objects_meshdict[mesh_object["model"]] = f"meshes/{mesh_object['model']}.t"
        simulation.generate_geometres_objects(objects_meshdict=objects_meshdict)
        simulation.generate_draglift_objects(objects_list=list(objects_meshdict.keys()))
    except Exception as e:
        print(f"Error in generating objects-related mtc models: {e}")
        raise

    # run
    print("Running simulation...")
    try:
        # NB: requires mpirun -> os.system("module load cimlibxx/master")
        if args.slurm:
            sh_path = os.path.join(simulation.simu_dir, "job.sh")
            ncores = simulation.n_cores
            ntasks = max(32, ncores)
            launcher = simulation.launcher
            driver = simulation.driver
            sh_content = (
                f"#!/bin/bash\n#SBATCH --job-name={simulation.simu_name}\n#SBATCH --output=log.out"
                f"\n#SBATCH --partition=MAIN\n#SBATCH --qos=calcul\n#SBATCH --nodes 1\n#SBATCH --ntasks {ntasks}"
                f"\n#SBATCH --ntasks-per-core 1\n#SBATCH --threads-per-core 1\n\nmodule load cimlibxx/master\n"
            )
            sh_content += (
                f"\n{' '.join(['mpirun', '-n', str(ncores), driver, launcher])}\n"
            )
            sh_content += (
                f"cd ../;\nsimu {args.commands} --postprocess "
                f"{format_command_line_args(args=args, leave_out=['commands', 'postprocess','slurm'])}\n"
            )
            with open(sh_path, "w") as f:
                f.write(sh_content)
            print(
                f"\nJob script written to {sh_path}, exiting now so you can sbatch it."
            )
            print(
                "An auto-generated postprocess command is at the end of the script, comment out if undesired."
            )
            return
        else:
            simulation.run_simulation()
    except Exception as e:
        print("WARNING: make sure mpirun is available!")
        print(f"Error running simulation: {e}")
        raise

    print("Done.")
    return


def main_simulate():
    """
    Main function to create a geometry object and fluid domain using GMSH.
    """

    args = _parser()

    if args.commands == "airfoil":
        geometry_args = {
            "airfoil_points_list": [args.airfoil_type],
            "chords": ast.literal_eval(args.chords),
            "thicknesses": ast.literal_eval(args.thicknesses),
            "angles": ast.literal_eval(args.angles),
            "centers_x": ast.literal_eval(args.centersx),
            "centers_y": ast.literal_eval(args.centersy),
            "num_airfoils": args.number_airfoils,
        }
        geometry_class = Airfoil
    else:
        print("Unknown command. Use -h for help.")
        return
    simulate(args=args, geometry_class=geometry_class, geometry_args=geometry_args)


if __name__ == "__main__":
    main_simulate()
