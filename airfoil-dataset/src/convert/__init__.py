import argparse
from utils import list_of_strings
from convert.mesh import main_mesh
from convert.vtu2xdmf import main_vtu2h5
from convert.vtu2xdmf import main_h52vtu


def subparser_mesh(subparsers: argparse._SubParsersAction):
    """
    Add the mesh subparser to the main parser.
    """
    mesh_parser = subparsers.add_parser(
        "mesh",
        help="Convert a gmsh mesh file to mtc (cimlib) format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mesh_parser.add_argument(
        "input_file",
        type=str,
        help="Path to input gmsh mesh file to convert.",
    )


def subparser_vtu2h5(subparsers: argparse._SubParsersAction):
    """
    Add the vtu2xdmf subparser to the main parser.
    """
    vtu2h5_parser = subparsers.add_parser(
        "vtu2h5",
        help="Convert a vtu mesh file to xdmf/h5 format (compresses vtus into a single xdmf/h5 file pair).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    vtu2h5_parser.add_argument(
        "infiles",
        nargs="*",
        type=str,
        help="Input files to read. If recursive, only prefix",
    )
    vtu2h5_parser.add_argument(
        "--outfile",
        nargs="?",
        type=str,
        help="Output file to write with .xdmf extension",
        default=None,
    )
    vtu2h5_parser.add_argument(
        "--P1_excluded",
        type=list_of_strings,
        help="Give a list of P1 field names you do not want to include in the compressed h5 file separated by commas",
    )
    vtu2h5_parser.add_argument(
        "--P0_excluded",
        type=list_of_strings,
        help="Give a list of P0 field names you do not want to include in the compressed h5 file separated by commas",
    )
    vtu2h5_parser.add_argument(
        "--P0C_excluded",
        type=list_of_strings,
        help="Give a list of P0C field names you do not want to include in the compressed h5 file separated by commas",
    )
    vtu2h5_parser.add_argument(
        "-n",
        "--nbprocs",
        type=int,
        default=1,
        help="Number of procs to use, currently not implemented",
    )
    vtu2h5_parser.add_argument(
        "-ow",
        "--overwrite",
        type=str,
        default=None,
        help=(
            "Value 'y' : automatically says yes to overwriting files. "
            "Value 'n' : automatically says no to overwriting files"
        ),
    )
    vtu2h5_parser.add_argument(
        "--delete",
        action="store_const",
        const=True,
        default=False,
        help="Automatically deletes vtu files after compression (be careful)",
    )
    vtu2h5_parser.add_argument(
        "--double_precision",
        action="store_const",
        const=True,
        default=False,
        help="Increase precision to double",
    )
    vtu2h5_parser.add_argument(
        "--add",
        action="store_const",
        const=True,
        default=False,
        help=(
            "Add files to an existing xdmf/h5 pair. You can select already converted files, it won't overwrite them. "
            "Although it will read them which may take time."
        ),
    )


def subparser_h52vtu(subparsers: argparse._SubParsersAction):
    """
    Add the h52vtu subparser to the main parser.
    """
    h52vtu_parser = subparsers.add_parser(
        "h52vtu",
        help="Convert a xdmf/h5 file pair to vtu format (decompresses the xdmf/h5).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    h52vtu_parser.add_argument(
        "infile",
        nargs="?",
        type=str,
        default=None,
        help="Input files to decompress",
    )
    h52vtu_parser.add_argument(
        "--outfile",
        nargs="?",
        type=str,
        help="Output file prefix",
        default=None,
    )
    h52vtu_parser.add_argument(
        "--last",
        nargs="?",
        type=int,
        help="Number of elements to decompress from the end (incompatible with --index)",
        default=None,
    )
    h52vtu_parser.add_argument(
        "--index",
        nargs="?",
        type=int,
        help="Specific index to decompress",
        default=None,
    )
    h52vtu_parser.add_argument(
        "-b",
        "--binary",
        action="store_const",
        const=True,
        default=False,
        help="Output binary files",
    )


def _parser():
    """
    Main parser for the convert module.
    """
    parser = argparse.ArgumentParser(
        description="Convert files between different formats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # Add subparsers
    subparser_mesh(subparsers)
    subparser_vtu2h5(subparsers)
    subparser_h52vtu(subparsers)

    args = parser.parse_args()

    return args


def main_convert():
    """
    Main function for the convert module.
    """
    args = _parser()

    if args.command == "mesh":
        main_mesh(args=args)
    elif args.command == "vtu2h5":
        main_vtu2h5(args=args)
    elif args.command == "h52vtu":
        main_h52vtu(args)
    else:
        print("Unknown command. Use -h for help.")


if __name__ == "__main__":
    main_convert()
