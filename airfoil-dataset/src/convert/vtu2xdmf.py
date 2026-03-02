# requires h5py
from __future__ import annotations
import meshio
import numpy as np
import os
from tqdm import tqdm
import argparse
from bisect import bisect_left
from utils import list_of_strings, read_vtu_fields, read_vtu_mesh


def compress_h5(
    filelist: list[str],
    outfile: str,
    P1_to_exclude: list[str],
    P0_to_exclude: list[str],
    P0C_to_exclude: list[str],
    nb_proc: int = 8,
    double_precision: bool = False,
    add: bool = False,
    overwrite: str | None = None,
    delete: bool = False,
    verbose: bool = True,
):
    precision = np.float64 if double_precision else np.float32

    # Making sure every path is absolute
    if not isinstance(filelist, list):
        filelist = [filelist]
    # sort by index of files
    filelist.sort()
    filelist = [os.path.relpath(f) for f in filelist]
    if outfile is None:
        outfile = os.path.commonprefix(filelist) + ".xdmf"
    else:
        if outfile.split(".")[-1] != "xdmf":
            outfile += ".xdmf"
        outfile = os.path.relpath(outfile)

    prev_times = []
    if add:
        if os.path.exists(outfile):
            with meshio.xdmf.TimeSeriesReader(outfile) as reader:
                for grid in reader.collection:
                    time = grid.find("Time")
                    prev_times.append(float(time.get("Value")))
            prev_times.sort()
        else:
            if verbose:
                print("File doesn't exist, creating it.")
    else:
        if os.path.exists(outfile):
            if overwrite is None:
                invalid = True
                while invalid:
                    yno = input("File exists, overwrite it ? y/n")
                    if yno == "n" or yno == "N" or yno == "no" or yno == "No":
                        invalid = False
                        print("Not overwriting file.")
                        return
                    elif yno == "y" or yno == "Y" or yno == "yes" or yno == "Yes":
                        invalid = False
                        print("Overwriting file.")
                        pass
                    else:
                        print("Invalid input. Valid inputs are y/n.")
            elif overwrite == "n":
                print("File exists, not overwriting.")
                return
            elif overwrite == "y":
                print("Overwriting file.")
                pass
            else:
                raise ValueError("Invalid value for overwrite: " + str(overwrite))

    # Read the mesh and field names
    p, c = read_vtu_mesh(filelist[0])
    P0, P1, P0C = read_vtu_fields(filelist[0])
    P0_names, P1_names, P0C_names = (
        [k for k in list(P0.keys()) if k not in P0_to_exclude],
        [k for k in list(P1.keys()) if k not in P1_to_exclude],
        [k for k in list(P0C.keys()) if k not in P0C_to_exclude],
    )
    if verbose:
        print(
            "Saved P1 fields are: ",
            (
                [x for i, x in enumerate(P1_names) if i < 10] + ["..."]
                if len(P1_names) > 10
                else P1_names
            ),
            "\nSaved P0 fields are: ",
            (
                [x for i, x in enumerate(P0_names) if i < 10] + ["..."]
                if len(P0_names) > 10
                else P0_names
            ),
            "\nSaved P0C fields are: ",
            (
                [x for i, x in enumerate(P0C_names) if i < 10] + ["..."]
                if len(P0C_names) > 10
                else P0C_names
            ),
        )

    with meshio.xdmf.TimeSeriesWriter(outfile, add=add) as writer:
        if not add:
            writer.write_points_cells(p, c)
        t = 0
        if verbose:
            pbar = tqdm(total=1 + len(filelist) // nb_proc)
        for file in filelist:
            # Read the fields from the current file
            P0, P1, P0C = read_vtu_fields(file)

            point_data = {}
            cell_data = {}
            user_data = {}

            # Process P0 fields
            for n in P0_names:
                cell_data[n] = np.array(P0[n]).astype(precision)

            # Process P1 fields
            for n in P1_names:
                point_data[n] = P1[n].astype(precision)

            # Process P0C fields
            for n in P0C_names:
                user_data[n] = P0C[n]

            # Determine the time value
            try:
                time = float(user_data["Temps"])
            except KeyError:
                try:
                    time = float(user_data["Time"])
                except KeyError:
                    time = t

            # Check if the time already exists in the previous times
            if prev_times:
                t_i = bisect_left(prev_times, time)
                if t_i != len(prev_times) and prev_times[t_i] == time:
                    continue

            # Write the data to the XDMF file
            writer.write_data(
                time,
                point_data=point_data,
                cell_data=cell_data,
                user_data=user_data,
            )

            t += 1
            if verbose:
                pbar.update(1)

    if delete:
        if verbose:
            print("Removing vtu files")
        for f in filelist:
            os.remove(f)


def decompress_h5(file_name, outfile, index=None, binary=False):
    # read
    if outfile:
        outfile_prefix = outfile
    else:
        outfile_prefix = file_name.split(".")[0]
    with meshio.xdmf.TimeSeriesReader(file_name) as reader:
        p, c = reader.read_points_cells()
        if index is not None:
            _, P1, P0, P0C = reader.read_data(index)
            mesh = meshio.Mesh(p, c, point_data=P1, cell_data=P0, user_data=P0C)
            mesh.write(outfile_prefix + ".vtu", binary=binary)
        else:
            if reader.num_steps > 1:
                _, _, _, P0C0 = reader.read_data(0)
                _, _, _, P0C1 = reader.read_data(1)
                try:
                    ctime0 = int(P0C0["CompteurTemps"])
                    ctime1 = int(P0C1["CompteurTemps"])
                except KeyError:
                    ctime0 = 0
                    ctime1 = 1
                deltactime = ctime1 - ctime0
            else:
                ctime0 = 0
                ctime1 = 1
                deltactime = 1
            for t in range(reader.num_steps):
                _, P1, P0, P0C = reader.read_data(t)
                mesh = meshio.Mesh(p, c, point_data=P1, cell_data=P0, user_data=P0C)
                try:
                    ctime = int(P0C["CompteurTemps"])
                except KeyError:
                    ctime = t
                ctime = str(ctime)
                ctime = ctime.zfill(
                    int(np.log10(ctime0 + reader.num_steps * deltactime)) + 1
                )
                mesh.write(outfile_prefix + "_" + ctime + ".vtu", binary=binary)


def decompress_h5_last(file_name, outfile, number=1, binary=False):
    # read
    if outfile:
        outfile_prefix = outfile
    else:
        outfile_prefix = file_name.split(".")[0]
    with meshio.xdmf.TimeSeriesReader(file_name) as reader:
        p, c = reader.read_points_cells()
        if reader.num_steps > 1:
            _, _, _, P0C0 = reader.read_data(0)
            _, _, _, P0C1 = reader.read_data(1)
            try:
                ctime0 = int(P0C0["CompteurTemps"])
                ctime1 = int(P0C1["CompteurTemps"])
            except KeyError:
                ctime0 = 0
                ctime1 = 1
            deltactime = ctime1 - ctime0
        else:
            ctime0 = 0
            ctime1 = 1
            deltactime = 1
        for i in range(number):
            t = reader.num_steps - 1 - i
            _, P1, P0, P0C = reader.read_data(t)
            mesh = meshio.Mesh(p, c, point_data=P1, cell_data=P0, user_data=P0C)
            try:
                ctime = int(P0C["CompteurTemps"])
            except KeyError:
                ctime = t
            ctime = str(ctime)
            ctime = ctime.zfill(
                int(np.log10(ctime0 + reader.num_steps * deltactime)) + 1
            )
            mesh.write(outfile_prefix + "_" + ctime + ".vtu", binary=binary)


def del_fields_h5(file_name: str, fields):
    with meshio.xdmf.TimeSeriesModifier(file_name) as modifier:
        modifier.del_fields(fields)


def clear_fields(file_name: str) -> None:
    """
    Clears the fields in the given mesh file.

    Parameters:
    filename (str): The path to the mesh to be processed.

    Returns:
    None
    """
    mesh = meshio.read(file_name)

    filtered_cells = {
        key: value
        for key, value in mesh.cells_dict.items()
        if key not in ["vertex", "line"]
    }

    new_mesh = meshio.Mesh(points=mesh.points, cells=filtered_cells)

    meshio.write(file_name, new_mesh)
    return


def _get_parser_compress(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "infiles",
        nargs="*",
        type=str,
        help="Input files to read. If recursive, only prefix",
    )
    parser.add_argument(
        "--outfile",
        nargs="?",
        type=str,
        help="Output file to write with .xdmf extension",
        default=None,
    )
    parser.add_argument(
        "--P1_excluded",
        type=list_of_strings,
        help="Give a list of P1 field names you do not want to include in the compressed h5 file separated by commas",
    )
    parser.add_argument(
        "--P0_excluded",
        type=list_of_strings,
        help="Give a list of P0 field names you do not want to include in the compressed h5 file separated by commas",
    )
    parser.add_argument(
        "--P0C_excluded",
        type=list_of_strings,
        help="Give a list of P0C field names you do not want to include in the compressed h5 file separated by commas",
    )
    parser.add_argument(
        "-n",
        "--nbprocs",
        type=int,
        default=1,
        help="Number of procs to use",
    )
    parser.add_argument(
        "-ow",
        "--overwrite",
        type=str,
        default=None,
        help=(
            "Value 'y' : automatically says yes to overwriting files. "
            "Value 'n' : automatically says no to overwriting files"
        ),
    )
    parser.add_argument(
        "--delete",
        action="store_const",
        const=True,
        default=False,
        help="Automatically deletes vtu files after compression (be careful)",
    )
    parser.add_argument(
        "--double_precision",
        action="store_const",
        const=True,
        default=False,
        help="Increase precision to double",
    )
    parser.add_argument(
        "--add",
        action="store_const",
        const=True,
        default=False,
        help=(
            "Add files to an existing xdmf/h5 pair. You can select already converted files, it won't overwrite them. "
            "Although it will read them which may take time."
        ),
    )
    return parser


def _get_parser_decompress(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "infile",
        nargs="?",
        type=str,
        default=None,
        help="Input files to decompress",
    )
    parser.add_argument(
        "--outfile",
        nargs="?",
        type=str,
        help="Output file prefix",
        default=None,
    )
    parser.add_argument(
        "--last",
        nargs="?",
        type=int,
        help="Number of elements to decompress from the end (incompatible with --index)",
        default=None,
    )
    parser.add_argument(
        "--index",
        nargs="?",
        type=int,
        help="Specific index to decompress",
        default=None,
    )
    parser.add_argument(
        "-b",
        "--binary",
        action="store_const",
        const=True,
        default=False,
        help="Output binary files",
    )
    return parser


def main_vtu2h5(args=None):
    if args is None:
        parser = _get_parser_compress("Compress vtu files to h5 format.")
        args = parser.parse_args()
    if args.P1_excluded is None:
        args.P1_excluded = []
    if args.P0_excluded is None:
        args.P0_excluded = []
    if args.P0C_excluded is None:
        args.P0C_excluded = []
    dict_of_files = {os.getcwd(): args.infiles}
    for folder, infiles in dict_of_files.items():
        print(
            "Now compressing "
            + os.path.commonprefix(infiles)
            + "*.vtu in folder "
            + folder
        )
        infiles = [os.path.join(folder, f) for f in infiles]
        try:
            compress_h5(
                filelist=infiles,
                outfile=args.outfile,
                P1_to_exclude=args.P1_excluded,
                P0_to_exclude=args.P0_excluded,
                P0C_to_exclude=args.P0C_excluded,
                nb_proc=args.nbprocs,
                double_precision=args.double_precision,
                add=args.add,
                overwrite=args.overwrite,
                delete=args.delete,
                verbose=True,
            )
        except Exception as e:
            print(e)
    return


def main_h52vtu(args=None):
    if args is None:
        parser = _get_parser_decompress("Decompress a h5 file into multiple vtus.")
        args = parser.parse_args()
    if args.last:
        decompress_h5_last(args.infile, args.outfile, args.last, args.binary)
    else:
        decompress_h5(args.infile, args.outfile, args.index, args.binary)
    return


if __name__ == "__main__":
    main_vtu2h5()
