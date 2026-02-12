import os
import shutil
import re
from typing import List, Tuple

import meshio
import numpy as np
from tqdm import tqdm


def msh_to_meshes(
    msh_file_path: str, time: float = 0.0
) -> Tuple[List[meshio.Mesh], List[float]]:
    """
    Returns a meshio mesh object from a gmsh msh file and a timestep value as lists.
    Does not handle cell data.
    """
    mesh = meshio.read(msh_file_path)
    # Filter to keep only triangle and tetra cell types
    filtered_cells = [cell for cell in mesh.cells if cell.type in ["triangle", "tetra"]]
    # TODO: handle cell data
    # Drop gmsh point data fields
    mesh.point_data = {
        k: v for k, v in mesh.point_data.items() if "gmsh" not in k.lower()
    }
    mesh = meshio.Mesh(
        mesh.points,
        filtered_cells,
        point_data=mesh.point_data,
        cell_data=None,
    )
    meshes = [mesh]
    times = [time]
    return meshes, times


def vtu_to_meshes(
    vtu_file_path: str, time: float = 0.0
) -> Tuple[List[meshio.Mesh], List[float]]:
    """
    Returns a meshio mesh object from a vtu/vtk file and a timestep value as lists.
    Does not handle cell data.
    If file is in ASCII, will remove UserData (cimlib) if present, overwrites original file.
    """
    # adapt to cimlib vtu format: remove 'UserData' tag
    try:
        with open(vtu_file_path, "rb") as f:
            header = f.read(1024)
            # check binary
            if b"\x00" in header:
                pass
            else:
                # probably ascii, can use encoding='utf-8'
                with open(vtu_file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # check ascii
                    if content.startswith("<"):
                        # Remove UserData tags if present
                        if "<UserData" in content:
                            content = re.sub(
                                r"<UserData.*?</UserData>", "", content, flags=re.DOTALL
                            )
                            temp_file = vtu_file_path + ".tmp"
                            with open(temp_file, "w", encoding="utf-8") as temp_f:
                                temp_f.write(content)
                            # Overwrite original file
                            shutil.move(temp_file, vtu_file_path)
    except Exception as e:
        print(f"Error processing VTU file {vtu_file_path}: {e}")
    mesh = meshio.read(vtu_file_path)
    # Filter to keep only triangle and tetra cell types
    filtered_cells = [cell for cell in mesh.cells if cell.type in ["triangle", "tetra"]]
    # TODO: handle cell data
    mesh = meshio.Mesh(
        mesh.points,
        filtered_cells,
        point_data=mesh.point_data,
        cell_data=None,
    )
    meshes = [mesh]
    times = [time]
    return meshes, times


def xdmf_to_meshes(
    xdmf_file_path: str, verbose: bool = False
) -> Tuple[List[meshio.Mesh], List[float]]:
    """
    Returns meshio mesh objects for every timestep in an XDMF archive file,
    and returns list of timesteps.
    """
    reader = meshio.xdmf.TimeSeriesReader(xdmf_file_path)
    points, cells = reader.read_points_cells()
    meshes = []
    times = []
    for i in tqdm(
        range(reader.num_steps),
        desc="Extracting meshes from XDMF file",
        disable=not verbose,
    ):
        try:
            time, point_data, cell_data, _ = reader.read_data(i)
        except ValueError:
            try:
                time, point_data, cell_data = reader.read_data(i)
            except Exception as e:
                print(f"Error reading time/point/cell(/user) data: {e}")
                raise
        mesh = meshio.Mesh(points, cells, point_data=point_data, cell_data=cell_data)
        meshes.append(mesh)
        times.append(time)
    return meshes, times


def meshes_to_xdmf(
    filename: str,
    meshes: List[meshio.Mesh],
    timestep: float | List[float] = 1.0,
    verbose: bool = False,
    drop_firststep=False,
) -> None:
    """
    Writes a time series of meshes (same points and cells) into XDMF/HDF5 format.
    The function will write two files: 'filename.xdmf' and 'filename.h5'.

    filename: chosen name for the archive files.
    meshes: List of meshes to compress, they need to share their cells and points.
    timestep: Timestep between two frames.
    """
    points = meshes[0].points
    cells = meshes[0].cells

    filename = os.path.splitext(filename)[0]
    h5_filename = f"{filename}.h5"
    xdmf_filename = f"{filename}.xdmf"

    # Open the TimeSeriesWriter for HDF5
    with meshio.xdmf.TimeSeriesWriter(xdmf_filename) as writer:
        # Write the mesh (points and cells) once
        writer.write_points_cells(points, cells)

        # Loop through time steps and write data
        if isinstance(timestep, (int, float)):
            timestep = [i * float(timestep) for i in range(len(meshes))]
        elif len(timestep) != len(meshes):
            raise ValueError(
                f"Length of timestep list ({len(timestep)}) must match the number of meshes ({len(meshes)})."
            )
        for mesh, t in tqdm(
            zip(meshes, timestep),
            desc="Compressing mesh into XDMF files",
            disable=not verbose,
        ):
            point_data = mesh.point_data
            cell_data = mesh.cell_data
            if not (drop_firststep and t == 0.0):
                writer.write_data(t, point_data=point_data, cell_data=cell_data)

    # The H5 archive is systematically created in cwd with the original meshio library
    if os.path.exists(os.path.join(os.getcwd(), os.path.split(h5_filename)[1])):
        shutil.move(
            src=os.path.join(os.getcwd(), os.path.split(h5_filename)[1]),
            dst=h5_filename,
        )
    if verbose:
        print(f"Time series written to {xdmf_filename} and {h5_filename}")


def convert_gmsh_to_mtc(input: str, output: str, verbose: bool = True) -> None:
    """
    Convert a gmsh mesh file to an mtc (.t) mesh file.

    Args:
    input (str): Path to the input gmsh mesh file.
    output (str): Path to the output mtc mesh file.
    verbose (bool): Print progress to stdout.
    """
    if verbose:
        print("Initialisation...\n")

    with open(input) as f:
        f.readline()
        version = f.readline().split()[0]
        if len(version) > 1:
            version = version.split(".")[0]
        if version != "4" and version != "2":
            raise ValueError("This version of gmsh isn't supported")

        flags = {"$Nodes": [], "$EndNodes": [], "$Elements": [], "$EndElements": []}

        connect_3d = []
        connect_2d = []
        connect_1d = []

        if verbose:
            print("Getting position flags...\n")

        t = f.readline()

        while t:
            t = t.strip("\t\n")
            if t.startswith("$"):
                for i in range(len(list(flags.keys()))):
                    if t == list(flags.keys())[i]:
                        flags[t].append(f.tell())
                        break
            t = f.readline()

        if verbose:
            print("Treating connectivities...\n")

        if version == "4":
            for index in range(len(flags["$Elements"])):
                f.seek(flags["$Elements"][index])

                t = f.readline()  # line ignored (nb of elements)
                t = f.readline()

                while t and f.tell() != flags["$EndElements"][index]:
                    t = t.strip("\t\n").split()

                    if len(t) <= 1:
                        break

                    if t[2] != "2" and t[2] != "4":
                        for i in range(int(t[-1])):
                            f.readline()

                    if t[2] == "2":  # triangle
                        for i in range(int(t[-1])):
                            elem = f.readline().strip("\t\n").split()
                            lig = [int(elem[1]), int(elem[2]), int(elem[3])]
                            connect_2d.append(lig)

                    if t[2] == "4":  # tetrahedron
                        for i in range(int(t[-1])):
                            elem = f.readline().strip("\t\n").split()
                            lig = [
                                int(elem[1]),
                                int(elem[2]),
                                int(elem[3]),
                                int(elem[4]),
                            ]
                            connect_3d.append(lig)

                    t = f.readline()

        if version == "2":
            for index in range(len(flags["$Elements"])):
                f.seek(flags["$Elements"][index])

                t = f.readline()  # line ignored (nb of elements)
                t = f.readline()

                while t and f.tell() != flags["$EndElements"][index]:
                    t = t.split()

                    if len(t) <= 1:
                        break

                    if t[1] == "2":  # triangle
                        lig = [int(t[-3]), int(t[-2]), int(t[-1])]
                        connect_2d.append(lig)

                    if t[1] == "4":  # tetrahedron
                        lig = [int(t[-4]), int(t[-3]), int(t[-2]), int(t[-1])]
                        connect_3d.append(lig)

                    t = f.readline()

        # Correction for gmsh numbering
        connect_2d = np.array(connect_2d, dtype=int) - 1
        connect_3d = np.array(connect_3d, dtype=int) - 1

        if verbose:
            print("Verifying nodes and edges...")

        # nodes

        nodes = []

        if version == "4":
            for index in range(len(flags["$Nodes"])):
                f.seek(flags["$Nodes"][index])
                f.readline()  # line ignored (nb of nodes)

                t = f.readline()

                while t and f.tell() != flags["$EndNodes"][index]:
                    t = t.strip("\t\n").split()

                    if len(t) <= 1:
                        break

                    for i in range(int(t[-1])):
                        f.readline()

                    for i in range(int(t[-1])):
                        node = f.readline().strip("\t\n").split()
                        nodes.append([float(node[0]), float(node[1]), float(node[2])])

                    t = f.readline()

        if version == "2":
            for index in range(len(flags["$Nodes"])):
                f.seek(flags["$Nodes"][index])
                f.readline()  # line ignored (nb of nodes)

                t = f.readline()

                while t and f.tell() != flags["$EndNodes"][index]:
                    t = t.strip("\t\n").split()

                    if len(t) <= 1:
                        break

                    nodes.append([float(t[1]), float(t[2]), float(t[3])])

                    t = f.readline()

    nodes = np.array(nodes)

    dim = 3
    if len(connect_3d) == 0:
        if np.all(nodes[:, 0] == nodes[0, 0]):
            dim = 2
            nodes = nodes[:, 1:]
        elif np.all(nodes[:, 1] == nodes[0, 1]):
            dim = 2
            nodes = nodes[:, -1:1]
        elif np.all(nodes[:, 2] == nodes[0, 2]):
            dim = 2
            nodes = nodes[:, :2]
        else:
            dim = 2.5

    # Apparently Cimlib prefers normals looking down in 2D
    # If normals are still wrong after that, there may be foldovers in your mesh
    if dim == 2:
        if verbose:
            print("   - Checking normals")  # Actually only checking the first normal
        normal = np.cross(
            nodes[connect_2d[0][1]] - nodes[connect_2d[0][0]],
            nodes[connect_2d[0][2]] - nodes[connect_2d[0][0]],
        )
        if normal > 0:
            connect_2d = connect_2d[:, [0, 2, 1]]

    if verbose:
        print("   - Detecting edges")

    if dim == 3:
        del connect_2d

        tris1 = connect_3d[:, [0, 2, 1]]  # Order is very important !
        tris2 = connect_3d[:, [0, 1, 3]]
        tris3 = connect_3d[:, [0, 3, 2]]
        tris4 = connect_3d[:, [1, 2, 3]]

        tris = np.concatenate((tris1, tris2, tris3, tris4), axis=0)
        tris_sorted = np.sort(
            tris, axis=1
        )  # creates a copy, may be source of memory error
        tris_sorted, uniq_idx, uniq_cnt = np.unique(
            tris_sorted, axis=0, return_index=True, return_counts=True
        )
        connect_2d = tris[uniq_idx][uniq_cnt == 1]

    if dim == 2:
        lin1 = connect_2d[:, [0, 1]]  # Once again, order is very important !
        lin2 = connect_2d[:, [2, 0]]
        lin3 = connect_2d[:, [1, 2]]

        lin = np.concatenate((lin1, lin2, lin3), axis=0)
        lin_sorted = np.sort(
            lin, axis=1
        )  # creates a copy, may be source of memory error
        lin_sorted, uniq_idx, uniq_cnt = np.unique(
            lin_sorted, axis=0, return_index=True, return_counts=True
        )
        connect_1d = lin[uniq_idx][uniq_cnt == 1]

    if verbose:
        print("   - Detecting unused nodes")

    used_nodes = np.unique(
        np.concatenate((connect_3d.flatten(), connect_2d.flatten()))
    )  # sorted
    bools_keep = np.zeros(len(nodes), dtype=bool)
    bools_keep[used_nodes] = True

    if verbose:
        print("   - Deleting unused nodes and reindexing\n")

    nodes = nodes[bools_keep]
    new_indices = np.cumsum(bools_keep) - 1

    if dim == 3 or dim == 2.5:
        connect_3d = new_indices[connect_3d]
        connect_2d = new_indices[connect_2d]

    if dim == 2:
        connect_2d = new_indices[connect_2d]
        connect_1d = new_indices[connect_1d]

    nb_elems = len(connect_2d) + len(connect_3d)
    if dim == 2:
        nb_elems += len(connect_1d)
        if verbose:
            print("Nb elements 1d : " + str(len(connect_1d)))

    if verbose:
        print("Nb elements 2d : " + str(len(connect_2d)))
        print("Nb elements 3d : " + str(len(connect_3d)))
        print("Dimension : " + str(dim) + "\n")
        print("Writing .t file...")

    # Correction for mtc numbering
    connect_3d += 1
    connect_2d += 1
    if len(connect_1d) > 0:
        connect_1d += 1

    with open(output, "w") as fo:
        lig = (
            str(len(nodes))
            + " "
            + str(dim)
            + " "
            + str(nb_elems)
            + " "
            + str(dim + 1)
            + "\n"
        )
        if dim == 2.5:
            lig = str(len(nodes)) + " 3 " + str(nb_elems) + " 4\n"
        fo.write(lig)

        for node in nodes:
            fo.write("{0:.8g} {1:.8g}".format(node[0], node[1]))
            if dim == 3 or dim == 2.5:
                fo.write(" {0:.8g}".format(node[2]))
            fo.write(" \n")

        for e in connect_3d:
            fo.write(
                str(e[0]) + " " + str(e[1]) + " " + str(e[2]) + " " + str(e[3]) + " \n"
            )

        for e in connect_2d:
            if dim == 3 or dim == 2.5:
                fo.write(str(e[0]) + " " + str(e[1]) + " " + str(e[2]) + " 0 \n")
            else:
                fo.write(str(e[0]) + " " + str(e[1]) + " " + str(e[2]) + " \n")

        if dim == 2:
            for e in connect_1d:
                fo.write(str(e[0]) + " " + str(e[1]) + " 0 \n")

    if verbose:
        print("Done.")
    return


def convert_mtc_to_vtk(input: str, output: str, verbose: bool = True) -> None:
    """
    Converts a .t mtc mesh into a vtu/vtk mesh
    """
    raise (NotImplementedError)
