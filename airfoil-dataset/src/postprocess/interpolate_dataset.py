import os
import meshio
import argparse
from scipy.interpolate import griddata
from tqdm import tqdm
from typing import List, Union
from utils import meshes_to_xdmf, xdmf_to_meshes


def interpolate_over_mesh(
    mesh_file: Union[str, meshio.Mesh],
    field_names: Union[str, List[str]],
    value_meshes: Union[List[str], List[meshio.Mesh]],
    out_path: str = None,
    out_filename: str = None,
    new_field_names: List[str] = None,
    scaling_factor: float = 1,
    methods: Union[str, List[str]] = "nearest",
    fill_value: float = 0.0,
    verbose: bool = True,
) -> List[meshio.Mesh]:
    """
    Gather values from a vtu and interpolate these values on a defined mesh.

    Args:
        mesh_file : File (vtu or vtk) containing the mesh we interpolate values on.
        value_meshes : List of paths to meshes, or meshes with the values we want to extract.
        field_name : Field name for the values we consider from the mesh.
        new_field_name : Optional, name for the interpolated field if changing.
        out_path : Optional, path to the output directory. If None, the meshes aren't saved.
        out_filename : Filename to iterate over for output files.
        scaling_factor : Factor to scale the value mesh coordinates and the values.
        methods : Interpolation method, between 'nearest' and 'linear', for each field.
    Returns:
        List of new interpolated meshes.
    """
    # Handle the possibility that methods is a single method
    if type(methods) is str:
        methods = [methods] * len(field_names)

    if type(mesh_file) is str:
        mesh_file = meshio.read(mesh_file)
    # We open all the meshes at once, might be a problem
    if type(value_meshes[0]) is str:
        value_meshes = [
            meshio.read(value_mesh_file) for value_mesh_file in value_meshes
        ]
    if type(field_names) is str:
        field_names = [field_names]
    if new_field_names is None:
        new_field_names = field_names
    if (out_path and not out_filename) or (not out_path and out_filename):
        raise ValueError(
            "To save the interpolated meshes, need both 'out_path' and 'out_filename' arguments."
        )

    out_meshes = []
    for i, value_mesh in enumerate(
        tqdm(
            value_meshes, ncols=50, desc="Interpolating over mesh", disable=not verbose
        )
    ):
        value_points = value_mesh.points * scaling_factor
        out_mesh = mesh_file.copy()
        out_mesh.point_data = dict()

        for field, new_field, method in zip(field_names, new_field_names, methods):
            value_data = value_mesh.point_data[field] * scaling_factor
            interpolation = griddata(
                points=value_points,
                values=value_data,
                xi=out_mesh.points,
                method=method,
                # fill_value=fill_value,
            )
            out_mesh.point_data[new_field] = interpolation

        if out_path:
            out_mesh.write(
                os.path.join(out_path, f"{out_filename}_{i:05d}.vtu"), binary=False
            )
        out_meshes.append(out_mesh)
    return out_meshes


def _parser():
    """Parse arguments for the interpolation of one dataset onto another mesh resolution."""
    parser = argparse.ArgumentParser(
        description="Interpolate values from one xdmf series to another mesh."
    )
    parser.add_argument(
        "-fine",
        "--fine_directory",
        type=str,
        default=".",
        help="Directory containing the fine mesh files.",
    )
    parser.add_argument(
        "-coarse",
        "--coarse_directory",
        type=str,
        default=".",
        help="Directory containing the coarse mesh files.",
    )
    parser.add_argument(
        "-out",
        "--out_directory",
        type=str,
        default=".",
        help="Directory to save the interpolated meshes.",
    )
    parser.add_argument(
        "--fields",
        type=str,
        nargs="+",
        default=["Vitesse", "Pression", "NodeType", "Reynolds"],
        help="Field names to interpolate.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="nearest",
        help="Interpolation method (choose from 'nearest','linear','cubic'). (default: 'nearest')",
    )
    return parser.parse_args()


def main_interpolate(args: argparse.Namespace | None = None):
    """Main function to interpolate values from one mesh to another."""
    if args is None:
        args = _parser()

    path_fine = args.fine_directory
    path_coarse = args.coarse_directory
    out_dir = args.out_directory
    os.makedirs(out_dir, exist_ok=True)

    fields = args.fields
    method = args.method

    indexes = [
        os.path.splitext(xdmf)[0].split("_")[-1]
        for xdmf in os.listdir(path_coarse)
        if xdmf.endswith(".xdmf")
    ]
    basename_coarse = "_".join(
        os.path.splitext(
            next(xdmf for xdmf in os.listdir(path_coarse) if xdmf.endswith(".xdmf"))
        )[0].split("_")[:-1]
    )
    basename_fine = "_".join(
        os.path.splitext(
            next(xdmf for xdmf in os.listdir(path_fine) if xdmf.endswith(".xdmf"))
        )[0].split("_")[:-1]
    )

    for idx in tqdm(indexes):
        try:

            coarse_mesh = xdmf_to_meshes(
                xdmf_file_path=os.path.join(
                    path_coarse, f"{basename_coarse}_{idx}.xdmf"
                ),
                verbose=False,
            )[0][0]
            fine_meshes, fine_times = xdmf_to_meshes(
                xdmf_file_path=os.path.join(path_fine, f"{basename_fine}_{idx}.xdmf"),
                verbose=False,
            )

            new_meshes = interpolate_over_mesh(
                mesh_file=coarse_mesh,
                field_names=fields,
                value_meshes=fine_meshes,
                methods=method,
                verbose=False,
                fill_value=0.0,
            )
            meshes_to_xdmf(
                filename=os.path.join(out_dir, f"{basename_fine}_interp_{idx}"),
                meshes=new_meshes,
                timestep=fine_times,
                verbose=False,
                drop_firststep=False,
            )
        except Exception as e:
            print(f"Error processing file with index {idx}: {e}", flush=True)


if __name__ == "__main__":
    main_interpolate(args=None)
