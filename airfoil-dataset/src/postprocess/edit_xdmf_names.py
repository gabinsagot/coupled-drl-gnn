import os
from tqdm import tqdm
import argparse
import re
import json
from utils import xdmf_to_meshes, meshes_to_xdmf


def check_name(root_dir: str = "./", recursive: bool = False) -> None:
    """
    Check if the h5 filename in the xdmf file content is correct.
    If not, replace the incorrect h5 filename with the correct one.
    Print the number of file pairs that had to be reformatted out of the total number of file pairs.
    """

    xdmf_files_list = []
    reformatted_count = 0
    for dirpath, _, filenames in os.walk(root_dir):
        # Filter for xdmf files
        xdmf_files = [f for f in filenames if f.endswith(".xdmf")]
        xdmf_files_list.extend([os.path.join(dirpath, f) for f in xdmf_files])

        if not recursive:
            break

    for xdmf_path in tqdm(xdmf_files_list, desc="Processing XDMF files"):
        h5_file = os.path.basename(xdmf_path).replace(".xdmf", ".h5")
        h5_path = os.path.join(os.path.dirname(xdmf_path), h5_file)

        if os.path.exists(h5_path):
            # Read the xdmf file content
            with open(xdmf_path, "r") as file:
                xdmf_content = file.read()
            h5_filename_to_write = os.path.basename(h5_path).split("/")[-1]
            # Check if the h5 filename in the xdmf content is correct
            if f">{h5_filename_to_write}:" not in xdmf_content:
                # Replace the incorrect h5 filename with the correct one
                new_content = re.sub(
                    r">([^>]+\.h5):", f">{h5_filename_to_write}:", xdmf_content
                )
                # Write the corrected content back to the xdmf file
                with open(xdmf_path, "w") as file:
                    file.write(new_content)
                reformatted_count += 1

    print(f"Reformatted {reformatted_count} out of {len(xdmf_files_list)} file pairs.")


def rename_fields(
    root_dir: str = "./",
    renaming_dict: dict = {
        "Vitesse": "Velocity",
    },
    recursive: bool = False,
) -> None:
    """
    Rename fields in the xdmf files.
    """
    xdmf_files_list = []
    reformatted_count = 0
    for dirpath, _, filenames in os.walk(root_dir):
        # Filter for xdmf files
        xdmf_files = [f for f in filenames if f.endswith(".xdmf")]
        xdmf_files_list.extend([os.path.join(dirpath, f) for f in xdmf_files])

        if not recursive:
            break

    for xdmf_path in tqdm(
        xdmf_files_list, desc="Processing XDMF files for field renaming"
    ):
        h5_file = os.path.basename(xdmf_path).replace(".xdmf", ".h5")
        h5_path = os.path.join(os.path.dirname(xdmf_path), h5_file)

        if os.path.exists(h5_path):
            # Read the xdmf file content
            with open(xdmf_path, "r") as file:
                xdmf_content = file.read()
            # Check if the h5 filename in the xdmf content is correct
            for old_name, new_name in renaming_dict.items():
                if f'Name="{old_name}"' in xdmf_content:
                    # Replace the incorrect h5 filename with the correct one
                    xdmf_content = re.sub(
                        f'Name="{old_name}"', f'Name="{new_name}"', xdmf_content
                    )
                    reformatted_count += 1
                else:
                    print(f"Field {old_name} not found in {xdmf_path}.")
            # Write the corrected content back to the xdmf file
            with open(xdmf_path, "w") as file:
                file.write(xdmf_content)
    print(
        f"Reformatted {[key for key in renaming_dict.keys()]} fields to "
        f"{[val for val in renaming_dict.values()]} in "
        f"{reformatted_count/len(renaming_dict.keys())}/{len(xdmf_files_list)} xdmf file pairs."
    )


def drop_fields(
    root_dir: str = "./",
    fields_to_drop: list = ["AppartientObject"],
    recursive: bool = False,
) -> None:
    """
    Drop fields in the xdmf files.
    """
    xdmf_files_list = []
    for dirpath, _, filenames in os.walk(root_dir):
        # Filter for xdmf files
        xdmf_files = [f for f in filenames if f.endswith(".xdmf")]
        xdmf_files_list.extend([os.path.join(dirpath, f) for f in xdmf_files])

        if not recursive:
            break

    for xdmf_path in tqdm(
        xdmf_files_list, desc="Processing XDMF files for field dropping"
    ):
        h5_file = os.path.basename(xdmf_path).replace(".xdmf", ".h5")
        h5_path = os.path.join(os.path.dirname(xdmf_path), h5_file)

        if os.path.exists(h5_path):
            meshes, times = xdmf_to_meshes(xdmf_file_path=xdmf_path, verbose=False)
            # check
            fields_to_actually_drop = [
                field
                for field in fields_to_drop
                if field in meshes[0].point_data.keys()
            ]
            fields_ignored = [
                field
                for field in fields_to_drop
                if field not in meshes[0].point_data.keys()
            ]
            if not fields_to_actually_drop:
                continue
            for mesh in meshes:
                mesh.point_data = {
                    k: v
                    for k, v in mesh.point_data.items()
                    if k not in fields_to_actually_drop
                }
            meshes_to_xdmf(
                filename=xdmf_path,
                meshes=meshes,
                timestep=times,
                verbose=False,
                drop_firststep=False,
            )
    try:
        print(
            f"Successfully dropped fields {', '.join(fields_to_actually_drop)} in {len(xdmf_files_list)}"
            f"xdmf file pairs. \n{', '.join(fields_ignored) if fields_ignored else '<none>'} were not found "
            "in the meshes and thus ignored."
        )
    except UnboundLocalError:
        raise FileNotFoundError(
            (
                "Dropping fields works on directories containing xdmf/h5 file pairs, not just xdmf files."
                "\nOR MAYBE: you forgot the -r flag to make it recursive?"
            )
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Makes sure xdmf/h5 file pairs have matching names and pointers "
            "and updates them if necessary, can do this recursively too. "
            "Also renames fields in the xdmf/h5 pair if desired (old_name -> new_name)."
            "Can also drop fields in the xdmf files."
        )
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default="./",
        help="Root directory of the data (where the xdmf/h5 file pairs are).",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively search for xdmf files in the directory.",
    )
    parser.add_argument(
        "--rename_fields",
        type=str,
        default=None,
        help='JSON string representing the dictionary for renaming fields. Example: \'{"old_name": "new_name"}\'',
    )
    parser.add_argument(
        "--drop_fields",
        type=str,
        default=None,
        help="Comma-separated list of fields to drop in xdmfs. Example: AppartientObject,Pression",
    )
    return parser.parse_args()


def main_edit(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = _parser()
    check_name(root_dir=args.directory, recursive=args.recursive)
    if args.drop_fields is not None:
        drop_fields_list = [field.strip() for field in args.drop_fields.split(",")]
        drop_fields(
            root_dir=args.directory,
            fields_to_drop=drop_fields_list,
            recursive=args.recursive,
        )
    if args.rename_fields is not None:
        rename_dict = json.loads(args.rename_fields)
        rename_fields(
            root_dir=args.directory, renaming_dict=rename_dict, recursive=args.recursive
        )


if __name__ == "__main__":
    main_edit(args=None)
