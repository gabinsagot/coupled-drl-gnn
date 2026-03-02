import os
import glob
import shutil
import argparse
import pandas as pd
import numpy as np
from convert.vtu2xdmf import compress_h5
from utils import get_simu_name


def results_to_xdmf(
    simu_dir: str,
    simu_name: str,
    save_path: str,
    dim: int = 2,
    vtu_start: int = 0,
    dt: float = 0.2,
) -> None:
    """compress simulation vtus in Resultats/2d* folder to xdmf/h5.

    Args:
        simu_dir (str): path to the simulation directory.
        simu_name (str): name of the simulation case.
        save_path (str): path to save the compressed xdmf file.
        dim (int): dimension of the simulation (2 or 3).
        vtu_start (int): start time in seconds for vtus to compress.
        dt (float): time step of the simulation.
    """
    # init
    vtu_folders = glob.glob(os.path.join(simu_dir, "Resultats", f"{dim}d*"))
    for vtu_folder in vtu_folders:
        folder_name = os.path.basename(vtu_folder).split(f"{dim}d")[-1]
        vtu_files = glob.glob(os.path.join(vtu_folder, "*.vtu"))
        vtu_start_idx = vtu_start / dt
        vtus_to_compress = []
        for vtu in vtu_files:
            vtu_idx = int(vtu.split("_")[-1].split(".")[0])
            if vtu_idx >= vtu_start_idx:
                vtus_to_compress.append(vtu)

        # compress
        xdmf_filename = os.path.join(save_path, f"{folder_name}{simu_name}.xdmf")
        try:
            compress_h5(
                filelist=vtus_to_compress,
                outfile=xdmf_filename,
                P1_to_exclude=[],
                P0_to_exclude=[],
                P0C_to_exclude=[],
                nb_proc=1,  # number of vtus handled simultaneously (unsupported)
                double_precision=False,
                add=False,
                overwrite=None,
                delete=False,
                verbose=False,
            )
        except Exception as e:
            raise RuntimeError(f"Error compressing {folder_name} vtus to xdmf: {e}")

        # remove vtus and pvds
        shutil.rmtree(vtu_folder, ignore_errors=True)


def results_to_csv(simu_dir: str, simu_name: str, save_path: str, dim: int = 2) -> None:
    """Read all the simu/Resultats/.txt files and join them into a single dataframe,
    where the columns are the outputs: Object,Temps,Fx,Fy,Fz,Mx,My,Mz. Save as csv file.
    Selecting the subdataframe according to object=='some_object' should give the simu results for that object.
    Each object does not necessarily have data for all functions (Fx,Fy,Fz,Mx,My,Mz).

    Note that the data is averaged over repeated timesteps to account for reprise of a simulation.

    Note that data in 2D is supposed to not include torque data (torque available only in 3D)

    Args:
        simu_dir (str): path to the simulation directory.
        simu_name (str): name of the simulation case.
        save_path (str): path to save the resulting csv file.
        dim (int): dimension of the simulation (2 or 3).
    """
    # init
    data_folder = os.path.join(simu_dir, "Resultats")
    data_files = glob.glob(os.path.join(data_folder, "*.txt"))

    if data_files:
        df = pd.DataFrame()
        dataframes = []
        objects = set(
            os.path.basename(f)
            .split("Efforts")[-1]
            .split("Torque")[-1]
            .split(".txt")[0]
            for f in data_files
            if "Efforts" in f or "Torque" in f
        )

        # Process each object
        for obj in sorted(objects):
            # init
            df_force = pd.DataFrame()
            df_torque = pd.DataFrame()

            # force data
            force_file = os.path.join(data_folder, f"Efforts{obj}.txt")
            if os.path.exists(force_file):
                df_force = pd.read_csv(force_file, sep="\t")
                if dim == 2:
                    df_force = df_force[["Temps", f"CxS{obj}", f"CyS{obj}"]]
                else:
                    df_force = df_force[
                        ["Temps", f"CxS{obj}", f"CyS{obj}", f"CzS{obj}"]
                    ]

            # torque data
            torque_file = os.path.join(data_folder, f"Torque{obj}.txt")
            if os.path.exists(torque_file):
                df_torque = pd.read_csv(torque_file, sep="\t")
                # check alternative naming
                if any(col.startswith("TorqueSurf") for col in df_torque.columns):
                    df_torque.rename(
                        columns={
                            "TorqueSurfX" + obj: "CmxS" + obj,
                            "TorqueSurfY" + obj: "CmyS" + obj,
                            "TorqueSurfZ" + obj: "CmzS" + obj,
                        },
                        inplace=True,
                    )
                    df_torque = df_torque[
                        ["Temps", f"CmxS{obj}", f"CmyS{obj}", f"CmzS{obj}"]
                    ]
                else:
                    df_torque = df_torque[
                        ["Temps", f"CmxS{obj}", f"CmyS{obj}", f"CmzS{obj}"]
                    ]

            # merge force and torque on 'Temps'
            if not df_force.empty and not df_torque.empty:
                df = pd.merge(df_force, df_torque, on=["Temps"], how="outer")
            elif not df_force.empty:
                df = df_force.copy()
                df["CmxS" + obj] = np.nan
                df["CmyS" + obj] = np.nan
                df["CmzS" + obj] = np.nan
                if dim == 2:
                    df["CzS" + obj] = np.nan
            elif not df_torque.empty:
                df = df_torque.copy()
                df["CxS" + obj] = np.nan
                df["CyS" + obj] = np.nan
                df["CzS" + obj] = np.nan

            if not df.empty:
                # handle reprise (repeated timesteps)
                df = df.groupby("Temps", as_index=False).mean()

                # add object column
                df["Object"] = obj

                # reformat
                df.rename(
                    columns={
                        f"CxS{obj}": "Fx",
                        f"CyS{obj}": "Fy",
                        f"CzS{obj}": "Fz",
                        f"CmxS{obj}": "Mx",
                        f"CmyS{obj}": "My",
                        f"CmzS{obj}": "Mz",
                    },
                    inplace=True,
                )

                dataframes.append(df)
            else:
                print(f"\t\tNo data found for object {obj}")

        # concat and reorder
        total_df = pd.concat(dataframes, ignore_index=True)
        total_df.sort_values(by=["Object", "Temps"], inplace=True)

        # save
        outputname = f"{simu_name}_data.csv"
        output_csv = os.path.join(save_path, outputname)
        total_df.to_csv(output_csv, index=False)
    else:
        print(f"\t\tNo data files found in {data_folder}")


def _parser():
    parser = argparse.ArgumentParser(description="Process CFD simulation results.")
    parser.add_argument("simu_dir", type=str, help="Path to the simulation directory.")
    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        default="./",
        help="Path to save the processed results.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=2,
        choices=[2, 3],
        help="Dimension of the simulation (2 or 3).",
    )
    parser.add_argument(
        "--vtu_start",
        type=int,
        default=0,
        help="Time (in seconds) of simulation at which vtus will compressed, before this time they will be discarded.",
    )
    parser.add_argument(
        "--dt", type=float, default=0.2, help="Timestep of the simulation in seconds."
    )
    return parser.parse_args()


def main_process_simulation(args: argparse.Namespace | None = None):
    """Main function to process CFD simulation results."""
    if args is None:
        args = _parser()

    simu_dir = args.simu_dir
    save_path = args.save_path
    dim = args.dim
    vtu_start = args.vtu_start
    dt = args.dt

    simu_name = get_simu_name(simu_dir)

    # compress to xdmf and save signals to csv
    print(f"Compressing {simu_dir} vtus to xdmf in {save_path} ...")
    results_to_xdmf(
        simu_dir=simu_dir,
        simu_name=simu_name,
        save_path=save_path,
        dim=dim,
        vtu_start=vtu_start,
        dt=dt,
    )
    print("Done.")
    print(f"Saving {simu_dir} signals (forces, etc.) to csv in {save_path} ...")
    results_to_csv(
        simu_dir=simu_dir,
        simu_name=simu_name,
        save_path=save_path,
        dim=dim,
    )
    print("Done.")


if __name__ == "__main__":
    main_process_simulation(args=None)
