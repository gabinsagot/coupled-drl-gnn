import pandas as pd
import numpy as np
import os
import glob
import shutil
import subprocess
from typing import Any, Dict, List

from convert.vtu2xdmf import compress_h5
from convert.mesh import convert_gmsh_to_mtc
from utils import move_meshes, Mtc
from geometries import Airfoil

# Classes and functions to setup, run, and postprocess a simulation


class Simulation:
    """Class to handle simulations. It includes the setup, running,
    and postprocessing of the simulation."""

    def __init__(
        self,
        meta_dict: Dict[str, Any],
        simu_path: str = "simu",
        simu_name: str = "simu0",
        save_path: str = "dataset",
        number_cores: int = 8,
        multigrid: bool = False,
        ignore_slurm: bool = False,
    ):
        """create simulation repo, load data and initilize cfd parameters

        :param meta_dict: (Dict[str, Any]) meta parameters dictionary
        :param simu_path: (str) path to where the simulation directory will be made
        :param simu_name: (str) name of the simulation (for example name of config)
        :param save_path: (str) path to where the simulation postprocess results will be saved (in dataset dir)
        :param number_cores: (int) number of cores to use for the simulation (mpirun). default=8
        :param multigrid: (bool) whether to use multigrid (multiple meshes with interpolation). default=False
        :param ignore_slurm: (bool) if True, always use mpirun instead of srun. default=False
        """
        self.meta: Dict[str, Any] = meta_dict

        self.simu_path: str = simu_path
        self.simu_name: str = simu_name
        self.simu_dir: str = os.path.join(self.simu_path, self.simu_name)
        self.save_path: str = save_path
        self.cfd_template: str = (
            os.path.join("cfd_bank", f"cfd_{self.meta['case']}_multigrid")
            if multigrid
            else os.path.join("cfd_bank", f"cfd_{self.meta['case']}")
        )

        self.dim: int = self.meta["dim"]
        self.driver: str = os.path.abspath(self.meta["driver"])
        self.case: str = self.meta["case"]
        self.multigrid: bool = multigrid

        self.cfd_params: Dict[str, Any] = self.meta["cfd_parameters"]
        self.graph_params: Dict[str, Any] = self.meta["graph_parameters"]
        self.dom_params: Dict[str, Any] = self.meta["domain_parameters"]

        self.n_cores: int = number_cores
        self.launcher: str = "lanceur/Principale.mtc"
        self.timeout: int = 14400  # 4 hours simulation timeout
        # Check if SLURM is available by checking for sinfo command
        if ignore_slurm:
            self.has_slurm: bool = False
        else:
            try:
                subprocess.run(
                    ["sinfo", "--version"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.has_slurm: bool = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.has_slurm: bool = False
        # TODO: integrate multigrid in single launcher for seamless simulation
        if multigrid:
            self.launcher_grid: str = "lanceur/PrincipaleGrid.mtc"
            if not os.path.exists(os.path.join(self.cfd_template, self.launcher_grid)):
                raise FileNotFoundError(
                    f"Additional launcher file {self.launcher_grid} must be included in cfd for coarse grid meshing."
                )
        self.IHM: str = "IHM.mtc"

        self.cfd_params["TempsFin"] = (
            self.graph_params["vtu_start_in_seconds"]
            + self.graph_params["how_many_vtu"]
            * self.cfd_params["PasDeTemps"]
            * self.cfd_params["FrequenceStockage_vtk"]
        )

    def prep_simulation(self):
        """Prepare the simulation"""
        os.makedirs(self.simu_dir, exist_ok=True)
        os.makedirs(self.save_path, exist_ok=True)
        shutil.copytree(self.cfd_template, self.simu_dir, dirs_exist_ok=True)
        self.apply_IHM()
        self.apply_domain_size()
        self.toggle_feature_outputs()
        if not self.cfd_params["mesh_adapt"]:
            self.disable_mesh_adaptation()
        if not self.cfd_params["calc_force"]:
            self.disable_force_computation()

    def apply_IHM(self):
        """Apply the cfd parameters to IHM.mtc file. Names in cfd_parameters
        of meta json should match the names of the parameters in IHM.mtc file."""
        IHM_path = os.path.join(self.simu_dir, self.IHM)
        # checks + formatting
        if self.cfd_params["variable_inlet"]:
            self.cfd_params["VxIn"] = 1.0
        boxes_values = {
            "BLMHbox1": self.cfd_params["Hbox123"][0],
            "BLMHbox2": self.cfd_params["Hbox123"][1],
            "BLMHbox3": self.cfd_params["Hbox123"][2],
        }
        freq_vtk = {"FreqVtkCoarse": self.cfd_params["FrequenceStockage_vtk"]}
        ihm_params = {**self.cfd_params, **boxes_values, **freq_vtk}

        # edit ihm
        ihm_mtc = Mtc(path=IHM_path)
        ihm_content = ihm_mtc.modif_target(
            mtc_content=ihm_mtc.content,
            raise_not_found=False,
            **ihm_params,
        )
        ihm_mtc.write(content=ihm_content, overwrite=True)

    def apply_domain_size(self):
        """Apply the domain size to the GeometresE.mtc file for correct boundary locations
        and flags used in the boundary conditions definitions."""
        GeometresE_path = os.path.join(self.simu_dir, "Geometrie/GeometresE.mtc")
        try:
            with open(GeometresE_path, "r+", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"\nError: File {GeometresE_path} not found.")
        # run through lines and edit origins of the domain boundaries
        bounds = {
            "WallIn": {"Origine": [self.dom_params["origin_x"], 0], "Normale": [1, 0]},
            "WallOut": {
                "Origine": [self.dom_params["origin_x"] + self.dom_params["dx"], 0],
                "Normale": [-1, 0],
            },
            "WallTop": {
                "Origine": [0, self.dom_params["origin_y"] + self.dom_params["dy"]],
                "Normale": [0, -1],
            },
            "WallBottom": {
                "Origine": [0, self.dom_params["origin_y"]],
                "Normale": [0, 1],
            },
        }
        new_lines = []
        i = 0
        while i < len(lines):
            new_line = lines[i]
            wall_keys = ["WallIn", "WallOut", "WallTop", "WallBottom"]
            matching_wall = next(
                (wall for wall in wall_keys if f"Nom= {wall}" in new_line), None
            )
            if matching_wall:
                if (i + 1 < len(lines) and "Origine= " in lines[i + 1]) and (
                    i + 2 < len(lines) and "Normale=" in lines[i + 2]
                ):
                    split_line1 = lines[i + 1].split("{")
                    split_line1[1] = (
                        f" Origine= {bounds[matching_wall]['Origine'][0]} {bounds[matching_wall]['Origine'][1]} }}"
                    )
                    newnew_line = "{".join(split_line1) + "\n"
                    split_line2 = lines[i + 2].split("{")
                    split_line2[1] = (
                        f" Normale= {bounds[matching_wall]['Normale'][0]} {bounds[matching_wall]['Normale'][1]} }}"
                    )
                    newnewnew_line = "{".join(split_line2) + "\n"
                    new_lines.append(new_line)
                    new_lines.append(newnew_line)
                    new_lines.append(newnewnew_line)
                    i += 3
                else:
                    new_lines.append(new_line)
                    i += 1
            else:
                new_lines.append(new_line)
                i += 1
        # write new GeometresE
        with open(GeometresE_path, "wt") as fout:
            fout.writelines(new_lines)

    def generate_geometres_objects(self, objects_meshdict: Dict[str, str]):
        """Generate the GeometresObjects.mtc file for the defining each object Appartient/Levelset.

        :param objects_list: (List[str]) list of the objects names
        :param objects_meshpaths: (List[str]) list of the paths to the objects meshes
        """
        geometres_objects = Mtc(
            os.path.join(self.simu_dir, "DragLift/GeometresObjects.mtc")
        )
        geometres_objects_content = []
        models_list = []
        # create submodels
        for object_name, object_mesh in objects_meshdict.items():
            geometres_objects_content.append(
                geometres_objects.format_definition_model(
                    model_name=object_name,
                    mesh_model="MaillagePrincipal",
                    dim=self.dim,
                )
            )
            geometres_objects_content.append(
                geometres_objects.format_geo_model(
                    model_name=object_name,
                    mesh_file=object_mesh,
                    mesh_model="MaillagePrincipal",
                )
            )
            geometres_objects_content.append(
                geometres_objects.format_distance_model(
                    model_name=object_name, mesh_model="MaillagePrincipal"
                )
            )
            models_list.append(f"Definition{object_name}")
            models_list.append(f"Geo{object_name}")
            models_list.append(f"Distance{object_name}")
        # create header model
        geometres_objects_content.insert(
            0,
            geometres_objects.format_ModeleDeModeles(
                model_name="GeometresObjects",
                submodels_list=models_list,
                mesh_model="MaillagePrincipal",
            ),
        )
        # write content
        geometres_objects.write(
            content="\n\n".join(geometres_objects_content), overwrite=True
        )
        return

    def generate_draglift_objects(self, objects_list: List[str]):
        """Generate the DragLiftObjects.mtc file for the defining each object DragLift"""
        # main dragliftobjects mtc
        draglift_objects = Mtc(
            os.path.join(self.simu_dir, "DragLift/DragLiftObjects.mtc")
        )
        draglift_objects_content = [
            draglift_objects.format_ModeleDeModeles(
                model_name="DragLiftObjects",
                submodels_list=[
                    f"DragLift{object_name}" for object_name in objects_list
                ],
                mesh_model="MaillagePrincipal",
            ),
            draglift_objects.format_mtc_declaration(
                list_mtc_paths=[
                    f"DragLift/DragLift{object_name}.mtc"
                    for object_name in objects_list
                ],
            ),
        ]
        draglift_objects.write(
            content="\n\n".join(draglift_objects_content), overwrite=True
        )
        # object-specific draglift mtcs
        for object_name in objects_list:
            draglift_object = Mtc(
                os.path.join(self.simu_dir, f"DragLift/DragLift{object_name}.mtc")
            )
            draglift_object_content = draglift_object.format_draglift_model(
                object_name=object_name,
                mesh_model="MaillagePrincipal",
                dim=self.dim,
            )
            draglift_object.write(content=draglift_object_content, overwrite=True)
        return

    def disable_mesh_adaptation(self):
        """
        Toggle the BLM mesh adaptation in Increments/increments.mtc file: if mesh_adapt parameter
        is set to false then launch the simulation without BLM adaptation, use provided GMSH domain.t mesh
        """
        increments_path = os.path.join(self.simu_dir, "Increments/increments.mtc")
        try:
            with open(increments_path, "r+", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"\nError: File {increments_path} not found.")
        # run through lines and edit params
        new_lines = []
        for line in lines:
            if "Modele= BLM" in line:
                line = "//" + line
            new_lines.append(line)
        # write new increments
        with open(increments_path, "wt") as fout:
            fout.writelines(new_lines)

    def disable_force_computation(self):
        """
        Toggle the drag/lift force computation in Increments/increments.mtc file: if calc_force parameter
        is set to false then launch the simulation without DragLift model.
        """
        increments_path = os.path.join(self.simu_dir, "Increments/increments.mtc")
        try:
            with open(increments_path, "r+", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"\nError: File {increments_path} not found.")
        # run through lines and edit params
        new_lines = []
        for line in lines:
            if "Modele= DragLift" in line:
                line = "//" + line
            new_lines.append(line)
        # write new increments
        with open(increments_path, "wt") as fout:
            fout.writelines(new_lines)

    def toggle_feature_outputs(self):
        """
        Make sure to only output in vtu (via IO/output.mtc) the feature fields
        listed in the configuration file under 'graph_parameters'. Raise warning
        if a feature is listed but not present in the simulation outputs (by default
        all possible features should be outputted to IO/Sortie.mtc model).
        """
        # init
        output_path = os.path.join(self.simu_dir, "IO/output.mtc")
        try:
            with open(output_path, "r+", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"\nError: File {output_path} not found.")
        # run through lines and edit params
        new_lines = []
        feature_list = self.graph_params["features"]
        i = 0
        while i < len(lines):
            line = lines[i]
            if "{ DependanceAEcrire=" in line:
                new_lines.append(line)
                i += 1
                while i < len(lines) and lines[i].strip() != "}":
                    if "Champ= " in lines[i]:
                        if any(
                            f"Champ= {feature}" in lines[i] for feature in feature_list
                        ):
                            new_lines.append(lines[i])
                        else:
                            new_lines.append("//" + lines[i])
                    else:
                        new_lines.append(lines[i])
                    i += 1

            else:
                new_lines.append(line)
                i += 1
        # write new Sorties
        with open(output_path, "wt") as fout:
            fout.writelines(new_lines)

    def modify_inlet_amplitude(self, amplitude: float):
        """Modify the inlet amplitude in Parametres.mtc file.
        This modifies the value for the champ 'AmplitudeInlet'

        Args:
            amplitude (float ): inlet velocity amplitude factor"""
        parametres_path = os.path.join(self.simu_dir, "IO/Parametres.mtc")
        try:
            parametres_mtc = Mtc(path=parametres_path)
            fields_to_modify = {"AmplitudeInlet": amplitude}
            parametres_mtc_content = parametres_mtc.modif_champ(
                mtc_content=parametres_mtc.content,
                raise_not_found=False,
                **fields_to_modify,
            )
            parametres_mtc.write(content=parametres_mtc_content, overwrite=True)
        except Exception as e:
            print(f"Error modifying AmplitudeInlet in {parametres_path}: {e}")
            raise
        return

    def run_simulation(self, verbose=False):
        """Run the simulation with srun (SLURM) or mpirun and subprocess, on self.n_cores CPUs."""
        lock_command = ["touch", os.path.join(self.simu_dir, "run.lock")]
        # Choose between srun (SLURM) or mpirun based on has_slurm flag
        if self.has_slurm:
            cfd_command = [
                "srun",
                "--exclusive",
                "-n",
                str(self.n_cores),
                self.driver,
                self.launcher,
            ]
        else:
            cfd_command = [
                "mpirun",
                "-n",
                str(self.n_cores),
                self.driver,
                self.launcher,
            ]
        log_file = os.path.join(self.simu_dir, "log.out")
        if self.multigrid:
            if self.has_slurm:
                grid_command = [
                    "srun",
                    "--exclusive",
                    "-n",
                    str(self.n_cores),
                    self.driver,
                    self.launcher_grid,
                ]
            else:
                grid_command = [
                    "mpirun",
                    "-n",
                    str(self.n_cores),
                    self.driver,
                    self.launcher_grid,
                ]
            movegrid1_command = ["mv", "meshes/domain.t", "meshes/domain_gmsh.t"]
            movegrid2_command = [
                "cp",
                "-r",
                "OutputMesh/Mesh_00002.t",
                "meshes/domain.t",
            ]
        try:
            start_time = pd.Timestamp.now()
            subprocess.run(lock_command, check=True)
            if self.multigrid:
                with open(log_file, "w") as log:
                    subprocess.run(
                        grid_command,
                        cwd=self.simu_dir,
                        check=True,
                        stdout=log,
                        stderr=log,
                        timeout=10800,  # 3 hours
                    )
                    subprocess.run(
                        movegrid1_command,
                        cwd=self.simu_dir,
                        check=True,
                        stdout=log,
                        stderr=log,
                    )
                    subprocess.run(
                        movegrid2_command,
                        cwd=self.simu_dir,
                        check=True,
                        stdout=log,
                        stderr=log,
                    )
            with open(log_file, "w") as log:
                subprocess.run(
                    cfd_command,
                    cwd=self.simu_dir,
                    check=True,
                    stdout=log,
                    stderr=log,
                    timeout=self.timeout,
                )
            end_time = pd.Timestamp.now()
            # postprocess simulation
            self.results_to_xdmf()
            if self.cfd_params["calc_force"]:
                self.results_to_csv()
            self.cleanup()
            if verbose:
                print(f"\t\t(simu {self.simu_name} walltime: {end_time - start_time}s)")
        except subprocess.CalledProcessError as e:
            print(
                f"Error running simulation {self.simu_name}: {e} \n\t\t-> see log: {log_file}"
            )
            raise

    def results_to_xdmf(self):
        """compress simulation vtus in Resultats/2d* folder to xdmf/h5"""
        # init
        vtu_folders = glob.glob(
            os.path.join(self.simu_dir, "Resultats", f"{self.dim}d*")
        )
        for vtu_folder in vtu_folders:
            folder_name = os.path.basename(vtu_folder).split(f"{self.dim}d")[-1]
            vtu_files = glob.glob(os.path.join(vtu_folder, "*.vtu"))
            vtu_start_idx = (
                self.graph_params["vtu_start_in_seconds"]
                / self.cfd_params["PasDeTemps"]
            )
            vtus_to_compress = []
            for vtu in vtu_files:
                vtu_idx = int(vtu.split("_")[-1].split(".")[0])
                if vtu_idx >= vtu_start_idx:
                    vtus_to_compress.append(vtu)

            # compress
            xdmf_filename = os.path.join(
                self.save_path, f"{folder_name}{self.simu_name}.xdmf"
            )
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

    def results_to_csv(self):
        """Read all the simu/Resultats/.txt files and join them into a single dataframe,
        where the columns are the outputs: Object,Temps,Fx,Fy,Fz,Mx,My,Mz. Save as csv file.
        Selecting the subdataframe according to object=='some_object' should give the simu results for that object.
        Each object does not necessarily have data for all functions (Fx,Fy,Fz,Mx,My,Mz).

        Note that the data is averaged over repeated timesteps to account for reprise of a simulation.

        Note that data in 2D is supposed to not include torque data (torque available only in 3D)
        """
        # init
        data_folder = os.path.join(self.simu_dir, "Resultats")
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
                    if self.dim == 2:
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
                    if self.dim == 2:
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
            outputname = f"{self.simu_name}_data.csv"
            output_csv = os.path.join(self.save_path, outputname)
            total_df.to_csv(output_csv, index=False)
        else:
            print(f"\t\tNo data files found in {data_folder}")

    def edit_domain_boundaries(self):
        """Edit the domain boundaries in GeometresE.mtc"""
        raise NotImplementedError

    def apply_outputs(self):
        """Make sure to include features fields to Sorties.mtc file"""
        raise NotImplementedError

    def cleanup(self):
        """Remove the simulation directory"""
        if self.save_path != self.simu_dir:
            shutil.rmtree(self.simu_dir, ignore_errors=True)
        else:
            print(
                f"Warning: simulation directory {self.simu_dir} is the same as save_path {self.save_path}, "
                "keeping only csv, xdmf, h5 files."
            )
            for item in os.listdir(self.simu_dir):
                item_path = os.path.join(self.simu_dir, item)
                if os.path.isfile(item_path):
                    if not (
                        item_path.endswith(".csv")
                        or item_path.endswith(".xdmf")
                        or item_path.endswith(".h5")
                    ):
                        os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path, ignore_errors=True)


# Example usage
if __name__ == "__main__":
    # test out on a single config
    meta_path = "config/airfoil.json"
    simu = Simulation(meta_path)
    #
    print("Prepping simulation...")
    simu.prep_simulation()
    #
    print("Creating simualtion geometry...")
    geometry = Airfoil(
        parameters_dict=simu.meta,
        airfoil_points_list=["NACA0010"],
        chords=[1.0],
        thicknesses=[1.0],
        angles=[0],
        centers_x=[0],
        centers_y=[0],
        num_airfoils=1,
        dim=2,
        path=simu.simu_dir,
    )
    geometry.auto_mesh_options()
    geometry.apply_box2params()
    _ = geometry.create_domain(save_mesh=True, dim_mesh=2)
    _ = geometry.create_object(force_model="", save_mesh=True, dim_mesh=2)
    geometry.finalize()
    convert_gmsh_to_mtc(
        input=os.path.join(simu.simu_dir, "object.msh"),
        output=os.path.join(simu.simu_dir, "object.t"),
        verbose=False,
    )
    convert_gmsh_to_mtc(
        input=os.path.join(simu.simu_dir, "domain.msh"),
        output=os.path.join(simu.simu_dir, "domain.t"),
        verbose=False,
    )
    move_meshes(
        output_directory=os.path.join(simu.simu_dir, "meshes"),
        extensions=[".t"],
        source_directory=simu.simu_dir,
    )
    move_meshes(
        output_directory=os.path.join(simu.simu_dir, "meshes_GMSH"),
        extensions=[".msh", ".geo_unrolled", ".vtk"],
        source_directory=simu.simu_dir,
    )
    #
    print("Running simulation...")
    # NB: requires mpirun -> os.system("module load cimlibxx/master")
    simu.run_simulation()
    #
    print("Simulation complete.")
