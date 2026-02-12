import os
import shutil
import subprocess
from typing import Any, Dict

from graphdrl.utils.file_handling import Mtc
from graphdrl.utils.meshio_mesh import vtu_to_meshes


class CimlibEnv:
    """
    Class to handle Boundary Layer Meshing and/or feature initialization via call to CIMLIB.
    """

    def __init__(
        self,
        parameters: Dict[str, Any],
        path: str = "env",
    ):
        """
        Create simulation/BLM repo, load data, and initialize parameters.

        Args:
            parameters (dict): Dictionary of case meta parameters located in environment_config.
            path (str): Path to the env directory, within which a cimlib template folder will be copied
            number_cores (int): Number of CPU cores to use for the run.
        """
        self.params: str = parameters
        self.path: str = path
        self.blm_adaptation: bool = self.params["traj_parameters"].get(
            "mesh_adapt", False
        )
        self.dir: str = (
            os.path.join(self.path, "blm")
            if self.blm_adaptation
            else os.path.join(self.path, "init_feats")
        )
        self.blm_max_iter: int = 20
        self.dim: int = self.params.get("dim", 2)
        self.n_cores: int = self.params["traj_parameters"].get("number_cores", 1)
        # cpu count sanity check
        if self.n_cores > os.cpu_count():
            raise ValueError(
                f"Error: You requested {self.n_cores} cores but your system only has {os.cpu_count()} cores."
            )
        self.timeout: int = 300  # seconds

        self.template_dir: str = (
            os.path.join("environment_config", "blm")
            if self.blm_adaptation
            else os.path.join("environment_config", "init_feats")
        )
        self.launcher: str = "lanceur/Principale.mtc"
        self.driver: str = os.path.abspath(self.params["traj_parameters"]["driver"])
        self.cfd_params: Dict[str, Any] = self.params["traj_parameters"]
        self.geo_params: Dict[str, Any] = self.params["geometry_parameters"]
        self.dom_params: Dict[str, Any] = self.params["domain_parameters"]

        self.IHM: str = "IHM.mtc"
        self.result_file: str = os.path.join(self.dir, "mesh_00000.vtu")

        self._check_mpirun()

    def _check_mpirun(self):
        """Check if mpirun is available in the system if needed."""
        if (
            self.blm_adaptation
            or self.params["traj_parameters"].get("init_features", False)
            or self.n_cores > 1
        ):
            try:
                subprocess.run(
                    ["mpirun", "--version"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except FileNotFoundError:
                raise EnvironmentError(
                    "mpirun not found. Please ensure that MPI is installed and mpirun is available in your PATH."
                )

    def prep(self):
        """Prepare the blm setup directory"""
        shutil.copytree(self.template_dir, self.dir, dirs_exist_ok=True)
        if self.blm_adaptation:
            self.apply_IHM()
        self.apply_domain_size()

    def apply_IHM(self):
        """Apply the traj parameters to IHM.mtc file. Names in traj_parameters
        of meta json should match the names of the parameters in IHM.mtc file."""
        IHM_path = os.path.join(self.dir, self.IHM)
        # checks + formatting
        boxes_values = {
            "BLMHbox1": self.cfd_params["Hbox123"][0],
            "BLMHbox2": self.cfd_params["Hbox123"][1],
            "BLMHbox3": self.cfd_params["Hbox123"][2],
        }
        max_iterations = {"BLMMaxIteration": self.blm_max_iter}
        ihm_params = {**self.cfd_params, **boxes_values, **max_iterations}

        # edit ihm
        ihm_mtc = Mtc(path=IHM_path)
        ihm_content = ihm_mtc.modif_target(
            mtc_content=ihm_mtc.content,
            raise_not_found=False,
            **ihm_params,
        )
        ihm_mtc.write(content=ihm_content, overwrite=True)

    def find_box2_params(self, num_objects: int = 1) -> dict:
        """
        Find the parameters for the box2.
        Args:
            num_objects (int): Number of objects to consider in the box2 size computation.
        Returns:
            dict: Dictionary of box2 parameters.
        """
        try:
            x_min = self.geo_params["origin"][0] - 1.5 * self.geo_params["chord"]
            y_min = self.geo_params["origin"][1] - 1.5 * self.geo_params["chord"]
            dx = num_objects * self.geo_params["spacing"] + 3 * self.geo_params["chord"]
            dy = self.geo_params["chord"] * 3
            box2_params = {"Center2": [x_min, y_min], "Box2": [dx, dy]}
        except KeyError as e:
            raise KeyError(
                f"Error: Missing geometry parameter {e} required for BLM Box2 computation."
            ) from e
        return box2_params

    def apply_blm_boxsize(self, num_objects: int = 1):
        """Find and apply the BLM Box2 and Center2 parameters in BLM/ from geometry specifications.
        Args:
            num_objects (int): Number of objects to consider in the box2 size computation.
        """
        blm_box2_path = os.path.join(self.dir, "BLM", "Box2.txt")
        blm_center2_path = os.path.join(self.dir, "BLM", "Center2.txt")
        # compute params
        box2_params = self.find_box2_params(num_objects=num_objects)
        # Box2
        box2_mtc = Mtc(path=blm_box2_path)
        box2_dx = box2_params["Box2"][0]
        box2_dy = box2_params["Box2"][1]
        box2_mtc.write(content=f"{box2_dx} {box2_dy}\n", overwrite=True)
        # Center2
        center2_mtc = Mtc(path=blm_center2_path)
        center2_x = box2_params["Center2"][0]
        center2_y = box2_params["Center2"][1]
        center2_mtc.write(content=f"{center2_x} {center2_y}\n", overwrite=True)

    def apply_domain_size(self):
        """Apply the domain size to the GeometresE.mtc file for correct boundary locations
        and flags used in the boundary conditions definitions."""
        GeometresE_path = os.path.join(self.dir, "Geometrie/GeometresE.mtc")
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

    def run(self):
        """Run a blm adaptation and.or feature initialization with mpirun and subprocess, on self.n_cores CPUs."""
        log_file = os.path.join(
            self.dir, "blm.out" if self.blm_adaptation else "init_feats.out"
        )
        command = [
            "mpirun",
            "-n",
            str(self.n_cores),
            self.driver,
            self.launcher,
        ]
        try:
            with open(log_file, "w") as log:
                try:
                    subprocess.run(
                        command,
                        cwd=self.dir,
                        check=True,
                        stdout=log,
                        stderr=log,
                        timeout=self.timeout,
                    )
                except subprocess.TimeoutExpired as e:
                    print(
                        f"Timeout while running CIMLIB with {self.n_cores} cores ({e}). Retrying with 1 core...",
                        flush=True,
                    )
                    # First timeout: retry with a single core
                    command_retry = list(command)
                    if int(command_retry[2]) > 1:
                        command_retry[2] = "1"
                    else:
                        raise RuntimeError(
                            "Timeout while running CIMLIB."
                            "Cannot retry with 1 core because the number of cores is already 1."
                            "Please review partition script or increase timeout value."
                        )
                    try:
                        subprocess.run(
                            command_retry,
                            cwd=self.dir,
                            check=True,
                            stdout=log,
                            stderr=log,
                            timeout=self.timeout,
                        )
                    except subprocess.TimeoutExpired as e2:
                        raise RuntimeError(
                            f"Timeout while running CIMLIB with 1 core ({e2}). "
                            "Error probably raised by a low timeout value or by partition error in mesh adaptation."
                            "For the latter, review partition script or reduce number of cores used for cimlib runs."
                            "For the former, increase timeout value or increase the number of cores used for speedup."
                        ) from e2
                meshes, times = vtu_to_meshes(vtu_file_path=self.result_file, time=0.0)
                self.cleanup()
        except subprocess.CalledProcessError as e:
            print(f"Error running blm in {self.dir}: {e} \n\t\t-> see log: {log_file}")
            raise
        return meshes, times

    def cleanup(self):
        """remove blm adaptation directory and files"""
        shutil.rmtree(self.dir, ignore_errors=True)
