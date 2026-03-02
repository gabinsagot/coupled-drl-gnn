import json
import pandas as pd
import os
import shutil
import time
from typing import Any, Dict, Tuple
from abc import ABC
import multiprocessing as mp
from tqdm import tqdm


from configs import (
    Configs,
    ConfigsAirfoil,
)
from geometries import (
    Geometry,
    Airfoil,
)
from simulation import Simulation
from convert.mesh import convert_gmsh_to_mtc
from utils import move_meshes, get_unique_path, print_section


class Dataset(ABC):
    """
    Dataset class to handle the generation of a dataset based on a meta file, and the
    decision to create or not the pool of configurations (load if pool exists).

    The class is used to handle the following workflow: create a pool of configurations (use Configs class),
    run a config simulation (using Simulation class to prep simulation, and then
    create geometry of config using Geometry class, run cfd, postprocess simulation)
    save simulation data to dataset, clean up simulation data and zip dataset if prompted.

    Args:
        meta_path (str): path to the meta file, to create meta_dict
        create_configs_pool (bool): whether to create the pool of configurations or load
        from existing pool found at meta_dict["configs_pool_path"]
    """

    def __init__(
        self,
        meta_path: str,
        create_configs_pool: bool = True,
        ignore_slurm: bool = False,
    ):
        with open(meta_path, "r") as fp:
            meta = json.load(fp)

        self.meta: Dict[str, Any] = meta

        # params dicts
        self.cfd_params: Dict[str, Any] = self.meta["cfd_parameters"]
        self.graph_params: Dict[str, Any] = self.meta["graph_parameters"]

        # case params
        self.type: str = self.meta["case"]
        self.case_name: str = self.meta["case_name"]
        self.n_cores: int = self.cfd_params["number_cores"]
        self.dim: int = self.meta["dim"]
        self.multigrid: bool = self.graph_params["multigrid"]
        self.n_jobs: int = self.meta["number_parallel_jobs"]
        self.ignore_slurm: bool = ignore_slurm
        max_cpus_available = int(os.environ.get("SLURM_NTASKS", mp.cpu_count()))
        if self.n_jobs * self.n_cores > max_cpus_available:  # sanity check
            raise ValueError(
                f"Number of parallel jobs ({self.n_jobs}) times "
                f"number of cores/job ({self.n_cores}) exceeds "
                f"available CPUs ({max_cpus_available})"
            )

        # configs
        self.load_configs: bool = not (create_configs_pool)
        self.configs_pool_path: str = self.meta["configs_pool_path"]
        self.n_configs: int = self.meta["num_configs"]
        self.configs_pool: Configs = self.which_config_type(
            meta_dict=self.meta, create=create_configs_pool
        )
        self.configs: pd.DataFrame = self.configs_pool.configs

        # paths
        self.simu_path: str = os.path.join("./simu", "simu_" + self.case_name)
        self.dataset_path: str = os.path.join("./dataset", "dataset_" + self.case_name)

    def prep_directories(self, clear: bool = False) -> None:
        """Create directories for the simulations and the dataset data"""
        if clear:
            if os.path.exists(self.simu_path):
                shutil.rmtree(self.simu_path)
            if os.path.exists(self.dataset_path):
                shutil.rmtree(self.dataset_path)
        self.simu_path = get_unique_path(self.simu_path)
        self.dataset_path = get_unique_path(self.dataset_path)
        os.makedirs(self.simu_path, exist_ok=True)
        os.makedirs(self.dataset_path, exist_ok=True)
        try:
            if self.load_configs:
                # copy the loaded configs pool instead of the original
                loaded_configs_pool_path = os.path.join(
                    self.configs_pool_path.replace(".txt", "_selected.txt"),
                )
                shutil.copy(loaded_configs_pool_path, self.dataset_path)
                shutil.copy(
                    os.path.splitext(loaded_configs_pool_path)[0] + ".pkl",
                    self.dataset_path,
                )
            else:
                # copy original configs pool
                shutil.copy(self.configs_pool_path, self.dataset_path)
                shutil.copy(
                    os.path.splitext(self.configs_pool_path)[0] + ".pkl",
                    self.dataset_path,
                )
        except Exception as e:
            print(f"No configs pool file found to copy to dataset: {e}")
            raise

    def which_config_type(self, meta_dict: str, create: bool) -> Configs:
        """Return the config pool class (Configs type) based on the case type.

        Args:
            meta_dict (str):  meta parameters dictionary
            create (bool): whether to create the pool of configurations"""
        if self.type == "airfoil":
            return ConfigsAirfoil(
                meta_dict=meta_dict,
                path=self.configs_pool_path,
                num_configs=self.n_configs,
                create=create,
            )
        else:
            raise ValueError(
                f"Invalid case type: {self.type}. Available options are: airfoil."
            )

    def create_geometry(self, config: pd.DataFrame, path_to_cfd: str) -> Geometry:
        """
        Create the geometry based on the case type, with associated cfd directory.
        Returns the geometry object

        Args:
            config (pd.DataFrame): configuration to create the geometry from
            path_to_cfd (str): path to the cfd directory where meshes will be created
        """
        if self.type == "airfoil":
            if "morph" in self.meta["geometry_parameters"]["airfoil_type"].lower():
                airfoil_points_list = ["morph"] * config["number_airfoils"]
            elif "naca" in self.meta["geometry_parameters"]["airfoil_type"].lower():
                airfoil_points_list = [config["naca_code"]] * config["number_airfoils"]
            else:
                raise ValueError(
                    f"Invalid airfoil type: {self.meta['geometry_parameters']['airfoil_type']}. "
                    f"Available options are: NACAxxxx, morph(ed)."
                )
            return Airfoil(
                parameters_dict=self.meta,
                airfoil_points_list=airfoil_points_list,
                chords=config["chords"].tolist(),
                thicknesses=config["thicknesses"].tolist(),
                angles=config["angles"].tolist(),
                centers_x=config["x_objects"].tolist(),
                centers_y=config["y_objects"].tolist(),
                num_airfoils=config["number_airfoils"],
                cambers=config["cambers"].tolist() if "cambers" in config else None,
                dim=self.dim,
                path=path_to_cfd,
            )
        else:
            raise ValueError(
                f"Invalid case type: {self.type}. Available options are: airfoil."
            )

    def print_info(self) -> None:
        """Print out detailed information about the dataset generation job."""
        print("\n\t\t\t*** Dataset Information ***\n")
        # general
        general_params = {
            "Dataset name": self.case_name,
            "Dataset type": self.type,
            "Total allocated cores": self.n_cores * self.n_jobs,
            "Number of configurations": self.n_configs,
        }
        print_section(title="General Parameters", content=general_params)
        # meta
        print_section(title="Dataset Meta Parameters Used", content=self.meta)
        # configs
        print("-" * 50)
        print("\t\t\tConfigs of the Dataset:")
        print("-" * 50)
        self.configs_pool.display()
        print(f"{'Total possible configs':<30}{self.configs_pool.get_size()}")
        print("-" * 50, "\n")
        # simu
        simulation_params = {
            "Driver used for simulations": os.path.abspath(self.meta["driver"]),
            "Number of cores per simulation": self.n_cores,
            "Number of parallel jobs": self.n_jobs,
            "BLM meshing used": self.cfd_params["mesh_adapt"],
            "Multigrid used": self.multigrid,
            "Features saved": self.graph_params["features"],
        }
        print_section(title="Simulation Parameters", content=simulation_params)

    def handle_meshes(self, cfd_path: str, other_meshes: Dict[str, Any] = None) -> None:
        """Handle the meshes created by the geometry class in specified cfd directory

        Args:
            cfd_path (str): path to the cfd directory where meshes are created
            other_meshes (Dict[str, Any]): dictionary of other meshes to convert and move (needs a 'model' key
            which is the name of the individual object without extension (e.g. 'cylinder0'))
        """
        convert_gmsh_to_mtc(
            input=os.path.join(cfd_path, "object.msh"),
            output=os.path.join(cfd_path, "object.t"),
            verbose=False,
        )
        convert_gmsh_to_mtc(
            input=os.path.join(cfd_path, "domain.msh"),
            output=os.path.join(cfd_path, "domain.t"),
            verbose=False,
        )
        if other_meshes is not None:
            for mesh in other_meshes:
                convert_gmsh_to_mtc(
                    input=os.path.join(cfd_path, mesh["model"] + ".msh"),
                    output=os.path.join(cfd_path, mesh["model"] + ".t"),
                    verbose=False,
                )
        # prevent resource unavailability
        time.sleep(1)
        move_meshes(
            output_directory=os.path.join(cfd_path, "meshes"),
            extensions=[".t"],
            source_directory=cfd_path,
        )
        move_meshes(
            output_directory=os.path.join(cfd_path, "meshes_GMSH"),
            extensions=[".msh", ".geo_unrolled", ".vtk"],
            source_directory=cfd_path,
        )

    def subrun(self, config: pd.DataFrame) -> Tuple[str, float]:
        """Run a single config generation: meshing, cfd, postprocessing

        Args:
            config (pd.DataFrame): configuration to run the simulation for

        Returns:
            Tuple[str, float]: config name and elapsed time in seconds
        """
        config_name = config["Config"]
        cfd_path = os.path.join(self.simu_path, f"{self.type}_{config_name}")
        start_time = time.time()
        # init simulation
        try:
            simulation = Simulation(
                meta_dict=self.meta,
                simu_path=self.simu_path,
                simu_name=f"{self.type}_{config_name}",
                save_path=self.dataset_path,
                number_cores=self.n_cores,
                multigrid=self.multigrid,
                ignore_slurm=self.ignore_slurm,
            )
            simulation.prep_simulation()
        except Exception as e:
            print(f"Error in initializing simulation for config {config_name}: {e}")
            raise
        # handle geometry
        try:
            geometry = self.create_geometry(config=config, path_to_cfd=cfd_path)
            geometry.auto_mesh_options()
            geometry.apply_box2params()
            geometry.create_domain(save_mesh=True, dim_mesh=self.dim)
            geometry.create_object(force_model="", save_mesh=True, dim_mesh=self.dim)
            all_objects_dict = geometry.create_each_object(save_mesh=True)
            geometry.finalize()
        except Exception as e:
            print(f"Error in creating geometry for config {config_name}: {e}")
            raise
        # handle meshes
        try:
            self.handle_meshes(cfd_path, other_meshes=all_objects_dict)
        except Exception as e:
            print(f"Error in handling meshes for config {config_name}: {e}")
            raise
        # object-specific simulation setup
        try:
            objects_meshdict = {}  # create dict of objectname: rel mesh path
            for mesh_object in all_objects_dict:
                objects_meshdict[mesh_object["model"]] = (
                    f"meshes/{mesh_object['model']}.t"
                )
            simulation.generate_geometres_objects(objects_meshdict=objects_meshdict)
            simulation.generate_draglift_objects(
                objects_list=list(objects_meshdict.keys())
            )
        except Exception as e:
            print(
                f"Error in generating objects-related mtc models for config {config_name}: {e}"
            )
            raise
        # set inlet BC velocity based on config
        try:
            if "v_inlet" in config:
                v_inlet = config["v_inlet"].tolist()[0]
                simulation.modify_inlet_amplitude(amplitude=v_inlet)
        except Exception as e:
            print(
                f"Error in modifying inlet BC amplitude for config {config_name}: {e}"
            )
            raise
        # run simulation and postprocess
        try:
            simulation.run_simulation()
            elapsed = time.time() - start_time
            return config_name, elapsed
        except Exception as e:
            tqdm.write(f"Error in running simulation {config_name}: {e}")
            raise

    def generate_serial(self) -> None:
        """Generate the dataset, by iterating over the selected configurations
        and running each simulation (init simu, create geometry, meshes, run cfd,
        post process simulation, save results to dataset)."""
        for row in self.configs.index:
            self.subrun(self.configs.iloc[row])
        print("Dataset generation complete!")

    def generate(self) -> None:
        """Generate the dataset, by iterating over the selected configurations
        and running each simulation (init simu, create geometry, meshes, run cfd,
        post process simulation, save results to dataset)."""
        total_configs = len(self.configs)
        completed = 0
        failed = 0
        start_time = time.time()
        cumulative_compute_time = 0.0  # track total compute time across all configs

        with mp.Pool(processes=self.n_jobs) as pool:
            try:
                # create iterator of arguments for worker
                tasks = [(self.configs.iloc[row], self) for row in self.configs.index]

                # use imap_unordered for progressive results with tqdm
                with tqdm(
                    total=total_configs,
                    desc="dataset generation",
                    unit="config",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
                    position=0,
                    leave=True,
                ) as pbar:
                    for result in pool.imap_unordered(_worker_wrapper, tasks):
                        if result["success"]:
                            completed += 1
                            cumulative_compute_time += result["time"]
                            avg_compute_time = cumulative_compute_time / completed
                            postfix = (
                                f"last: {result['config']} ({result['time']:.1f}s) | "
                                f"avg: {avg_compute_time:.1f}s/config | "
                                f"v {completed} - x {failed}"
                            )
                            pbar.set_postfix_str(postfix)
                        else:
                            failed += 1
                            postfix = (
                                f"last: {result['config']} (failed) | "
                                f"v {completed} - x {failed}"
                            )
                            pbar.set_postfix_str(postfix)
                            tqdm.write(
                                f"x config {result['config']} failed: {result['error']}"
                            )
                        pbar.update(1)
            except Exception as e:
                tqdm.write(f"error in process pool: {e}")

        total_time = time.time() - start_time
        avg_compute_time = cumulative_compute_time / completed if completed > 0 else 0
        tqdm.write(
            f"\nDataset generation complete! {completed}/{total_configs} successful in {int(total_time)}s "
            f"(avg compute time: {avg_compute_time:.1f}s/config)"
        )
        if failed > 0:
            tqdm.write(f"Warning: {failed} config(s) failed")

    def zip(self) -> None:
        """Zip the dataset folder to zip format in dataset folder.
        Warning: deletes dataset folder after successful zipping."""
        base_zip_filename = os.path.join(
            os.path.dirname(self.dataset_path),
            self.meta["configs_pool_path"]
            .split("/")[-1]
            .split(".")[0]
            .split("configs_pool_")[-1],
        )
        zip_filename = base_zip_filename
        if os.path.exists(zip_filename + ".zip"):
            counter = 1
            while os.path.exists(f"{base_zip_filename}_{counter}.zip"):
                counter += 1
            zip_filename = f"{base_zip_filename}_{counter}"
        try:
            print("\nZipping dataset...")
            shutil.make_archive(zip_filename, "zip", self.dataset_path)
            print("\tZipped dataset! File saved at:", zip_filename + ".zip")
            print("Cleaning up dataset folder...")
            shutil.rmtree(self.dataset_path)
            print("\tDataset folder cleaned up!")
        except Exception as e:
            print(f"Error in zipping dataset: {e}")
            raise


def _worker_wrapper(args: Tuple[pd.DataFrame, Dataset]) -> Dict[str, Any]:
    """Wrapper to unpack arguments for multiprocessing.imap_unordered."""
    return worker(*args)


def worker(config: pd.DataFrame, dataset_instance: Dataset) -> Dict[str, Any]:
    """Worker function to run the simulation for a single config.
    This function is used in the multiprocessing pool. It returns a dict with results.
    This is used to handle errors in the multiprocessing call and continue running the other
    simulations.

    Args:
        config (pd.DataFrame): configuration to run the simulation for
        dataset_instance (Dataset): instance of the Dataset class

    Returns:
        Dict with keys: 'config', 'success', 'time', 'error'
    """
    try:
        config_name, elapsed = dataset_instance.subrun(config)
        return {"config": config_name, "success": True, "time": elapsed, "error": None}
    except Exception as e:
        tqdm.write(
            f"error in running config {config['Config']} subrun via safe worker: {e}"
        )
        return {
            "config": config["Config"],
            "success": False,
            "time": 0,
            "error": str(e),
        }
