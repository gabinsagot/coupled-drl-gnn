import os
import shutil
import signal
import subprocess
from typing import List

from absl import logging
import json
import numpy as np

from graphdrl.environment.geometry import Foil
from graphdrl.environment.idw import compute_idw_mesh, compute_foil_points
from graphdrl.environment.cimlib import CimlibEnv
from graphdrl.environment.trajectory import create_full_trajectory_airfoil, make_empty_trajectory_airfoil
from graphdrl.utils.reward import MaximizeLiftToDrag as Reward


class airfoil:
    """Optimization configuration for PBO run using continuous action space."""

    def __init__(self, path):
        """
        Initialize the optimization configuration: define environment, GNN, reward, and other parameters.
        Needed essentials by PBO: path, env_name, action_type, x_min, x_max, x_0, act_size, obs_size, obs,
        env(), step(), observe(), close()

        Args:
            path (str): Path to the optimization run directory.
        """
        # PBO params
        self.path = path
        self.env_name = "airfoil"
        self.action_type = "continuous"
        self.action_mapping = "continuous"
        self.act_size = 7
        # Action space for continuous is [-1,1]
        self.x_min = np.array([-1.0] * self.act_size)
        self.x_max = np.array([1.0] * self.act_size)
        self.obs_size = self.act_size
        self.obs      = np.zeros(self.obs_size)
        self.x_0      = np.array([np.random.rand((1)) for i in range(self.act_size)])   # initial action
        self.bad_rwrd = -10.0
        # Remap to physical scale:
        self.physical_scale = np.array([0.1, 0.15, 0.15, 0.1,           # Camber limits
                                        0.2, 0.2, 0.2])                  # Thickness limits (everything except close to trailing and leading edge)
        
        self.target_surface = 0.080 # NACA0010 area, chord = 1.0. Target surface for reward calculations.

        # case params
        # TODO, implement own geometry creation
        self.case = CaseParameters(
            path=f"environment_config/{self.env_name}.json",
            needs_cimlib=True,
            geometry_class=Foil,
        )

        # GNN params
        self.gnn_parameters = GNNParameters(
            path=f"environment_config/{self.env_name}.json"
        )
        # TODO
        self.graph_feature_names = {
            "velocity": ["x0", "x1"],
            "pressure": "x2",
            "levelset": "x3",
            "nodetype": "x6",
        }
        self.keep_predictions = True  # whether to keep xdmfs
        self.keep_forces = True  # whether to keep forces csv

        # TODO : Reward params
        self.RewardClass = Reward(start_step=150, bad_reward=self.bad_rwrd)

        self.episode = 0

        # Misc
        print(f"Airfoil Optimization initialized. Using {self.RewardClass.reward_type} reward.", flush=True) # TODO: improve message with info at initialization
        logging.set_verbosity(logging.ERROR)

    def env(self, x: np.ndarray, ep: int):
        """Run the GNN prediction for the given environment ep with the given actions x."""

        # Run the GNN prediction
        try:
            # Step 1: setup env folder
            self.setup_env_folder(ep=ep)
            # Rescale actions to physical scale
            x = self.convert_actions_to_physical_scale(actions=x)

            self.write_actions(actions=x, ep=ep)
            # Step 2: create trajectory
            surface = self.create_env_trajectory(actions=x, ep=ep)
        except Exception as e:
            logging.error(f"Setup of environment {ep} failed: {e}", exc_info=True)
            print(f"Episode {ep} reward: {self.bad_rwrd:.6f}", flush=True)        
            return self.bad_rwrd
        
        try:
            # Step 3: run gnn prediction
            self.gnn_prediction()
            # print("GNN prediction done.", flush=True)
        except Exception as e:
            logging.error(f"Prediction failed at {ep} (fatal): {e}", exc_info=True)
            os.killpg(os.getpgrp(), signal.SIGTERM)  # kill parent + child

        # Compute the reward and clean files
        try:
            # Step 4: compute reward
            self.reward = self.RewardClass.compute_reward(
                ep=ep,
                xdmf_path=os.path.join(self.output_path, f"graph_{ep}.xdmf"),
                feature_names=self.graph_feature_names,
                save_data=self.keep_forces,
                airfoil_surface=surface,
                airfoil_target_surface=self.target_surface,
            )
            print(f"Episode {ep} reward: {self.reward:.6f}", flush=True)

            # # Step 5: cleanup
            # self.cleanup_directory(
            #     f"{self.output_path}",
            #     keep_prediction=self.keep_predictions,
            #     keep_reward_data=self.keep_forces,
            # )

            # Step 6: increment episode
            self.episode += 1

            return self.reward

        except Exception as e:
            print(
                f"Environment {ep} postprocess failed (not fatal), assigned bad reward instead: \n\t{e}",
                flush=True,
            )
            return self.RewardClass.bad_reward



    def gnn_prediction(self, strict_load: bool = True):
        predict_config_path = os.path.join(self.output_path, "predict_config.json")
        predict_command = [
            "python",
            "-m",
            "graphphysics.predict",
            f"--predict_parameters_path={predict_config_path}",
            f"--model_path={self.gnn_parameters.model_path}",
            "--no_edge_feature" if self.gnn_parameters.no_edge_feature else "",
            f"--prediction_save_path={self.output_path}",
            "--no_strict_load" if not strict_load else "",
        ]
        self.run_subprocess(
            command=predict_command,
            cwd=".",
            log_file=os.path.join(self.output_path, "predict.log"),
        )

    def setup_env_folder(self, ep: int):
        """
        Setup env folder for the episode (ep) prediction run.
        """
        # ep folder
        self.output_path = os.path.join(self.path, str(ep))
        os.makedirs(self.output_path, exist_ok=True)

        # gnn prediction config file
        predict_json_path = os.path.join(self.output_path, "predict_config.json")
        dataset_json_path = os.path.join(self.output_path, "traj_config.json")
        shutil.copy(self.gnn_parameters.predict_config_path, predict_json_path)
        shutil.copy(self.gnn_parameters.traj_config_path, dataset_json_path)

        # Read and modify the predict config file
        with open(predict_json_path, "r") as f:
            predict_config_dict = json.load(f)
        predict_config_dict["dataset"]["xdmf_folder"] = os.path.abspath(
            self.output_path
        )
        predict_config_dict["dataset"]["meta_path"] = os.path.abspath(dataset_json_path)
        with open(predict_json_path, "w") as f:
            json.dump(predict_config_dict, f, indent=2)

        idw_base_path = os.path.join(os.path.dirname(self.gnn_parameters.predict_config_path), "idw_base_mesh/domain_naca0010.t")
        shutil.copy(idw_base_path, self.output_path)

    def create_env_trajectory(self, actions: List[float], ep: int):
        """Create the trajectory file based on the actions."""
        # make sure actions are valid
        if len(actions) != self.act_size:
            raise ValueError("Invalid actions for trajectory creation.")
        # make sure actions are list of floats
        if not all(isinstance(act, (float, int)) for act in actions):
            raise ValueError(f"Actions must be a list of floats or ints, not {type(actions)}.")

        #### CREATE EMPTY TRAJECTORY #### (.xdmf, etc)
        parameters=self.case.params

        dim = parameters.get("dim", 2)
        dt = parameters["traj_parameters"].get("dt", None)
        if dt is None:
            raise KeyError("You must include a 'dt' value in 'traj_parameters' to indicate timestep between traj frames")
        
        # init cimlib env 
        cimlib_env = CimlibEnv(parameters=parameters, path=self.output_path)
        cimlib_env.prep()

        # # Create geometries
        try:
            name = "object"
            surface = self.create_geometry(actions, name, ep)
        except Exception as e:
            raise ValueError(f"ERROR: Geometry creation failed at episode {ep}: {e}. Assigning bad reward")

        meshes, times = cimlib_env.run()
        cimlib_env.cleanup()

        empty_trajectory = make_empty_trajectory_airfoil(meshes, times, dt)

        # Fill the trajetory 
        try:
            _ = create_full_trajectory_airfoil(
                parameters=self.case.params,
                empty_trajectory=empty_trajectory,
                output_name=os.path.join(self.output_path, f"{self.env_name}_{ep}.xdmf"),
            )
        except Exception as e:
            raise RuntimeError(f"Trajectory creation failed: {e}")
        # note: handle full traj by converting to graph and directly feed to GNN?

        return surface


    def create_geometry(self, actions : np.ndarray, name : str, ep : int, plot : bool = False, naca0010 : bool = False):
        """
        Generates the mesh for the object at the given shape and copies into the simulation/prediction directory
        
        Args: 
            actions: ArrayLike containing the actions to undertake
            name (str): name of the foil

        Returns: 
            float = foil's approximate area (polygon between points)

        Foil geometry generation:
        1) Writes all intermediates to results/.../0/{ep}/init_feats/{txt,geo,msh,t}/
        2) Produces t: results/.../0/{ep}/init_feats/meshes/xxxxxx.t
        """
        # Episode-local sandbox
        episode_root = os.path.join(self.output_path, "init_feats")
        geo_dir = os.path.join(episode_root, "geo")
        msh_dir = os.path.join(episode_root, "msh")
        t_dir   = os.path.join(episode_root, "t")
        for d in (geo_dir, msh_dir, t_dir):
            os.makedirs(d, exist_ok=True)
            
        x_trans_domain = 2.5
        y_trans_domain = 2.0

        # Build the foil
        foil = Foil(10, 1.0, 1.0, work_dir=episode_root, name = name, suffix=f"_{ep}")
        foil.name = name  # 'object'
        foil.generate_airfoil_points(random=False)
        # print("Applying translation actions to foil at ep", ep, flush=True)

        foil.apply_camber_thickness(actions) # Apply deformation actions exccept rotations, applied later
        foil.apply_translation(x_trans_domain, y_trans_domain) # Translate it where the boundary layer mesh is originally

        self.foil_area = foil.compute_surface() # Computes approximate area of the foil (polygons)
        
        # Save original foil points to compute displacements
        naca0010_foil = Foil(10, 1.0, 1.0, work_dir=episode_root, suffix=f"_{ep}")
        naca0010_foil.generate_airfoil_points(random = False)
        naca0010_foil.apply_translation(x_trans_domain, y_trans_domain) # Translate it where the boundary layer mesh is originally

        # Deform domain with IDW according to translation-type actions
        init_foil_points, translated_foil_points = compute_foil_points(naca0010_foil, foil, ep, interp_type="bezier", density=150)

        # TODO : mauavis chemin, à régler
        original_domain_path = os.path.join(self.output_path, "domain_naca0010.t")
        translated_mesh_pts, triangles = compute_idw_mesh(init_foil_points, translated_foil_points, ep, original_domain_path, episode_root, n=1, a=4, b=2, save_t_file=True)
        # Get every new control points & give it to foil.points()
        foil.points = translated_foil_points

        ### Apply the rotations and deform mesh
        rotated_foil = Foil(10, 1.0, 1.0, work_dir=episode_root, name = name, suffix=f"_{ep}")
        rotated_foil.generate_airfoil_points(random=False)
        rotated_foil.points = foil.points

        # rotation = np.pi * actions[-1] / 180.0  #convert degrees to radians
        rotation = np.pi * (-12.0) / 180.0  #convert degrees to radians
        rotated_foil.apply_rotation(rotation)
        if plot : rotated_foil.plot()
        # Deform domain with IDW according to rotation-type actions
        translated_foil_points, rotated_foil_points = compute_foil_points(foil, rotated_foil, ep, interp_type="bezier", density=0)

        # Get every new control points & give it to foil.points()
        foil.points = rotated_foil_points

        # Generate new .t file via get_mesh and get_t
        try :
            geo_path = foil.get_geo()
            msh_path = foil.get_mesh_timeout(geo_path, timeout=5)
        except Exception as e:
            raise RuntimeError(f"Unable to mesh geometry at episode {ep} : {e}") from e
        try :
            t_file_path = os.path.join(episode_root, "t", f"{name}_{ep}.t")
            foil.convert_gmsh_to_mtc(msh_path, t_file_path, False)
        except Exception as e:
            raise RuntimeError(f"Unable to build .t file at episode {ep} : {e}")

        try :
            # Copy to results/.../0/{ep}/init_feats/meshes/object.t
            tmp_t_dir = os.path.join(episode_root, "t")
            orig_t_path = os.path.join(tmp_t_dir, f"object_{ep}.t")

            meshes_dir = os.path.join(episode_root, "meshes")        
            # if os.path.isfile(meshes_dir):
            #     os.remove(meshes_dir)
            os.makedirs(meshes_dir, exist_ok=True)
            tmp_dst = os.path.join(meshes_dir, "object.t")
            final_dst = os.path.join(meshes_dir, "object.t")

            # Copy to a tmp name, then rename to avoid partially written files
            shutil.copyfile(orig_t_path, tmp_dst)
            os.rename(tmp_dst, final_dst)

        finally :
            meshes_dir = os.path.join(episode_root, "meshes")
            final_dst = os.path.join(meshes_dir, f"object.t")
            if not os.path.isfile(final_dst):
                print(f"WARNING : The final object.t is not found in {final_dst}", flush=True)
                raise FileNotFoundError(f"Failed to materialize {final_dst} at ep {ep}")
            
        # Apply deformation to the mesh if the geometry was successful.
            # Select the number of steps for the rotation:
        compute_idw_mesh(translated_foil_points, rotated_foil_points, ep, original_domain_path, episode_root, mesh_pts = translated_mesh_pts, n=1, a=4, b=2, save_t_file=True, repair_msh=False) # type: ignore
        # Run mtcexe
        cmd = (
            f'cd "{meshes_dir}" && '
            f'module load cimlibxx/master && '
            f'echo 0 | mtcexe object.t > /dev/null 2>&1'
        )
        # os.system(f"bash -lc '{cmd}'")
        # print("t_file copied and processed with mtc.")
        return foil.surface


    def run_subprocess(
        self, command: str, cwd: str, shell: bool = False, log_file: str = None
    ):
        """Run subprocess command and handle errors."""
        log_file = log_file or os.path.join(cwd, "predict.log")
        err_file = (log_file).replace(".log", ".err")
        try:
            with open(log_file, "w") as log, open(err_file, "w") as err:
                subprocess.run(
                    command, cwd=cwd, check=True, stdout=log, stderr=err, shell=shell
                )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Subprocess failed. See logs: {err_file} and {log_file}"
            ) from e

    def cleanup_directory(
        self,
        directory: str,
        keep_prediction: bool = True,
        keep_reward_data: bool = True,
    ):
        """
        Remove files and directories after the env handling is done.
        Args:
            directory (str): Directory to clean up.
            keep_prediction (bool): Whether to keep prediction files. Defaults to True.
        """
        try:
            if not os.path.exists(directory):
                return
            if not keep_prediction and not keep_reward_data:
                shutil.rmtree(directory)
                return
            # criteria
            pred_exts = (".xdmf", ".h5")
            pred_keyword = "graph"
            reward_ext = ".csv"
            keep_log_with_prediction = True  # keep predict.log when keeping predictions

            errors = []
            for name in os.listdir(directory):
                path = os.path.join(directory, name)
                try:
                    keep = False
                    # keep xdmf
                    if (
                        keep_prediction
                        and any(name.endswith(ext) for ext in pred_exts)
                        and pred_keyword in name
                    ):
                        keep = True
                    # keep csv
                    if keep_reward_data and name.endswith(reward_ext):
                        keep = True
                    # keep log
                    if (
                        keep_prediction
                        and keep_log_with_prediction
                        and name.endswith(".log")
                    ):
                        keep = True
                    if not keep:
                        if os.path.isfile(path) or os.path.islink(path):
                            os.remove(path)
                        elif os.path.isdir(path):
                            shutil.rmtree(path)
                except Exception as e:
                    errors.append(f"{path}: {e}")
            if errors:
                raise RuntimeError(
                    "cleanup_directory encountered errors:\n" + "\n".join(errors)
                )
        except Exception as e:
            raise RuntimeError("clean up failed") from e

    # Step environment
    def step(self, actions: np.ndarray, ep: int):
        """Perform an env step and return the reward."""
        conv_actions = self.convert_actions_continuous(actions)
        try:
            reward = self.env(actions, ep)
        except Exception as e:
            print(f"\n !!!!!! Step failed !!!!!!\n {e}", flush=True)
            return self.bad_rwrd, conv_actions

        return reward, conv_actions

    ### Convert actions
    def convert_actions_to_physical_scale(self, actions):
        # print("Actiosn before conversion", actions, flush=True)
        # Actions are taken in [-1;1], so transform according to expected action form in apply_xxx foil methods
        # Positive camber values (more or less concave): remap to [0.05, 1]
        actions[:4] = (0.45*actions[:4])+0.55        
        # Positive thicknesses: remap to [0.05, 1]
        actions[4:] = (0.45*actions[4:])+0.55  

        # Convert actions
        #print("Actions remapped avant physical scale : ", actions)
        physical_actions  = np.multiply(actions, self.physical_scale)
        # print("Actions converties au physical scale : ", physical_actions, flush=True)

        return physical_actions

    # Action conversion
    def convert_actions_continuous(
        self, actions: np.ndarray
    ) -> List[float]:  # for continuous actions: actions is a set of values in this case
        # Convert actions array to list
        conv_actions = actions.tolist()
        return conv_actions

    # Provide observation
    def observe(self):
        # Always return the same observation
        return self.obs

    # Close environment
    def close(self):
        pass

    def write_actions(self, actions: List[float], ep: int):
        """Write the actions to a file for the given run."""
        actions_file = os.path.abspath(os.path.join(self.path, "actions.log"))
        # Create the actions file if it doesn't exist
        if not os.path.exists(actions_file):
            with open(actions_file, "w") as f:
                act_names = "\t".join([f"a_{i}" for i in range(len(actions))])
                f.write(f"ep\t{act_names}\n")
        try:
            with open(actions_file, "a") as f:
                action_str = "\t".join(map(str, actions))
                f.write(f"{ep}\t{action_str}\n")
        except Exception as e:
            print(f"Error writing actions to file: {e}", flush=True)


class CaseParameters:
    def __init__(
        self,
        path: str = None,
        needs_cimlib: bool = True,
        geometry_class=Foil,
    ):
        """
        Initialize the case parameters and run sanity checks.
        Args:
            path (str): Path to the JSON file containing case parameters. If None, default parameters are used.
            needs_cimlib (bool): Whether the case requires CIMLIB initialization.
            geometry_class: The geometry class to be used for the case.
        """
        params = json.load(open(path)) if path is not None else self.define_params()
        self.geometry_class = geometry_class
        self.needs_cimlib_init = needs_cimlib  # if needs cimlib init

        self.trajectory_length = params["traj_parameters"].get("trajectory_length", 600)
        self.dt = params["traj_parameters"].get("dt", 0.2)
        self.dim = params.get("dim", 2)
        self.inlet_type = params["traj_parameters"].get("inlet_type", "uniform")
        self.inlet_amplitude = params["traj_parameters"].get("inlet_amplitude", 1.0)
        self.params = self.complete_params(params)

        self._sanity_checks()

    def _sanity_checks(self):
        if self.params.get("case", "") != "foil":
            raise ValueError(
                f"CaseParameters: Expected case 'foil', got '{self.params.get('case', '')}'"
            )
        if self.params.get("dim", 2) != 2:
            raise ValueError(
                f"CaseParameters: Only 2D case supported, got dim={self.params.get('dim', 2)}"
            )
        geom_params = self.params.get("geometry_parameters", {})
        required_geom_keys = ["chord", "thickness", "span", "origin", "spacing"]
        for key in required_geom_keys:
            if key not in geom_params:
                raise ValueError(f"CaseParameters: Missing geometry parameter '{key}'")
        dom_params = self.params.get("domain_parameters", {})
        required_dom_keys = ["origin_x", "origin_y", "origin_z", "dx", "dy", "dz"]
        for key in required_dom_keys:
            if key not in dom_params:
                raise ValueError(f"CaseParameters: Missing domain parameter '{key}'")
        traj_params = self.params.get("traj_parameters", {})
        required_traj_keys = ["dt", "trajectory_length", "mesh_adapt"]
        for key in required_traj_keys:
            if key not in traj_params:
                raise ValueError(
                    f"CaseParameters: Missing trajectory parameter '{key}'"
                )
        if traj_params.get("mesh_adapt", False):
            if "Hbox123" not in traj_params:
                raise ValueError(
                    "CaseParameters: 'Hbox123' must be defined in traj_parameters for mesh adaptation"
                )
            if (
                not isinstance(traj_params.get("Hbox123", []), list)
                or len(traj_params.get("Hbox123", [])) != 3
            ):
                raise ValueError(
                    "CaseParameters: 'Hbox123' must be a list of three values [Hmin, Hmax, Hgrad]"
                )
            if "driver" not in traj_params:
                raise ValueError(
                    "CaseParameters: 'driver' must be defined in traj_parameters for mesh adaptation"
                )
            if not os.path.isfile(traj_params["driver"]):
                raise ValueError(
                    f"CaseParameters: 'driver' path '{traj_params['driver']}' does not exist"
                )

    def complete_params(self, params):
        # Fill in missing traj_parameters keys with default values
        filled_traj_params = {
            "dt": self.dt,
            "trajectory_length": self.trajectory_length,
            "inlet_type": self.inlet_type,
            "inlet_amplitude": self.inlet_amplitude,
        }
        filled_traj_params.update(params.get("traj_parameters", {}))
        params["traj_parameters"] = filled_traj_params
        return params

    def define_params(self):
        return {
            "case": "foil",
            "dim": 2,
            "geometry_parameters": {
                "chord": 2,
                "thickness": 0.1,
                "span": 0,
                "origin": [0, 1.5, 0],
                "spacing": 4,
            },
            "domain_parameters": {
                "origin_x": -7,
                "origin_y": 0,
                "origin_z": 0,
                "dx": 100,
                "dy": 15,
                "dz": 0,
            },
            "traj_parameters": {
                "dt": 0.2,
                "trajectory_length": 600,
                "Hbox123": [0.005, 0.5, 2.0],
                "mesh_adapt": True,
                "number_cores": 1,
                "inlet_type": "abl",
                "inlet_amplitude": 1.0,
                "driver": "environment_config/driver/cimlib_CFD_driver",
            },
        }


class GNNParameters:
    def __init__(self, path: str):
        """Initialize the GNN parameters, and run sanity checks.
        Args:
            path (str): Path to the JSON file containing GNN parameters.
        """
        params = (
            json.load(open(path))["gnn_parameters"]
            if path is not None
            else self.define_params()
        )
        self.model_path = os.path.abspath(params.get("model_path"))
        self.traj_config_path = os.path.abspath(params.get("traj_config_path"))
        self.predict_config_path = os.path.abspath(params.get("predict_config_path"))
        self.no_edge_feature = params.get("no_edge_feature", True)

        self.sanity_checks()

    def define_params(self):
        """Define the default GNN parameters."""
        return {
            "model_path": "environment_config/models/mock_model.ckpt",
            "traj_config_path": "environment_config/trajectory_airfoil.json",
            "predict_config_path": "environment_config/predict_airfoil.json",
            "no_edge_feature": True,
        }

    def sanity_checks(self):
        if not os.path.isfile(self.model_path):
            raise ValueError(
                f"GNNParameters: model_path '{self.model_path}' does not exist"
            )
        if not os.path.isfile(self.traj_config_path):
            raise ValueError(
                f"GNNParameters: traj_config_path '{self.traj_config_path}' does not exist"
            )
        if not os.path.isfile(self.predict_config_path):
            raise ValueError(
                f"GNNParameters: predict_config_path '{self.predict_config_path}' does not exist"
            )
        if not isinstance(self.no_edge_feature, bool):
            raise ValueError("GNNParameters: no_edge_feature must be a boolean value")
