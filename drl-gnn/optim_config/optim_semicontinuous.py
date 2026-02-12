import os
import shutil
import signal
import subprocess
from typing import List

from absl import logging
import json
import numpy as np

from graphdrl.environment.geometry import Panels
from graphdrl.environment.trajectory import create_trajectory
from graphdrl.utils.reward import MinimizeAvgFy as Reward


class optim_semicontinuous:
    """Optimization configuration for PBO run using continuous action choices mapped to a discrete action space."""

    def __init__(self, path):
        """
        Initialize the optimization configuration: define environment, GNN, reward, and other parameters.
        Needed essentials by PBO: path, env_name, action_type, x_min, x_max, x_0, act_size, obs_size, obs,
        env(), step(), observe(), close()

        Args:
            path (str): Path to the optimization run directory.
        """
        # PBO params (discrete actions)
        self.path = path
        self.env_name = "panels"
        self.action_type = "continuous"
        self.action_mapping = "discrete"
        self.n_panels = 3
        self.n_actions = self.n_panels
        # Action space for continuous is [-1,1]
        self.x_min = np.array([-1.0] * self.n_actions)
        self.x_max = np.array([1.0] * self.n_actions)
        self.x_0 = np.array([0.0] * self.n_actions)
        self.action_scale_factor = 30.0  # scale factor for actions
        self.act_size = self.n_actions
        self.obs_size = self.act_size
        self.obs = np.zeros(self.obs_size)

        # discrete action space values (for discrete action conversion)
        if self.action_mapping == "discrete":
            self.action_step = 5  # step between actions in physical scale
            self.discrete_x_values = np.linspace(
                self.x_min[0] * self.action_scale_factor,
                self.x_max[0] * self.action_scale_factor,
                int(
                    (self.x_max[0] - self.x_min[0])
                    * self.action_scale_factor
                    / self.action_step
                )
                + 1,
            )

        # case params
        self.case = CaseParameters(
            path=f"environment_config/{self.env_name}.json",
            needs_cimlib=True,
            geometry_class=Panels,
        )

        # GNN params
        self.gnn_parameters = GNNParameters(
            path=f"environment_config/{self.env_name}.json"
        )
        self.graph_feature_names = {
            "velocity": ["x0", "x1"],
            "pressure": "x2",
            "levelset": "x3",
            "nodetype": "x6",
        }
        self.keep_predictions = False  # whether to keep xdmfs
        self.keep_forces = True  # whether to keep forces csv

        # Reward params
        self.RewardClass = Reward(
            start_step=300,
            bad_reward=-1e3,
            num_objects=self.n_panels,
            object_weights=[1.0] * self.n_panels,
        )

        # Misc
        print(
            f"Panels Optimization initialized. Using {self.RewardClass.reward_type} reward.",
            flush=True,
        )  # TODO: improve message with info at initialization
        logging.set_verbosity(logging.ERROR)

    def env(self, x: np.ndarray, ep: int):
        """Run the GNN prediction for the given environment ep with the given actions x."""
        try:
            # Step 1: setup env folder
            self.setup_env_folder(ep=ep)
            if self.action_mapping == "discrete":
                x = self.map_to_discrete_actions(continuous_actions=x)
            if self.action_type == "continuous":
                x = self.scale_actions_to_physical(actions=x)
            self.write_actions(actions=x, ep=ep)

            # Step 2: create trajectory
            self.create_env_trajectory(actions=x, ep=ep)

            # Step 3: run gnn prediction
            self.gnn_prediction()
        except Exception as e:
            logging.error(f"Environment {ep} failed (fatal): {e}", exc_info=True)
            os.killpg(os.getpgrp(), signal.SIGTERM)  # kill parent + child
        try:
            # Step 4: compute reward
            self.reward = self.RewardClass.compute_reward(
                ep=ep,
                xdmf_path=os.path.join(self.output_path, f"graph_{ep}.xdmf"),
                feature_names=self.graph_feature_names,
                save_data=self.keep_forces,
            )
            print(f"Episode {ep} reward: {self.reward:.6f}", flush=True)

            # Step 5: cleanup
            self.cleanup_directory(
                f"{self.output_path}",
                keep_prediction=self.keep_predictions,
                keep_reward_data=self.keep_forces,
            )

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

    def create_env_trajectory(self, actions: List[float], ep: int):
        """Create the trajectory file based on the actions."""
        # make sure actions are valid
        if not actions or len(actions) != self.n_actions:
            raise ValueError("Invalid actions for trajectory creation.")
        # make sure actions are list of floats
        if not all(isinstance(act, (float, int)) for act in actions):
            raise ValueError(
                f"Actions must be a list of floats or ints, not {type(actions)}."
            )
        # Create geometry object
        geometry_args = {
            "num_panels": self.n_panels,
            "angles": actions,
            "dim": self.case.dim,
        }

        # Create trajectory
        try:
            _ = create_trajectory(
                path=self.output_path,
                parameters=self.case.params,
                geometry_class=self.case.geometry_class,
                geometry_args=geometry_args,
                init_features=self.case.needs_cimlib_init,
                output_name=os.path.join(
                    self.output_path, f"{self.env_name}_{ep}.xdmf"
                ),
            )
        except Exception as e:
            raise RuntimeError(f"Trajectory creation failed: {e}")
        # note: handle full traj by converting to graph and directly feed to GNN?

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

    def map_to_discrete_actions(self, continuous_actions: np.ndarray) -> List[float]:
        """
        Map continuous actions values (rescaled to physical values) to closest discrete actions values,
        by finding the index of the closest discrete value for each continuous action.
        Args:
            continuous_actions (np.ndarray): Array of continuous action values.
        Returns:
            List[int]: List of discrete action values."""
        cont = np.asarray(continuous_actions).ravel()
        cont = cont * self.action_scale_factor  # remap to physical scale
        if cont.size != self.n_actions:
            raise ValueError(f"Expected {self.n_actions} actions, got {cont.size}")
        indices = [
            int(np.argmin(np.abs(self.discrete_x_values - float(a)))) for a in cont
        ]
        discrete_values = self.discrete_x_values[indices].tolist()
        return [float(v) for v in discrete_values]

    def scale_actions_to_physical(self, actions: np.ndarray) -> List[float]:
        """
        Scale continuous action values to physical values using the action scale factor.
        Args:
            actions (np.ndarray): Array of continuous action values.
        Returns:
            List[float]: List of scaled action values.
        """
        cont = np.asarray(actions).ravel()
        if cont.size != self.n_actions:
            raise ValueError(f"Expected {self.n_actions} actions, got {cont.size}")
        scaled_actions = (cont * self.action_scale_factor).tolist()
        return [float(v) for v in scaled_actions]

    # Step environment
    def step(self, actions: np.ndarray, ep: int):
        """Perform an env step and return the reward."""
        conv_actions = (
            self.convert_actions_discrete(actions)
            if self.action_type == "discrete"
            else self.convert_actions_continuous(actions)
        )
        try:
            reward = self.env(conv_actions, ep)
        except Exception as e:
            print(f"\n !!!!!! Step failed !!!!!!\n {e}", flush=True)
            return self.reward_parameters.bad_reward, conv_actions

        return reward, conv_actions

    # Action conversion
    def convert_actions_continuous(
        self, actions: np.ndarray
    ) -> List[float]:  # for continuous actions: actions is a set of values in this case
        # Convert actions array to list
        conv_actions = actions.tolist()
        return conv_actions

    def convert_actions_discrete(
        self, actions: np.ndarray
    ) -> List[
        float
    ]:  # overwrite for discrete actions: actions is a set of indices in this case
        conv_actions = len(actions) * [None]
        for i in range(len(actions)):
            conv_actions[i] = self.x_values[i][int(actions[i])]
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
        geometry_class=Panels,
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
        if self.params.get("case", "") != "panels":
            raise ValueError(
                f"CaseParameters: Expected case 'panels', got '{self.params.get('case', '')}'"
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
            "case": "panels",
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
            "traj_config_path": "environment_config/trajectory_panels.json",
            "predict_config_path": "environment_config/predict_panels.json",
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
