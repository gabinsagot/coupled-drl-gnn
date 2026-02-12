import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import meshio
import numpy as np
import pandas as pd
from tqdm import tqdm

from graphdrl.utils.file_handling import gather_cases, gather_cases_for_reward_recursive
from graphdrl.utils.forces import (
    compute_forces_from_xdmf,
    build_object_containers,
)
from graphdrl.utils.meshio_mesh import xdmf_to_meshes


class Reward:
    def __init__(
        self,
        reward_type="Default",
        stretch_factor: float | None = None,
        stretch_type: str = "lin",
    ):
        self.reward_type = reward_type
        self.stretch_factor = stretch_factor
        self.stretch_type = stretch_type.lower()
        # sanity check
        if self.stretch_factor is not None:
            if not isinstance(self.stretch_factor, (int, float)):
                raise ValueError("stretch_factor must be a number (int or float).")
            if self.stretch_factor <= 0:
                raise ValueError("stretch_factor must be positive.")
            if self.stretch_type not in ["lin", "tanh", "pow", "log"]:
                raise ValueError(
                    "stretch_type must be one of ['lin', 'tanh', 'pow', 'log']."
                )

    def compute_reward(self, ep: int, **kwargs):
        raise NotImplementedError("compute_reward() must be implemented by subclasses")

    def contrast_stretch(
        self,
        reward: float,
    ) -> float:
        """
        Apply contrast stretching to the reward if stretch_factor is set. Uses a tanh-based scaling
        to map rewards to the range [-1, 1]: negative rewards are mapped to [-1, 0] and positive rewards to [0, 1].
        Not recommended for maximizing positive rewards since they will be squashed towards 1, use on minimization
        optimizations (maximizing negative rewards) instead.
        The stretch_factor controls the steepness of the tanh curve: higher values lead to more
        pronounced stretching near 0.

        Contrast functions used:
            lin: r' = r * k, where k is the stretch_factor.
            tanh: r' = tanh(k r) / tanh(k), where k is the stretch_factor.
            power-based: r' = sign(r) * |r|^k
            log: r' = log(1 + k * |r|) * sign(r) / log(1 + k)

        Args:
            reward (float): Original reward value.
            stretch_type (str): Type of contrast stretching to apply. Currently only 'tanh' and 'pow' are supported.
        Returns:
            float: Contrast-stretched reward value.
        """
        if self.stretch_factor is None:
            return reward
        k = self.stretch_factor
        r = np.asarray(reward)
        if self.stretch_type == "lin":
            # linear scaling
            stretched = r * k
        elif self.stretch_type == "pow":
            # power-based contrast stretching
            stretched = np.sign(r) * (np.abs(r) ** k)
        elif self.stretch_type == "log":
            # log-based contrast stretching
            stretched = np.log1p(1 + k * np.abs(r)) * np.sign(r) / np.log1p(k)
        elif self.stretch_type == "tanh":
            # tanh-based contrast stretching
            stretched = np.tanh(k * r) / np.tanh(k)
        if np.isscalar(reward) or r.shape == ():
            return float(stretched)
        return stretched


class VelocityFluctuationReward(Reward):
    def __init__(
        self,
        start_step: int = 300,
        bad_reward: float = -1e3,
        reward_boxes: dict | str = None,
        stretch_factor: float | None = None,
        stretch_type: str = "lin",
    ):
        """
        Initialize the reward parameters and run sanity checks.
        Args:
            start_step (int): Timestep number to start averaging velocity fields from.
            bad_reward (float): Reward to return in case of failure. Must be negative.
            reward_boxes (dict|str): Dictionary or path to json defining reward boxes. If None, default boxes are used.
        """
        super().__init__(
            reward_type="VelocityFluctuation",
            stretch_factor=stretch_factor,
            stretch_type=stretch_type,
        )
        if isinstance(reward_boxes, str):
            import json

            with open(reward_boxes, "r") as f:
                reward_boxes = json.load(f)
        self.reward_boxes = (
            reward_boxes if reward_boxes is not None else self._define_boxes()
        )
        self.start_step = start_step
        self.bad_reward = bad_reward

        self._sanity_checks()

    def compute_reward(
        self, ep: int, xdmf_path: str, velocity_field_keys: str | List[str]
    ) -> float:
        """
        Compute the reward for a given case based on the velocity fluctuations
        within the defined reward boxes.

        Args:
            ep (int): Episode number.
            xdmf_path (str): Path to the XDMF file of the case.
            velocity_field_keys (str | List[str]): Name of the velocity field component(s) in the mesh point data.
        Returns:
            reward (float): Computed reward value.
        """
        reward = self._compute_reward_case(
            case_xdmf_path=xdmf_path,
            velocity_field=velocity_field_keys,
            start_step=self.start_step,
            box_dict=self.reward_boxes,
            verbose=False,
        )
        return self.contrast_stretch(reward)

    def _sanity_checks(self):
        if not isinstance(self.start_step, int) or self.start_step < 0:
            raise ValueError("start_step must be a non-negative integer.")
        if not isinstance(self.bad_reward, (int, float)) or self.bad_reward >= 0:
            raise ValueError("bad_reward must be a negative number.")
        if not isinstance(self.reward_boxes, dict) or not self.reward_boxes:
            raise ValueError("reward_boxes must be a non-empty dictionary.")
        for box_name, box in self.reward_boxes.items():
            required_keys = ["x_min", "y_min", "dx", "dy"]  # 'weight' can be optional
            if not isinstance(box, dict):
                raise ValueError(f"Box '{box_name}' must be a dictionary.")
            if not all(key in box.keys() for key in required_keys):
                raise ValueError(
                    f"Box '{box_name}' must contain keys: {required_keys}, "
                    f"but is missing {set(required_keys) - set(box.keys())}."
                )
            if not all(isinstance(box[key], (int, float)) for key in required_keys):
                raise ValueError(
                    f"All values in box '{box_name}' must be numbers (int or float)."
                )
            if box["dx"] <= 0 or box["dy"] <= 0:
                raise ValueError(
                    f"Box '{box_name}' dimensions 'dx' and 'dy' must be positive."
                )

    def _define_boxes() -> Dict[str, Dict[str, float]]:
        return {
            "box0": {"weight": 1.0, "x_min": 1.25, "y_min": 0.5, "dx": 1.5, "dy": 3.5},
            "box1": {"weight": 1.0, "x_min": 5.25, "y_min": 0.5, "dx": 1.5, "dy": 3.5},
            "box2": {"weight": 1.0, "x_min": 9.25, "y_min": 0.5, "dx": 1.5, "dy": 3.5},
            "box3": {"weight": 1.0, "x_min": 13.25, "y_min": 0.5, "dx": 1.5, "dy": 3.5},
            "box4": {"weight": 1.0, "x_min": 17.25, "y_min": 0.5, "dx": 1.5, "dy": 3.5},
            "box5": {"weight": 1.0, "x_min": 21.25, "y_min": 0.5, "dx": 9.0, "dy": 3.5},
        }

    def _create_single_box_mask(
        self,
        mesh: meshio.Mesh,
        box_dict: Dict[str, float],
    ) -> np.ndarray:
        """
        Create a mask for a single reward box in the mesh based on the box dimensions.
        Args:
            mesh: meshio Mesh object (a single timestep snapshot).
            box_dict: dictionary of box dimensions::
                {
                    "x_min": float,
                    "y_min": float,
                    "dx": float,
                    "dy": float
                }
        Returns:
            mask: boolean array of shape (num_points,) where True indicates the point
            is inside the box and False otherwise.
        """
        # Extract box parameters
        x0 = box_dict["x_min"]
        y0 = box_dict["y_min"]
        dx = box_dict["dx"]
        dy = box_dict["dy"]

        # Get the points from the mesh
        points = mesh.points

        # Create a mask for the points inside the box
        mask = (
            (points[:, 0] >= x0)
            & (points[:, 0] <= x0 + dx)
            & (points[:, 1] >= y0)
            & (points[:, 1] <= y0 + dy)
        )
        return mask

    def _create_reward_box_mask(
        self,
        mesh: meshio.Mesh,
        box_dict: Dict[str, Any],
    ) -> np.ndarray:
        """
        Create a mask for the reward box in the mesh based on the box dimensions.

        Args:
            mesh: meshio Mesh object (a single timestep snapshot).
            box_dict: dictionary of box dimensions::
                {
                "box_name": {
                    "x_min": float,
                    "y_min": float,
                    "dx": float,
                    "dy": float
                }

        Returns:
            mask: boolean array of shape (num_points,) where True indicates the point
            is inside the union of all boxes defio boned in box_dict and False otherwise.
        """
        # Initialize mask to False
        mask = np.zeros(mesh.points.shape[0], dtype=bool)
        # Loop over all boxes and update mask
        for box_name, box_dict in box_dict.items():
            box_mask = self._create_single_box_mask(mesh, box_dict)
            mask = mask | box_mask
        return mask

    def _compute_time_avg_velocity_components(
        self,
        meshes: List[meshio.Mesh],
        velocity_field: str | List[str] = "Vitesse",
        start_step: int = 0,
        mask: np.ndarray = None,
    ) -> Tuple[float, float]:
        """
        Compute the average velocity components (Vx and Vy) in the mesh, optionally within a specified box mask.

        Args:
            meshes: list of meshio Mesh objects (each representing a single timestep snapshot).
            velocity_field: name of the velocity field in the mesh point data, or list of names
            of the components of the velocity vector.
            start_step: timestep index to start the averaging from (default is 0).
            mask: optional boolean array of shape (num_points,) where True indicates the point is inside the box.

        Returns:
            avg_vx: np.ndarray representing the average Vx over time for each node (shape (num_nodes,)).
            avg_vy: np.ndarray representing the average Vy over time for each node (shape (num_nodes,)).
        """
        vx_list = []
        vy_list = []
        # sanity checks
        if (
            isinstance(velocity_field, str)
            and velocity_field not in meshes[0].point_data
        ):
            raise KeyError(f"Field {velocity_field} not found in mesh")
        if isinstance(velocity_field, list):
            for field in velocity_field:
                if field not in meshes[0].point_data:
                    raise KeyError(f"Field {field} not found in mesh")
        # process
        for mesh in meshes[start_step:]:
            if isinstance(velocity_field, str):
                velocity = np.array(mesh.point_data[velocity_field])
                vx = velocity[:, 0]
                vy = velocity[:, 1]
            else:
                vx = np.array(mesh.point_data[velocity_field[0]])
                vy = np.array(mesh.point_data[velocity_field[1]])
            if mask is not None:
                vx = vx[mask]
                vy = vy[mask]
            vx_list.append(vx)
            vy_list.append(vy)
        avg_vx = np.mean(np.array(vx_list), axis=0)
        avg_vy = np.mean(np.array(vy_list), axis=0)
        return avg_vx, avg_vy

    def _compute_velocity_rms_fluctuation(
        self,
        meshes: List[meshio.Mesh],
        velocity_field: str | List[str] = "Vitesse",
        start_step: int = 0,
        mask_fluc: np.ndarray = None,
        mask_avg: np.ndarray = None,
        weights: np.ndarray = None,
    ) -> float:
        """
        Compute the velocity fluctuations (RMS of velocity magnitude fluctuations)
        in the mesh, optionally within a specified box mask.

        Args:
            meshes: list of meshio Mesh objects (each representing a single timestep snapshot).
            velocity_field: name of the velocity field in the mesh point data, or list of names of
            the components of the velocity vector.
            start_step: timestep index to start the computation from (default is 0).
            mask: optional boolean array of shape (num_points,) where True indicates the point is inside the box.
            mask_avg: optional boolean array of shape (num_points,) used to compute the average velocity over timesteps.
            weights: optional array of shape (num_points,) representing weights (by box) for each point in the mesh.

        Returns:
            rms_fluctuation: float representing the RMS of velocity magnitude
            fluctuations over all nodes over timesteps.
        """
        # avg velocity
        avg_vx, avg_vy = self._compute_time_avg_velocity_components(
            meshes=meshes,
            velocity_field=velocity_field,
            start_step=start_step,
            mask=mask_avg,
        )
        if mask_fluc is not None:
            avg_vx = avg_vx[mask_fluc]
            avg_vy = avg_vy[mask_fluc]

        # fluctuations
        rms_time_fluc_list = []
        for mesh in meshes[start_step:]:
            if isinstance(velocity_field, str):
                velocity = np.array(mesh.point_data[velocity_field])
                vx = velocity[:, 0]
                vy = velocity[:, 1]
            else:
                vx = np.array(mesh.point_data[velocity_field[0]])
                vy = np.array(mesh.point_data[velocity_field[1]])
            if mask_fluc is not None:
                vx = vx[mask_fluc]
                vy = vy[mask_fluc]
            fluc_vx = vx - avg_vx
            fluc_vy = vy - avg_vy
            fluc = fluc_vx**2 + fluc_vy**2
            rms_time_fluc_list.append(fluc)
        all_flucs = np.concatenate(rms_time_fluc_list)
        rms_fluctuation = np.mean(all_flucs, axis=0) ** 0.5
        if weights is not None:
            if mask_fluc is not None:
                weights = weights[mask_fluc]
            rms_fluctuation *= weights
        rms = np.mean(rms_fluctuation)
        return rms

    def _compute_reward_case(
        self,
        case_xdmf_path: str,
        box_dict: Dict[str, Any],
        velocity_field: str | List[str] = "Vitesse",
        start_step: int = 0,
        verbose: bool = False,
    ) -> float:
        """
        Compute the reward for a single case based on the provided parameters.

        Args:
            case_xdmf_path: path to the XDMF file of the case.
            box_dict: dictionary with boxes and their parameters in JSON format:
                "box_name": {
                    "x_min": float,
                    "y_min": float,
                    "dx": float,
                    "dy": float
                }
            velocity_field: str | List[str] (optional, default "Vitesse"), name of the velocity field in
            the mesh point data, or list of names of the components of the velocity vector.
            start_step: int (optional, default 0), timestep index to start the computation from.
            verbose: if True, print detailed information during processing.

        Returns:
            reward: float representing the computed reward for the case.
        """
        # Load meshes
        meshes, times = xdmf_to_meshes(case_xdmf_path, verbose=verbose)
        if len(meshes) == 0:
            raise ValueError(f"No meshes found in {case_xdmf_path}")

        # Create mask for reward box
        mask_box = self._create_reward_box_mask(meshes[0], box_dict)

        # create weighted mask if weights are provided in box_dict
        if all("weight" in box for box in box_dict.values()):
            # total_weight = sum(box["weight"] for box in box_dict.values())
            # if total_weight <= 0:
            #     raise ValueError("Total weight of boxes must be positive")
            weights = np.zeros(meshes[0].points.shape[0])
            for box in box_dict.values():
                box_mask = self._create_single_box_mask(mesh=meshes[0], box_dict=box)
                weights[box_mask] += box.get("weight", 1.0)  # / total_weight
                # TODO: normalize weights accross nodes for smoother final avg over nodes
        else:
            weights = None

        # Create mask for average velocity computation (optional)
        mask_avg = None

        # Compute RMS of velocity fluctuations in the box
        rms_fluctuation = self._compute_velocity_rms_fluctuation(
            meshes=meshes,
            velocity_field=velocity_field,
            start_step=start_step,
            mask_fluc=mask_box,
            mask_avg=mask_avg,
            weights=weights,
        )

        # Compute reward
        reward = -rms_fluctuation
        if verbose:
            print(f"Computed reward for case {case_xdmf_path}: {reward}")
        return reward

    def _compare_rewards_cases(
        self,
        case_folder: str,
        case_base_name: str,
        box_dict: Dict[str, Any],
        velocity_field: str | List[str] = "Vitesse",
        start_step: int = 0,
        output_csv: str = None,
        verbose: bool = False,
        compare: bool = False,
        target_field: str | List[str] = "V_vect_targ",
    ) -> pd.DataFrame:
        """
        Compare rewards for all cases in a folder and save results to a CSV file.

        Args:
            case_folder: path to the folder containing case XDMF files.
            case_base_name: base name of the case files.
            box_dict: dictionary with box parameters::
                {"box_dimensions": {"x_min": float,
                                    "y_min": float,
                                    "dx": float,
                                    "dy": float},
                }
            start_step: int (optional, default 0), timestep index to start the computation from.
            velocity_field: str or list of str (optional, default "Vitesse"), name of the velocity field in
            the mesh point data, or list of names of the components of the velocity vector.
            output_csv: path to save the output CSV file (optional).
            verbose: if True, print detailed information during processing.
            compare: if True, compute CFD reward on target field for comparison.
            target_field: str or list of str (optional, default "V_vect_targ"), name of the target velocity field in
            the mesh point data, or list of names of the components of the target velocity vector.
        Returns:
            DataFrame with case names as rows and their corresponding rewards as columns.
        """
        # Initialize
        cases = gather_cases(case_folder, case_base_name)
        if len(cases) == 0:
            raise ValueError(
                f"No cases found in {case_folder} with base name {case_base_name}"
            )
        results = []

        # Process each case
        for case_name, case_xdmf_path in tqdm(
            cases.items(), desc="Processing cases", disable=not verbose
        ):
            reward = self._compute_reward_case(
                case_xdmf_path=case_xdmf_path,
                box_dict=box_dict,
                velocity_field=velocity_field,
                start_step=start_step,
                verbose=False,
            )
            if compare:
                reward_truth = self._compute_reward_case(
                    case_xdmf_path=case_xdmf_path,
                    box_dict=box_dict,
                    velocity_field=target_field,
                    start_step=start_step,
                    verbose=False,
                )
                results.append(
                    {
                        "case": case_name,
                        "reward_gnn": reward,
                        "reward_cfd": reward_truth,
                    }
                )
            else:
                results.append({"case": case_name, "reward": reward})

        # Create DataFrame
        df = pd.DataFrame(results)

        # Save to CSV if output path is provided
        if output_csv:
            df.to_csv(output_csv, index=False)

        return df

    def _plot_comparison_rewards(df: pd.DataFrame, output_path: str):
        """
        Plot comparison of rewards between GNN and CFD.
        """

        plt.figure(figsize=(10, 4))
        x = np.arange(len(df["case"]))
        width = 0.4
        plt.bar(
            x - width / 2,
            df["reward_gnn"],
            width,
            label="GNN",
            color="royalblue",
        )
        plt.bar(
            x + width / 2,
            df["reward_cfd"],
            width,
            label="CFD",
            color="darkgreen",
        )
        plt.xticks(x, df["case"], rotation=45)
        plt.xlabel("Case")
        plt.ylabel("Reward")
        plt.title("Comparison of Rewards")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


class MinimizeAvgFx(Reward):
    def __init__(
        self,
        start_step: int = 300,
        bad_reward: float = -1e3,
        num_objects: int = 6,
        object_weights: List[float] = None,
        stretch_factor: float | None = None,
        stretch_type: str = "lin",
    ):
        """
        Initialize the MinimizeAvgFx reward.
        """
        super().__init__(
            reward_type="MinimizeAvgFx",
            stretch_factor=stretch_factor,
            stretch_type=stretch_type,
        )
        self.bad_reward = bad_reward
        self.start_step = start_step
        self.num_objects = num_objects

        # individual object weights in force reward
        if object_weights is not None and len(object_weights) != num_objects:
            raise ValueError("Length of object_weights must match num_objects.")
        self.object_weights = (
            object_weights if object_weights is not None else [1.0] * num_objects
        )

        # panels origin in [0,1.5,0], chord=2.0, spacing=4.0
        self.object_containers = self._build_object_containers(
            n_obj=num_objects, x0=0, y0=1.5, chord=2.0, spacing=4.0
        )

        self.mu = 1e-3  # dynamic viscosity

    def _build_object_containers(
        self, n_obj: int, x0: float, y0: float, chord: float, spacing: float
    ) -> dict | None:
        """
        Create object container boxes dict for the force computation.
        """
        return build_object_containers(
            n_obj=n_obj, x0=x0, y0=y0, chord=chord, spacing=spacing
        )

    def compute_reward(
        self,
        ep: int,
        xdmf_path: str,
        feature_names: dict = {},
        save_data: bool = True,
        load_data: bool = False,
    ) -> float:
        """
        Compute the reward for a given case based on the average abs drag force.

        Args:
            ep (int): Episode number.
            xdmf_path (str): Path to the XDMF file of the case.
            feature_names (dict): Dictionary mapping feature names used in the XDMF file.
            Needs to include keys for velocity, pressure, and nodetype.
            save_data (bool): Whether to save computed forces to a CSV file.
            load_data (bool): Whether to load forces from a CSV file instead of computing from XDMF.
        Returns:
            reward (float): Computed reward value.
        """
        if load_data:
            # load forces from csv file
            forces_file = os.path.abspath(
                os.path.join(os.path.dirname(xdmf_path), f"forces_{ep}.csv")
            )
            if not os.path.exists(forces_file):
                raise ValueError(f"forces file {forces_file} does not exist.")
            forces_df = pd.read_csv(forces_file)
        else:
            forces_df = compute_forces_from_xdmf(
                case_xdmf_path=xdmf_path,
                mu=self.mu,
                feature_names=feature_names,
                start_step=self.start_step,
                object_containers=self.object_containers,
                verbose=False,
            )
        # save forces to file
        if save_data and not load_data:
            forces_file = os.path.abspath(
                os.path.join(os.path.dirname(xdmf_path), f"forces_{ep}.csv")
            )
            if not os.path.exists(forces_file):
                forces_df.to_csv(forces_file, index=False)

        # actual reward computation
        reward = 0
        for obj_id, obj_weight in enumerate(self.object_weights):
            reward -= obj_weight * np.abs(
                forces_df[forces_df["Object"] == obj_id]["Fx"].mean()
            )
        return self.contrast_stretch(reward)


class MinimizeL2AvgFx(MinimizeAvgFx):
    def __init__(
        self,
        start_step: int = 300,
        bad_reward: float = -1e3,
        num_objects: int = 6,
        object_weights: List[float] = None,
        stretch_factor: float | None = None,
        stretch_type: str = "lin",
    ):
        """
        Initialize the MinimizeL2AvgFx reward.
        """
        super().__init__(
            start_step=start_step,
            bad_reward=bad_reward,
            num_objects=num_objects,
            object_weights=object_weights,
            stretch_factor=stretch_factor,
            stretch_type=stretch_type,
        )
        self.reward_type = "MinimizeL2AvgFx"

    def compute_reward(
        self,
        ep: int,
        xdmf_path: str,
        feature_names: dict = {},
        save_data: bool = True,
        load_data: bool = False,
    ) -> float:
        """
        Compute the reward for a given case based on the average L2 drag force.

        Args:
            ep (int): Episode number.
            xdmf_path (str): Path to the XDMF file of the case.
            feature_names (dict): Dictionary mapping feature names used in the XDMF file.
            Needs to include keys for velocity, pressure, and nodetype.
            save_data (bool): Whether to save computed forces to a CSV file.
            load_data (bool): Whether to load forces from a CSV file instead of computing from XDMF.
        Returns:
            reward (float): Computed reward value.
        """
        if load_data:
            # load forces from csv file
            forces_file = os.path.abspath(
                os.path.join(os.path.dirname(xdmf_path), f"forces_{ep}.csv")
            )
            if not os.path.exists(forces_file):
                raise ValueError(f"forces file {forces_file} does not exist.")
            forces_df = pd.read_csv(forces_file)
        else:
            forces_df = compute_forces_from_xdmf(
                case_xdmf_path=xdmf_path,
                mu=self.mu,
                feature_names=feature_names,
                start_step=self.start_step,
                object_containers=self.object_containers,
                verbose=False,
            )
        # save forces to file
        if save_data and not load_data:
            forces_file = os.path.abspath(
                os.path.join(os.path.dirname(xdmf_path), f"forces_{ep}.csv")
            )
            if not os.path.exists(forces_file):
                forces_df.to_csv(forces_file, index=False)

        # actual reward computation
        reward = 0
        for obj_id, obj_weight in enumerate(self.object_weights):
            fx_mean = forces_df[forces_df["Object"] == obj_id]["Fx"].mean()
            reward += obj_weight * (fx_mean**2)
        reward = -np.sqrt(reward)
        return self.contrast_stretch(reward)


class MaximizeAvgFx(MinimizeAvgFx):
    def __init__(
        self,
        start_step: int = 300,
        bad_reward: float = -1e3,
        num_objects: int = 6,
        object_weights: List[float] = None,
        stretch_factor: float | None = None,
        stretch_type: str = "lin",
    ):
        """
        Initialize the MaximizeAvgFx reward.
        """
        super().__init__(
            start_step=start_step,
            bad_reward=bad_reward,
            num_objects=num_objects,
            object_weights=object_weights,
            stretch_factor=stretch_factor,
            stretch_type=stretch_type,
        )
        self.reward_type = "MaximizeAvgFx"

    def compute_reward(
        self,
        ep: int,
        xdmf_path: str,
        feature_names: dict = {},
        save_data: bool = True,
        load_data: bool = False,
    ) -> float:
        """
        Compute the reward for a given case based on the average drag force.

        Args:
            ep (int): Episode number.
            xdmf_path (str): Path to the XDMF file of the case.
            feature_names (dict): Dictionary mapping feature names used in the XDMF file.
            Needs to include keys for velocity, pressure, and nodetype.
            save_data (bool): Whether to save computed forces to a CSV file.
            load_data (bool): Whether to load forces from a CSV file instead of computing from XDMF.
        Returns:
            reward (float): Computed reward value.
        """
        if load_data:
            # load forces from csv file
            forces_file = os.path.abspath(
                os.path.join(os.path.dirname(xdmf_path), f"forces_{ep}.csv")
            )
            if not os.path.exists(forces_file):
                raise ValueError(f"forces file {forces_file} does not exist.")
            forces_df = pd.read_csv(forces_file)
        else:
            forces_df = compute_forces_from_xdmf(
                case_xdmf_path=xdmf_path,
                mu=self.mu,
                feature_names=feature_names,
                start_step=self.start_step,
                object_containers=self.object_containers,
                verbose=False,
            )
        # save forces to file
        if save_data and not load_data:
            forces_file = os.path.abspath(
                os.path.join(os.path.dirname(xdmf_path), f"forces_{ep}.csv")
            )
            if not os.path.exists(forces_file):
                forces_df.to_csv(forces_file, index=False)

        # actual reward computation
        reward = 0
        for obj_id, obj_weight in enumerate(self.object_weights):
            reward += obj_weight * forces_df[forces_df["Object"] == obj_id]["Fx"].mean()
        return self.contrast_stretch(reward)


class MinimizeAvgFy(MinimizeAvgFx):
    def __init__(
        self,
        start_step: int = 300,
        bad_reward: float = -1e3,
        num_objects: int = 6,
        object_weights: List[float] = None,
        stretch_factor: float | None = None,
        stretch_type: str = "lin",
    ):
        """
        Initialize the MinimizeAvgFy reward.
        """
        super().__init__(
            start_step=start_step,
            bad_reward=bad_reward,
            num_objects=num_objects,
            object_weights=object_weights,
            stretch_factor=stretch_factor,
            stretch_type=stretch_type,
        )
        self.reward_type = "MinimizeAvgFy"

    def compute_reward(
        self,
        ep: int,
        xdmf_path: str,
        feature_names: dict = {},
        save_data: bool = True,
        load_data: bool = False,
    ) -> float:
        """
        Compute the reward for a given case based on the average absolute y-force.

        Args:
            ep (int): Episode number.
            xdmf_path (str): Path to the XDMF file of the case.
            feature_names (dict): Dictionary mapping feature names used in the XDMF file.
            Needs to include keys for velocity, pressure, and nodetype.
            save_data (bool): Whether to save computed forces to a CSV file.
            load_data (bool): Whether to load forces from a CSV file instead of computing from XDMF.
        Returns:
            reward (float): Computed reward value.
        """
        if load_data:
            # load forces from csv file
            forces_file = os.path.abspath(
                os.path.join(os.path.dirname(xdmf_path), f"forces_{ep}.csv")
            )
            if not os.path.exists(forces_file):
                raise ValueError(f"forces file {forces_file} does not exist.")
            forces_df = pd.read_csv(forces_file)
        else:
            forces_df = compute_forces_from_xdmf(
                case_xdmf_path=xdmf_path,
                mu=self.mu,
                feature_names=feature_names,
                start_step=self.start_step,
                object_containers=self.object_containers,
                verbose=True,
            )
        # save forces to file
        if save_data and not load_data:
            forces_file = os.path.abspath(
                os.path.join(os.path.dirname(xdmf_path), f"forces_{ep}.csv")
            )
            if not os.path.exists(forces_file):
                forces_df.to_csv(forces_file, index=False)

        # actual reward computation
        reward = 0
        for obj_id, obj_weight in enumerate(self.object_weights):
            reward -= obj_weight * np.abs(
                forces_df[forces_df["Object"] == obj_id]["Fy"].mean()
            )
        return self.contrast_stretch(reward)


class MinimizeL2AvgFy(MinimizeAvgFx):
    def __init__(
        self,
        start_step: int = 300,
        bad_reward: float = -1e3,
        num_objects: int = 6,
        object_weights: List[float] = None,
        stretch_factor: float | None = None,
        stretch_type: str = "lin",
    ):
        """
        Initialize the MinimizeL2AvgFy reward.
        """
        super().__init__(
            start_step=start_step,
            bad_reward=bad_reward,
            num_objects=num_objects,
            object_weights=object_weights,
            stretch_factor=stretch_factor,
            stretch_type=stretch_type,
        )
        self.reward_type = "MinimizeL2AvgFy"

    def compute_reward(
        self,
        ep: int,
        xdmf_path: str,
        feature_names: dict = {},
        save_data: bool = True,
        load_data: bool = False,
    ) -> float:
        """
        Compute the reward for a given case based on the average L2 y-force.

        Args:
            ep (int): Episode number.
            xdmf_path (str): Path to the XDMF file of the case.
            feature_names (dict): Dictionary mapping feature names used in the XDMF file.
            Needs to include keys for velocity, pressure, and nodetype.
            save_data (bool): Whether to save computed forces to a CSV file.
            load_data (bool): Whether to load forces from a CSV file instead of computing from XDMF.
        Returns:
            reward (float): Computed reward value.
        """
        if load_data:
            # load forces from csv file
            forces_file = os.path.abspath(
                os.path.join(os.path.dirname(xdmf_path), f"forces_{ep}.csv")
            )
            if not os.path.exists(forces_file):
                raise ValueError(f"forces file {forces_file} does not exist.")
            forces_df = pd.read_csv(forces_file)
        else:
            forces_df = compute_forces_from_xdmf(
                case_xdmf_path=xdmf_path,
                mu=self.mu,
                feature_names=feature_names,
                start_step=self.start_step,
                object_containers=self.object_containers,
                verbose=False,
            )
        # save forces to file
        if save_data and not load_data:
            forces_file = os.path.abspath(
                os.path.join(os.path.dirname(xdmf_path), f"forces_{ep}.csv")
            )
            if not os.path.exists(forces_file):
                forces_df.to_csv(forces_file, index=False)

        # actual reward computation
        reward = 0
        for obj_id, obj_weight in enumerate(self.object_weights):
            fy_mean = forces_df[forces_df["Object"] == obj_id]["Fy"].mean()
            reward += obj_weight * (fy_mean**2)
        reward = -np.sqrt(reward)
        return self.contrast_stretch(reward)


class MaximizeAvgLift(MinimizeAvgFy):
    def __init__(
        self,
        start_step: int = 300,
        bad_reward: float = -1e3,
        num_objects: int = 6,
        object_weights: List[float] = None,
        stretch_factor: float | None = None,
        stretch_type: str = "lin",
    ):
        """
        Initialize the MaximizeAvgLift reward.
        """
        super().__init__(
            start_step=start_step,
            bad_reward=bad_reward,
            num_objects=num_objects,
            object_weights=object_weights,
            stretch_factor=stretch_factor,
            stretch_type=stretch_type,
        )
        self.reward_type = "MaximizeAvgLift"

    def compute_reward(
        self,
        ep: int,
        xdmf_path: str,
        feature_names: dict = {},
        save_data: bool = True,
        load_data: bool = False,
    ) -> float:
        """
        Compute the reward for a given case based on the average lift force.

        Args:
            ep (int): Episode number.
            xdmf_path (str): Path to the XDMF file of the case.
            feature_names (dict): Dictionary mapping feature names used in the XDMF file.
            Needs to include keys for velocity, pressure, and nodetype.
            save_data (bool): Whether to save computed forces to a CSV file.
            load_data (bool): Whether to load forces from a CSV file instead of computing from XDMF.
        Returns:
            reward (float): Computed reward value.
        """
        if load_data:
            # load forces from csv file
            forces_file = os.path.abspath(
                os.path.join(os.path.dirname(xdmf_path), f"forces_{ep}.csv")
            )
            if not os.path.exists(forces_file):
                raise ValueError(f"forces file {forces_file} does not exist.")
            forces_df = pd.read_csv(forces_file)
        else:
            forces_df = compute_forces_from_xdmf(
                case_xdmf_path=xdmf_path,
                mu=self.mu,
                feature_names=feature_names,
                start_step=self.start_step,
                object_containers=self.object_containers,
                verbose=False,
            )
        # save forces to file
        if save_data and not load_data:
            forces_file = os.path.abspath(
                os.path.join(os.path.dirname(xdmf_path), f"forces_{ep}.csv")
            )
            if not os.path.exists(forces_file):
                forces_df.to_csv(forces_file, index=False)

        # actual reward computation
        reward = 0
        for obj_id, obj_weight in enumerate(self.object_weights):
            reward += obj_weight * forces_df[forces_df["Object"] == obj_id]["Fy"].mean()
        return self.contrast_stretch(reward)


class MaximizeAvgDownforce(MinimizeAvgFx):
    def __init__(
        self,
        start_step: int = 300,
        bad_reward: float = -1e3,
        num_objects: int = 6,
        object_weights: List[float] = None,
        stretch_factor: float | None = None,
        stretch_type: str = "lin",
    ):
        """
        Initialize the MaximizeAvgDownforce reward.
        """
        super().__init__(
            start_step=start_step,
            bad_reward=bad_reward,
            num_objects=num_objects,
            object_weights=object_weights,
            stretch_factor=stretch_factor,
            stretch_type=stretch_type,
        )
        self.reward_type = "MaximizeAvgDownforce"

    def compute_reward(
        self,
        ep: int,
        xdmf_path: str,
        feature_names: dict = {},
        save_data: bool = True,
        load_data: bool = False,
    ) -> float:
        """
        Compute the reward for a given case based on the average downforce.

        Args:
            ep (int): Episode number.
            xdmf_path (str): Path to the XDMF file of the case.
            feature_names (dict): Dictionary mapping feature names used in the XDMF file.
            Needs to include keys for velocity, pressure, and nodetype.
            save_data (bool): Whether to save computed forces to a CSV file.
            load_data (bool): Whether to load forces from a CSV file instead of computing from XDMF.
        Returns:
            reward (float): Computed reward value.
        """
        if load_data:
            # load forces from csv file
            forces_file = os.path.abspath(
                os.path.join(os.path.dirname(xdmf_path), f"forces_{ep}.csv")
            )
            if not os.path.exists(forces_file):
                raise ValueError(f"forces file {forces_file} does not exist.")
            forces_df = pd.read_csv(forces_file)
        else:
            forces_df = compute_forces_from_xdmf(
                case_xdmf_path=xdmf_path,
                mu=self.mu,
                feature_names=feature_names,
                start_step=self.start_step,
                object_containers=self.object_containers,
                verbose=False,
            )
        # save forces to file
        if save_data and not load_data:
            forces_file = os.path.abspath(
                os.path.join(os.path.dirname(xdmf_path), f"forces_{ep}.csv")
            )
            if not os.path.exists(forces_file):
                forces_df.to_csv(forces_file, index=False)

        # actual reward computation
        reward = 0
        for obj_id, obj_weight in enumerate(self.object_weights):
            reward -= obj_weight * forces_df[forces_df["Object"] == obj_id]["Fy"].mean()
        return self.contrast_stretch(reward)


class MinimizeCombinedForces(MinimizeAvgFx):
    def __init__(
        self,
        start_step: int = 300,
        bad_reward: float = -1e3,
        num_objects: int = 6,
        object_weights: List[float] = None,
        drag_weight: float = 0.5,
        lift_weight: float = 0.5,
        stretch_factor: float | None = None,
        stretch_type: str = "lin",
    ):
        """
        Initialize the MinimizeCombinedForces reward.
        """
        super().__init__(
            start_step=start_step,
            bad_reward=bad_reward,
            num_objects=num_objects,
            object_weights=object_weights,
            stretch_factor=stretch_factor,
            stretch_type=stretch_type,
        )
        self.reward_type = "MinimizeCombinedForces"
        self.drag_weight = drag_weight
        self.lift_weight = lift_weight

    def compute_reward(
        self,
        ep: int,
        xdmf_path: str,
        feature_names: dict = {},
        save_data: bool = True,
        load_data: bool = False,
    ) -> float:
        """
        Compute the reward for a given case based on the combined average drag and lift forces.

        Args:
            ep (int): Episode number.
            xdmf_path (str): Path to the XDMF file of the case.
            feature_names (dict): Dictionary mapping feature names used in the XDMF file.
            Needs to include keys for velocity, pressure, and nodetype.
            save_data (bool): Whether to save computed forces to a CSV file.
            load_data (bool): Whether to load forces from a CSV file instead of computing from XDMF.
        Returns:
            reward (float): Computed reward value.
        """
        if load_data:
            # load forces from csv file
            forces_file = os.path.abspath(
                os.path.join(os.path.dirname(xdmf_path), f"forces_{ep}.csv")
            )
            if not os.path.exists(forces_file):
                raise ValueError(f"forces file {forces_file} does not exist.")
            forces_df = pd.read_csv(forces_file)
        else:
            forces_df = compute_forces_from_xdmf(
                case_xdmf_path=xdmf_path,
                mu=self.mu,
                feature_names=feature_names,
                start_step=self.start_step,
                object_containers=self.object_containers,
                verbose=False,
            )
        # save forces to file
        if save_data and not load_data:
            forces_file = os.path.abspath(
                os.path.join(os.path.dirname(xdmf_path), f"forces_{ep}.csv")
            )
            if not os.path.exists(forces_file):
                forces_df.to_csv(forces_file, index=False)

        # actual reward computation
        reward = 0
        for obj_id, obj_weight in enumerate(self.object_weights):
            drag = forces_df[forces_df["Object"] == obj_id]["Fx"].mean()
            lift = forces_df[forces_df["Object"] == obj_id]["Fy"].mean()
            reward -= obj_weight * (
                self.drag_weight * np.abs(drag) + self.lift_weight * np.abs(lift)
            )
        return self.contrast_stretch(reward)


class MinimizeSqrtAvgFx(MinimizeAvgFx):
    def __init__(
        self,
        start_step: int = 300,
        bad_reward: float = -1e3,
        num_objects: int = 6,
        object_weights: List[float] = None,
        stretch_factor: float | None = None,
        stretch_type: str = "lin",
    ):
        """
        Initialize the MinimizeSqrtAvgFx reward.
        """
        super().__init__(
            start_step=start_step,
            bad_reward=bad_reward,
            num_objects=num_objects,
            object_weights=object_weights,
            stretch_factor=stretch_factor,
            stretch_type=stretch_type,
        )
        self.reward_type = "MinimizeSqrtAvgFx"

    def compute_reward(
        self,
        ep: int,
        xdmf_path: str,
        feature_names: dict = {},
        save_data: bool = True,
        load_data: bool = False,
    ) -> float:
        """
        Compute the reward for a given case based on the average sqrt drag force.

        Args:
            ep (int): Episode number.
            xdmf_path (str): Path to the XDMF file of the case.
            feature_names (dict): Dictionary mapping feature names used in the XDMF file.
            Needs to include keys for velocity, pressure, and nodetype.
            save_data (bool): Whether to save computed forces to a CSV file.
            load_data (bool): Whether to load forces from a CSV file instead of computing from XDMF.
        Returns:
            reward (float): Computed reward value.
        """
        if load_data:
            # load forces from csv file
            forces_file = os.path.abspath(
                os.path.join(os.path.dirname(xdmf_path), f"forces_{ep}.csv")
            )
            if not os.path.exists(forces_file):
                raise ValueError(f"forces file {forces_file} does not exist.")
            forces_df = pd.read_csv(forces_file)
        else:
            forces_df = compute_forces_from_xdmf(
                case_xdmf_path=xdmf_path,
                mu=self.mu,
                feature_names=feature_names,
                start_step=self.start_step,
                object_containers=self.object_containers,
                verbose=False,
            )
        # save forces to file
        if save_data and not load_data:
            forces_file = os.path.abspath(
                os.path.join(os.path.dirname(xdmf_path), f"forces_{ep}.csv")
            )
            if not os.path.exists(forces_file):
                forces_df.to_csv(forces_file, index=False)

        # actual reward computation
        reward = 0
        for obj_id, obj_weight in enumerate(self.object_weights):
            fx_mean = forces_df[forces_df["Object"] == obj_id]["Fx"].mean()
            reward -= obj_weight * np.sqrt(np.abs(fx_mean))
        return self.contrast_stretch(reward)


class MinimizeSqrtAvgFy(MinimizeAvgFx):
    def __init__(
        self,
        start_step: int = 300,
        bad_reward: float = -1e3,
        num_objects: int = 6,
        object_weights: List[float] = None,
        stretch_factor: float | None = None,
        stretch_type: str = "lin",
    ):
        """
        Initialize the MinimizeSqrtAvgFy reward.
        """
        super().__init__(
            start_step=start_step,
            bad_reward=bad_reward,
            num_objects=num_objects,
            object_weights=object_weights,
            stretch_factor=stretch_factor,
            stretch_type=stretch_type,
        )
        self.reward_type = "MinimizeSqrtAvgFy"

    def compute_reward(
        self,
        ep: int,
        xdmf_path: str,
        feature_names: dict = {},
        save_data: bool = True,
        load_data: bool = False,
    ) -> float:
        """
        Compute the reward for a given case based on the average sqrt y-force.

        Args:
            ep (int): Episode number.
            xdmf_path (str): Path to the XDMF file of the case.
            feature_names (dict): Dictionary mapping feature names used in the XDMF file.
            Needs to include keys for velocity, pressure, and nodetype.
            save_data (bool): Whether to save computed forces to a CSV file.
            load_data (bool): Whether to load forces from a CSV file instead of computing from XDMF.
        Returns:
            reward (float): Computed reward value.
        """
        if load_data:
            # load forces from csv file
            forces_file = os.path.abspath(
                os.path.join(os.path.dirname(xdmf_path), f"forces_{ep}.csv")
            )
            if not os.path.exists(forces_file):
                raise ValueError(f"forces file {forces_file} does not exist.")
            forces_df = pd.read_csv(forces_file)
        else:
            forces_df = compute_forces_from_xdmf(
                case_xdmf_path=xdmf_path,
                mu=self.mu,
                feature_names=feature_names,
                start_step=self.start_step,
                object_containers=self.object_containers,
                verbose=False,
            )
        # save forces to file
        if save_data and not load_data:
            forces_file = os.path.abspath(
                os.path.join(os.path.dirname(xdmf_path), f"forces_{ep}.csv")
            )
            if not os.path.exists(forces_file):
                forces_df.to_csv(forces_file, index=False)

        # actual reward computation
        reward = 0
        for obj_id, obj_weight in enumerate(self.object_weights):
            fy_mean = forces_df[forces_df["Object"] == obj_id]["Fy"].mean()
            reward -= obj_weight * np.sqrt(np.abs(fy_mean))
        return self.contrast_stretch(reward)


class MinimizeSpecificFy(MinimizeAvgFx):
    def __init__(
        self,
        start_step: int = 300,
        bad_reward: float = -1e3,
        num_objects: int = 6,
        object_weights: List[float] = None,
        specific_lifts: List[float] | float = [0, 0, 0, 0, 0, 0],
        stretch_factor: float | None = None,
        stretch_type: str = "lin",
    ):
        """
        Initialize the MinimizeSpecificFy reward. Minimizes the abs difference between each object's
        average Fy and a target value. If only one specific_lift is provided, the reward becomes minimizing
        the abs difference between that value and the sum of Fy of all objects (no abs in Fy sum or object-wise).

        specific_lifts: List of target lift values for each object to minimize the difference from.
        """
        super().__init__(
            start_step=start_step,
            bad_reward=bad_reward,
            num_objects=num_objects,
            object_weights=object_weights,
            stretch_factor=stretch_factor,
            stretch_type=stretch_type,
        )
        self.reward_type = "MinimizeSpecificFy"
        if isinstance(specific_lifts, list) and len(specific_lifts) != num_objects:
            raise ValueError("Length of specific_lifts must match num_objects.")
        self.on_total_lift: bool = isinstance(specific_lifts, (int, float))
        self.specific_lifts = specific_lifts

    def compute_reward(
        self,
        ep: int,
        xdmf_path: str,
        feature_names: dict = {},
        save_data: bool = True,
        load_data: bool = False,
    ) -> float:
        """
        Compute the reward for a given case based on the distance from specific target Fy values.

        Args:
            ep (int): Episode number.
            xdmf_path (str): Path to the XDMF file of the case.
            feature_names (dict): Dictionary mapping feature names used in the XDMF file.
            Needs to include keys for velocity, pressure, and nodetype.
            save_data (bool): Whether to save computed forces to a CSV file.
            load_data (bool): Whether to load forces from a CSV file instead of computing from XDMF.
        Returns:
            reward (float): Computed reward value.
        """
        if load_data:
            # load forces from csv file
            forces_file = os.path.abspath(
                os.path.join(os.path.dirname(xdmf_path), f"forces_{ep}.csv")
            )
            if not os.path.exists(forces_file):
                raise ValueError(f"forces file {forces_file} does not exist.")
            forces_df = pd.read_csv(forces_file)
        else:
            forces_df = compute_forces_from_xdmf(
                case_xdmf_path=xdmf_path,
                mu=self.mu,
                feature_names=feature_names,
                start_step=self.start_step,
                object_containers=self.object_containers,
                verbose=False,
            )
        # save forces to file
        if save_data and not load_data:
            forces_file = os.path.abspath(
                os.path.join(os.path.dirname(xdmf_path), f"forces_{ep}.csv")
            )
            if not os.path.exists(forces_file):
                forces_df.to_csv(forces_file, index=False)

        # actual reward computation
        reward = 0
        if self.on_total_lift:
            for obj_id, obj_weight in enumerate(self.object_weights):
                fy_mean = forces_df[forces_df["Object"] == obj_id]["Fy"].mean()
                reward += obj_weight * fy_mean
            reward = -np.abs(reward - self.specific_lifts)
        else:
            for obj_id, obj_weight in enumerate(self.object_weights):
                fy_mean = forces_df[forces_df["Object"] == obj_id]["Fy"].mean()
                reward -= obj_weight * np.abs(fy_mean - self.specific_lifts[obj_id])
        return self.contrast_stretch(reward)


def parser():
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Reward computation utilities",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "reward_type",
        type=str,
        help=(
            "Type of reward to compute. Choices include: VelocityFluctuation, MinimizeAvgFx, "
            "MaximizeAvgFx, MinimizeAvgFy, MaximizeAvgLift, MaximizeAvgDownforce, MinimizeL2AvgFx, "
            "MinimizeL2AvgFy, MinimizeCombinedForces, MinimizeSqrtAvgFx, MinimizeSqrtAvgFy, "
            "MinimizeSpecificFy"
        ),
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        type=str,
        required=True,
        help="Path of the case XDMF files to compute reward on (one or more can be passed).",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Whether to search for case files recursively in directories.",
    )
    parser.add_argument(
        "-n",
        "--num_objects",
        type=int,
        default=1,
        help="Number of objects to consider for reward computation.",
    )
    parser.add_argument(
        "--start_step",
        type=int,
        default=300,
        help="Timestep index to start the reward computation from.",
    )
    parser.add_argument(
        "-k",
        "--stretch_factor",
        type=float,
        default=None,
        help="Factor for contrast stretching of the reward (k>0).",
    )
    parser.add_argument(
        "--stretch_type",
        type=str,
        default="lin",
        help="Type of reward stretching to apply (e.g., 'lin', 'tanh', 'pow', 'log').",
    )
    parser.add_argument(
        "--save_data",
        action="store_true",
        help="Whether to save computed data from reward to a CSV file.",
    )
    parser.add_argument(
        "--load_data",
        action="store_true",
        help="Whether to load forces from a CSV file instead of computing from XDMF.",
    )
    parser.add_argument(
        "--feature_names",
        nargs="*",
        default=None,
        help=(
            "Feature names mapping for velocity, pressure, levelset, and nodetype in the XDMF file. "
            "Format: key=value key2=value2 ... OR a single JSON string "
            '\'{"velocity":"Vitesse","pressure":"Pression","levelset":"LevelSetObject","nodetype":"NodeType"}\''
        ),
    )
    parser.add_argument(
        "-o",
        "--output_name",
        type=str,
        default=None,
        help="Name of the output file for the reward computation results (csv).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print detailed information during processing.",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Path to save reward plot (if applicable).",
    )
    args = parser.parse_args()

    # convert --feature_names from list of "key=value" or a single JSON string to a dict
    if args.feature_names:
        # allow passing a single JSON string
        if len(args.feature_names) == 1 and args.feature_names[0].strip().startswith(
            "{"
        ):
            try:
                parsed = json.loads(args.feature_names[0])
            except Exception as e:
                raise argparse.ArgumentTypeError(
                    f"Invalid JSON for --feature_names: {e}"
                )
            if not isinstance(parsed, dict):
                raise argparse.ArgumentTypeError(
                    "--feature_names JSON must decode to a dict"
                )
            args.feature_names = parsed
        else:
            parsed = {}
            for item in args.feature_names:
                if "=" not in item:
                    raise argparse.ArgumentTypeError(
                        f"Invalid feature name '{item}'. Expected format key=value"
                    )
                k, v = item.split("=", 1)
                parsed[k] = v
            args.feature_names = parsed
    else:
        args.feature_names = {}

    return args


def main():
    import sys
    import re
    from typing import List

    def gather_cases_list(args) -> List[str]:
        """Gather xdmf files (recursively from dir if specified)."""
        if args.recursive:
            all_cases = []
            for case_path in args.cases:
                gathered = gather_cases_for_reward_recursive(
                    case_path, get_data_files_only=args.load_data
                )
                all_cases.extend(gathered)
            print(f"Found {len(all_cases)} cases recursively for reward computation.")
            return all_cases
        return args.cases

    def init_reward_obj(args):
        """Initialize the reward object based on the reward type and arguments."""
        # Special handling for VelocityFluctuation (load reward_boxes from user input or env var)
        if args.reward_type == "VelocityFluctuation":
            reward_boxes = None
            if sys.stdin.isatty():
                resp = input(
                    "Path to reward_boxes JSON (leave empty to use defaults): "
                ).strip()
                if resp:
                    if os.path.exists(resp):
                        reward_boxes = resp
                    else:
                        print(f"Warning: '{resp}' not found. Using defaults.")
            else:
                resp = os.environ.get("REWARD_BOXES", "").strip()
                if resp and os.path.exists(resp):
                    reward_boxes = resp
            return VelocityFluctuationReward(
                start_step=args.start_step,
                reward_boxes=reward_boxes,
                stretch_factor=args.stretch_factor,
                stretch_type=args.stretch_type,
            )
        if args.reward_type == "MinimizeSpecificFy":
            specific_lifts = None
            if sys.stdin.isatty():
                resp = input(
                    "Enter specific_lifts as comma-separated values (leave empty for all zeros): "
                ).strip()
                if resp:
                    try:
                        specific_lifts = [float(val) for val in resp.split(",")]
                    except ValueError:
                        print(
                            "Invalid input for specific_lifts. Using zeros by default."
                        )
            else:
                resp = os.environ.get("SPECIFIC_LIFTS", "").strip()
                if resp:
                    try:
                        specific_lifts = [float(val) for val in resp.split(",")]
                    except ValueError:
                        print("Invalid SPECIFIC_LIFTS env var. Using zeros by default.")
            if len(specific_lifts) == 1 and not (args.num_objects == 1):
                specific_lifts = float(specific_lifts[0])
            return MinimizeSpecificFy(
                start_step=args.start_step,
                num_objects=args.num_objects,
                specific_lifts=(
                    specific_lifts
                    if specific_lifts is not None
                    else [0] * args.num_objects
                ),
                stretch_factor=args.stretch_factor,
                stretch_type=args.stretch_type,
            )
        # other rewards
        try:
            reward_class = globals()[args.reward_type]
        except KeyError:
            raise ValueError(f"Unknown reward type: {args.reward_type}")
        return reward_class(
            start_step=args.start_step,
            num_objects=args.num_objects,
            stretch_factor=args.stretch_factor,
            stretch_type=args.stretch_type,
        )

    def compute_rewards(reward_obj, cases: List[str], args):
        rows = []
        # iterate over episodes / cases
        for ep_idx, case_xdmf in tqdm(
            enumerate(cases), desc="Computing rewards", disable=args.verbose
        ):
            # try to extract episode number from filename, fallback to index
            case_name = os.path.basename(case_xdmf).split(".")[0].split("_")[-1]
            m = re.search(r"(\d+)", case_name)
            ep_val = int(m.group(1)) if m else ep_idx

            # compute reward
            if args.reward_type == "VelocityFluctuation":
                velocity_field = (
                    args.feature_names.get("velocity", "Vitesse")
                    if isinstance(args.feature_names, dict)
                    else "Vitesse"
                )
                reward = reward_obj.compute_reward(
                    ep=ep_val, xdmf_path=case_xdmf, velocity_field_keys=velocity_field
                )
            else:
                reward = reward_obj.compute_reward(
                    ep=ep_val,
                    xdmf_path=case_xdmf,
                    feature_names=args.feature_names,
                    save_data=args.save_data,
                    load_data=args.load_data,
                )

            rows.append({"ep": ep_val, "case": case_xdmf, "reward": reward})
            if args.verbose:
                print(f"Reward for ep #{ep_val} (case {case_xdmf}): {reward:.6f}")
        return pd.DataFrame(rows)

    def save_plot(df: pd.DataFrame, plot_path: str, reward_type: str):
        out_path = plot_path if plot_path.endswith(".png") else plot_path + ".png"
        plt.figure(figsize=(12, 8))
        plt.scatter(
            df["ep"],
            df["reward"],
            marker="o",
            facecolors="none",
            edgecolors="darkblue",
        )
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        if not df["reward"].empty:
            ymin = df["reward"].min()
            ymax = df["reward"].max()
            plt.ylim(
                ymin - 0.1 * abs(ymin),
                max(ymax + 0.1 * abs(ymax), 0),
            )
        plt.title(f"Rewards over Episodes ({reward_type})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"Saved reward plot to {out_path}")

    # MAIN EXECUTION
    args = parser()

    cases = gather_cases_list(args)
    if not cases:
        print("No cases to process. Exiting.")
        return

    reward_obj = init_reward_obj(args)
    rewards_df = compute_rewards(reward_obj, cases, args)

    if args.plot:
        save_plot(rewards_df, args.plot, args.reward_type)

    if args.save_data:
        df = rewards_df.sort_values(by="ep")
        if not args.output_name:
            output_csv = f"rewards_{args.reward_type}.csv"
        else:
            output_csv = (
                args.output_name
                if args.output_name.endswith(".csv")
                else args.output_name + ".csv"
            )
        df.to_csv(output_csv, index=False)
        print(f"Saved rewards summary to {output_csv}")


if __name__ == "__main__":
    main()
