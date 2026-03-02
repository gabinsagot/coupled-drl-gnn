import json
import pandas as pd
import numpy as np
import os
from typing import Any, Dict
import argparse
from abc import ABC, abstractmethod


class Configs(ABC):
    """
    Class to handle the configurations of the dataset.
    """

    def __init__(self, path: str, num_configs: int | None = None, create: bool = False):
        """
        :param path: path to the configurations pool file (no extension,
        here should be at least a pkl file and maybe a txt file for visualizing content)
        :param num_configs: number of configurations to generate (depending on if create passed)
        :param create: if True, generates a new configurations pool file with num_configs configurations,
        if False, loads the existing configurations pool file
        """
        self.pool_path: str = os.path.splitext(path)[0]
        if create:
            if num_configs is None:
                raise ValueError("num_configs must be passed if create is True")
            if os.path.exists(self.pool_path + ".pkl") or os.path.exists(
                self.pool_path + ".txt"
            ):
                raise FileExistsError(
                    f"File {self.pool_path}.pkl already exists, delete/rename it first if you want to create a new one"
                )
            else:
                print(f"Creating configurations pool file at {self.pool_path}")
                dir_name = os.path.dirname(self.pool_path)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)
        self.configs: pd.DataFrame = (
            self.create_configs_pool(n_configs=num_configs)
            if create
            else self.load_configs(n_configs=num_configs)
        )

    def get_size(self) -> int:
        """Returns the number of configurations in the pool"""
        return len(self.configs.index)

    def load_configs(self, n_configs: int | None = None) -> pd.DataFrame:
        """Loads the configurations of the dataset,
        Must be a pickle file (.pkl), .txt files are only for visuals"""
        # configs = pd.read_csv(self.pool_path, sep='\t')
        try:
            pool_path_pkl = self.pool_path + ".pkl"
            configs = pd.read_pickle(pool_path_pkl)
            if n_configs is not None:
                if n_configs > len(configs.index):
                    raise ValueError(
                        f"Cannot load {n_configs} configurations, only {len(configs.index)} available"
                    )
                configs = configs.sample(n=n_configs).reset_index(drop=True)
            configs.to_csv(self.pool_path + "_selected.txt", sep="\t", index=False)
            configs.to_pickle(self.pool_path + "_selected.pkl")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Cannot load configurations pool file {self.pool_path}, maybe generate one first?"
            )
        return configs

    def get_config(self, config_name: str) -> pd.DataFrame:
        """Returns a single line dataframe of the configuration parameters,
        for the configuration named config_name"""
        if config_name not in self.configs["Config"].values:
            raise ValueError(f"Configuration {config_name} not found in the pool")
        config = self.configs[self.configs["Config"] == config_name]
        return config

    def select_random_configs(
        self,
        n_configs: int,
        save_selection: bool = True,
    ) -> pd.DataFrame:
        """Selects n_configs random configurations from configs,
        returns a DataFrame with the selected configurations.

        :param n_configs: number of configurations to select
        :param save_selection: if True, saves the selected configurations as a configurations pool file
        """
        selected_configs = self.configs.sample(n=n_configs)
        if save_selection:
            selected_configs.to_pickle("selected_configs.pkl")
            selected_configs.to_csv("selected_configs.txt", sep="\t", index=False)
        return selected_configs

    def display(self):
        print(self.configs)

    def shift_x_coords(
        self, config_x_values: np.ndarray, x_origin: float
    ) -> np.ndarray:
        """Shifts x coordinates of a configuration to put first object at x_min (from json)"""
        x_shift = x_origin - min(config_x_values)
        config_x_values_ = np.array([x_i + x_shift for x_i in config_x_values])
        return config_x_values_

    def similarity_check(
        self,
        config1: pd.DataFrame,
        config2: pd.DataFrame,
        similarity_threshold_dict: dict[str, float] = None,
        which_trigger: str = "one",
    ) -> bool:
        """Checks if two configurations are too similar based on a similarity threshold (L2 norm). Similarity threshold
        corresponds to distance between two configs based on min of L2 norms for each parameter passed in list.

        Args:
            config1 (pd.DataFrame): first configuration
            config2 (pd.DataFrame): second configuration, can be multiple rows in which case will check for each row.
            similarity_threshold_dict (dict[str, float], optional): dict of minimum L2 distance per parameter,\
                between a pair of 2 configurations.
            which_trigger (str, optional): How to consider similarity on parameters - "all" means all parameters
            must be similar to trigger similarity check, "one" means only one similar parameter will trigger
            similarity check. Defaults to "one".
        Returns:
            bool: True if the configurations are too similar (below similarity threshold), False otherwise
        """
        # keep only non-zero thresholds (ie parameters to check)
        filtered_threshs_dict = {
            k: v for k, v in similarity_threshold_dict.items() if v != 0
        }
        # sanity checks
        if not filtered_threshs_dict:
            raise ValueError(
                "For config similarity check, you need to define at least one non-zero similarity threshold"
            )
        if not isinstance(config1, pd.DataFrame) or not isinstance(
            config2, pd.DataFrame
        ):
            raise TypeError("config1 and config2 must be pandas DataFrames")
        if config1.shape[0] != 1:
            raise ValueError("config1 must be a single row DataFrame")
        if config2.shape[0] < 1:
            raise ValueError("config2 must have at least one row")
        if not all(
            ((param in config1.columns) and (param in config2.columns))
            for param in filtered_threshs_dict.keys()
        ):
            raise ValueError(
                f"Some parameters {list(filtered_threshs_dict.keys())} are not in the configuration DataFrame columns"
            )
        for param in filtered_threshs_dict.keys():
            if not isinstance(config1[param].iloc[0], np.ndarray) or not isinstance(
                config2[param].iloc[0], np.ndarray
            ):
                raise TypeError(
                    f"Parameter {param} must be of type np.ndarray in both configurations but is of type "
                    f"{type(config1[param].iloc[0])} and {type(config2[param].iloc[0])}"
                )
        if not isinstance(similarity_threshold_dict, dict):
            raise TypeError(
                "similarity_threshold_dict must be a dict with parameter names as keys and float thresholds as values"
            )
        if which_trigger not in ["one", "all"]:
            raise ValueError('which_trigger must be either "one" or "all"')

        # L2 norm by parameter
        is_similar = True if which_trigger == "all" else False
        for param, thresh in filtered_threshs_dict.items():
            for idx in range(config2.shape[0]):
                # catch case where two configs have different number of objects (de facto disimilar)
                if len(config2[param].iloc[idx]) != len(config1[param].iloc[0]):
                    continue
                else:
                    l2_norm = np.linalg.norm(
                        config1[param].iloc[0] - config2[param].iloc[idx]
                    )
                    if which_trigger == "one":
                        is_similar = (l2_norm < thresh) or is_similar
                    elif which_trigger == "all":
                        is_similar = (l2_norm < thresh) and is_similar
        return is_similar

    @abstractmethod
    def create_configs_pool(self, n_configs: int) -> pd.DataFrame:
        """Creates a configurations pool file with n_configs configurations,
        respecting the similarity threshold, saves at path location.
        Specific configs classes will overide this method to create theirs.

        :param n_configs: number of configurations to generate
        :param similarity_threshold: minimum distance between configurations
        """
        raise NotImplementedError("This method must be over-written")


class ConfigsAirfoil(Configs):
    def __init__(
        self,
        meta_dict: Dict[str, Any],
        path: str,
        num_configs: int,
        create: bool = False,
    ):
        self.geo_params: Dict[str, Any] = meta_dict["geometry_parameters"]
        self.shift_x_objects: bool = meta_dict["shift_x_objects"]
        self.cfd_params: Dict[str, Any] = meta_dict["cfd_parameters"]
        self.variable_inlet: bool = self.cfd_params["variable_inlet"]
        # sanity check
        if self.variable_inlet and (
            not isinstance(self.cfd_params["VxIn"], (list, tuple))
            or len(self.cfd_params["VxIn"]) != 2
        ):
            raise ValueError(
                "When passing 'variable_inlet:true', VxIn must be a list or tuple of length 2 (min/max values)"
            )
        super().__init__(path=path, num_configs=num_configs, create=create)

    def check_no_overlap(
        self,
        x_list: list[float],
        y_list: list[float],
        chords: list[float],
        margin=1,
    ) -> bool:
        """Checks if the airfoils defined by the x, y, chords list overlap.
        Returns True if they don't overlap, False otherwise.
        This is a simple distance check based on chord lengths, not exact geometry.

        :param x_list: list of x origins of airfoils
        :param y_list: list of y origins of airfoils
        :param chords: list of chord lengths of airfoils
        :param margin: minimum distance between airfoils
        """
        # compute max distances based on chord lengths
        max_distances = [(2 * chord**2) ** 0.5 for chord in chords]
        for i in range(len(x_list)):
            for j in range(i + 1, len(x_list)):
                distance = (
                    (x_list[i] - x_list[j]) ** 2 + (y_list[i] - y_list[j]) ** 2
                ) ** 0.5
                if distance < (max_distances[i] + max_distances[j] + margin):
                    return False
        return True

    def create_configs_pool(self, n_configs: int) -> pd.DataFrame:
        """Creates a configurations pool file with n_configs airfoil configurations,
        respecting a certain dissimilarity threshold, saves at path location.
        Distinguishes between classic and morphing airfoil generation methods, given that
        they require different configuration generation approaches.

        :param n_configs: number of configurations to generate
        """
        # check if morphing
        if "morph" in self.geo_params.get("airfoil_type", "NACA").lower():
            morphing = True
        else:
            morphing = False

        if self.variable_inlet:
            configs_pool = pd.DataFrame(
                columns=[
                    "Config",
                    "cambers" if morphing else "naca_code",
                    "thicknesses",
                    "angles",
                    "chords",
                    "x_objects",
                    "y_objects",
                    "v_inlet",
                    "number_airfoils",
                ]
            )
            v_in_min = self.cfd_params["VxIn"][0]
            v_in_max = self.cfd_params["VxIn"][1]
        else:
            configs_pool = pd.DataFrame(
                columns=[
                    "Config",
                    "cambers" if morphing else "naca_code",
                    "thicknesses",
                    "angles",
                    "chords",
                    "x_objects",
                    "y_objects",
                    "number_airfoils",
                ]
            )
        n_airfoils_min = self.geo_params["number_airfoils"][0]
        n_airfoils_max = self.geo_params["number_airfoils"][1]
        if morphing:
            cambers_min = np.array(
                [cambers_[0] for cambers_ in self.geo_params["camber"]]
            )
            cambers_max = np.array(
                [cambers_[1] for cambers_ in self.geo_params["camber"]]
            )
        else:
            naca_code = self.geo_params["airfoil_type"]
        if morphing:
            thickness_min = np.array(
                [thickness_[0] for thickness_ in self.geo_params["thickness"]]
            )
            thickness_max = np.array(
                [thickness_[1] for thickness_ in self.geo_params["thickness"]]
            )
        else:
            thickness_min = self.geo_params["thickness"][0]
            thickness_max = self.geo_params["thickness"][1]
        chord_min = self.geo_params["chord"][0]
        chord_max = self.geo_params["chord"][1]
        angle_min = self.geo_params["angle_of_attack"][0]
        angle_max = self.geo_params["angle_of_attack"][1]
        angle_step = self.geo_params["angle_step"]
        # sanity check
        if angle_min + angle_step > angle_max:
            raise ValueError(
                f"angle_min + angle_step ({angle_min} + {angle_step}) must be less than or equal to angle_max "
                f"({angle_max})... Please check your geometry_parameters in the config file."
            )
        possible_angles = np.arange(
            start=angle_min, stop=angle_max + angle_step, step=angle_step
        )
        x_min = self.geo_params["x_object"][0]
        x_max = self.geo_params["x_object"][1]
        y_min = self.geo_params["y_object"][0]
        y_max = self.geo_params["y_object"][1]
        similarity_thresholds = {  # per parameter
            "chords": (chord_max - chord_min) / 10 if chord_min < chord_max else 0,
            "thicknesses": (
                (thickness_max - thickness_min) / 10
                if not morphing and thickness_min < thickness_max
                else (
                    (np.min(thickness_max - thickness_min)) / 10
                    if morphing and np.any(thickness_min < thickness_max)
                    else 0
                )
            ),
            "cambers": (
                (np.min(cambers_max - cambers_min)) / 10
                if morphing and np.any(cambers_min < cambers_max)
                else 0
            ),
            "v_inlet": (v_in_max - v_in_min) / 10 if self.variable_inlet else 0,
            "angles": angle_step if angle_min < angle_max else 0,
        }
        total_tries = 0
        for i in range(n_configs):
            total_tries_config = 0
            while True:
                # num airfoils
                n_airfoils = (
                    n_airfoils_min
                    if n_airfoils_min == n_airfoils_max
                    else np.random.randint(n_airfoils_min, n_airfoils_max + 1)
                )
                # chords
                chords = np.round(
                    np.random.uniform(chord_min, chord_max, n_airfoils), 2
                )
                # cambers
                if morphing:
                    cambers = np.round(
                        np.random.uniform(
                            cambers_min, cambers_max, (n_airfoils, len(cambers_min))
                        ),
                        3,
                    )
                # thicknesses
                if morphing:
                    thicknesses = np.round(
                        np.random.uniform(
                            thickness_min,
                            thickness_max,
                            (n_airfoils, len(thickness_min)),
                        ),
                        3,
                    )
                else:
                    thicknesses = np.round(
                        np.random.uniform(thickness_min, thickness_max, n_airfoils), 2
                    )
                # aoa
                angles = np.random.choice(
                    possible_angles, size=n_airfoils, replace=True
                )
                # x y origins
                x_objects = np.round(np.random.uniform(x_min, x_max, n_airfoils), 2)
                if self.shift_x_objects:
                    x_objects = self.shift_x_coords(
                        config_x_values=x_objects,
                        x_origin=min(self.geo_params["x_object"]),
                    )
                y_objects = np.round(np.random.uniform(y_min, y_max, n_airfoils), 2)
                # config name
                config_name = f"{n_airfoils}{i:04d}"
                # inlet
                if self.variable_inlet:
                    v_inlet = np.round(np.random.uniform(v_in_min, v_in_max, 1), 1)
                    new_config = pd.DataFrame(
                        {
                            "Config": config_name,
                            "cambers" if morphing else "naca_code": (
                                [cambers] if morphing else [naca_code]
                            ),
                            "thicknesses": [thicknesses],
                            "angles": [angles],
                            "chords": [chords],
                            "x_objects": [x_objects],
                            "y_objects": [y_objects],
                            "v_inlet": [v_inlet],
                            "number_airfoils": n_airfoils,
                        }
                    )
                else:
                    new_config = pd.DataFrame(
                        {
                            "Config": config_name,
                            "cambers" if morphing else "naca_code": (
                                [cambers] if morphing else [naca_code]
                            ),
                            "thicknesses": [thicknesses],
                            "angles": [angles],
                            "chords": [chords],
                            "x_objects": [x_objects],
                            "y_objects": [y_objects],
                            "number_airfoils": n_airfoils,
                        }
                    )

                # checks
                is_similar = False
                if not configs_pool.empty:
                    is_similar = self.similarity_check(
                        config1=new_config,
                        config2=configs_pool,
                        similarity_threshold_dict=similarity_thresholds,
                        which_trigger="all",
                    )
                no_overlap = self.check_no_overlap(
                    x_list=x_objects,
                    y_list=y_objects,
                    chords=chords,
                    margin=0.5,
                )
                # TODO: add check for surface area
                if not is_similar and no_overlap:
                    configs_pool = (
                        new_config
                        if configs_pool.empty
                        else pd.concat([configs_pool, new_config], ignore_index=True)
                    )
                    break
                total_tries += 1
                total_tries_config += 1
                if total_tries > 5000 or total_tries_config > 1000:
                    raise ValueError(
                        "Cannot generate configurations respecting "
                        "the various constraints (5000+ tries were made)"
                    )
        configs_pool.to_csv(self.pool_path + ".txt", sep="\t", index=False)
        configs_pool.to_pickle(self.pool_path + ".pkl")
        return configs_pool


# FOR TESTING


def _parser_configs(
    description: str = "Generate configuration pools",
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, help="path to the configuration file")
    parser.add_argument(
        "--n_configs", type=int, help="number of configurations to generate"
    )
    parser.add_argument(
        "--path_to_pool", type=str, help="path of the output configurations pool file"
    )
    return parser.parse_args()


def main_airfoil():
    args = _parser_configs(
        description="Generate configurations pool for airfoil problem"
    )
    meta_dict = json.load(open(args.config))
    configs = ConfigsAirfoil(
        meta_dict=meta_dict,
        path=args.path_to_pool,
        num_configs=args.n_configs,
        create=True,
    )
    configs.display()
    print("\nconfig pool size: \n", configs.get_size())
    some_config_name = configs.configs.iloc[0]["Config"]
    print(
        f"\nconfig {some_config_name}: \n",
        configs.get_config(config_name=some_config_name),
    )
    print(
        "\nselected_configs: \n",
        configs.select_random_configs(
            n_configs=3, similarity_threshold=0.1, save_selection=True
        ),
    )


if __name__ == "__main__":
    main_airfoil()
