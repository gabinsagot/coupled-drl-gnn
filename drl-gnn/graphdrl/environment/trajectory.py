import os
from typing import List

import numpy as np
import meshio

from graphdrl.environment.geometry import Geometry
from graphdrl.environment.cimlib import CimlibEnv
from graphdrl.utils.file_handling import move_meshes
from graphdrl.utils.meshio_mesh import (
    meshes_to_xdmf,
    convert_gmsh_to_mtc,
    msh_to_meshes,
    vtu_to_meshes,
)
from graphdrl.utils.nodetype import NodeType


BASE_FEATURES = ["NodeType", "LevelSetObject", "Vitesse", "Pression"]


class Trajectory:
    """
    Class to handle a trajectory of meshes in time, and their associated data.
    Applies only to fixed mesh trajectories (mesh stays the same over time).
    """

    def __init__(
        self, meshes: List[meshio.Mesh], times: List[float], timestep: float = None
    ):
        """
        Initialize the Trajectory with a list of meshes and corresponding times.

        Args:
            meshes (List[meshio.Mesh]): List of meshio Mesh objects representing the trajectory.
            times (List[float]): List of time values corresponding to each mesh.
            timestep (float): Time step between each mesh.
        """
        if len(meshes) != len(times):
            raise ValueError("Length of meshes and times must be the same.")
        if timestep is None:
            if len(times) == 1:
                raise ValueError(
                    "Timestep must be provided if there is only one frame."
                )
            else:
                timestep = times[1] - times[0]
        if timestep is not None:
            if len(times) > 1:
                inferred_timestep = times[1] - times[0]
                if not np.isclose(timestep, inferred_timestep):
                    raise ValueError(
                        "Provided timestep does not match the difference between time values."
                    )
        self.meshes = meshes
        self.times = times
        self.timestep = timestep
        self.length = len(self.meshes)
        self.points = self.get_points()
        self.cells = self.get_cells()
        self.num_points = self.points.shape[0]

    def __len__(self):
        return len(self.meshes)

    def get_fields(self):
        """
        Returns list of names of the P1 (point-defined) fields of the trajectory.
        """
        return list(self.meshes[0].point_data.keys())

    def get_mesh_at_time(self, time: float) -> meshio.Mesh:
        """
        Get the mesh at a specific time.
        """
        if time in self.times:
            index = self.times.index(time)
            return self.meshes[index]
        else:
            raise ValueError(f"Time {time} not found in trajectory times.")

    def get_mesh_at_step(self, step: int) -> meshio.Mesh:
        """
        Get the mesh at a specific step index.
        """
        if 0 <= step < self.length:
            return self.meshes[step]
        else:
            raise IndexError(
                f"Step index {step} out of range (0 to {self.length - 1})."
            )

    def save(self, filename: str = "./traj.xdmf") -> None:
        """
        Saves trajectory to xdmf to given path with given name.
        """
        filename = filename if filename.endswith(".xdmf") else f"{filename}.xdmf"
        meshes_to_xdmf(
            filename=filename,
            meshes=self.meshes,
            timestep=self.times,
            verbose=False,
            drop_firststep=False,
        )

    def get_points(self):
        """
        Get mesh nodes.
        """
        return self.meshes[0].points

    def get_cells(self):
        """
        Get mesh cells.
        """
        return self.meshes[0].cells

    def slice(self, start: int = None, end: int = None) -> "Trajectory":
        """
        Slice the trajectory to a specific step range, between start and end.
        If start or end is None, it defaults to the beginning or end of the trajectory respectively.
        """
        # simple slicing with support for None and negative indices
        n = self.length
        if start is None and end is None:
            return self

        try:
            s = 0 if start is None else int(start)
            e = n if end is None else int(end)
        except (TypeError, ValueError):
            raise ValueError("start and end must be integers or None")

        if s < 0:
            s += n
        if e < 0:
            e += n

        s = max(0, min(s, n))
        e = max(0, min(e, n))

        if s > e:
            raise ValueError(f"start ({s}) must be <= end ({e}) after normalization")

        if s == 0 and e == n:
            return self

        return Trajectory(
            meshes=self.meshes[s:e], times=self.times[s:e], timestep=self.timestep
        )

    def extend(
        self, total_length: int = None, total_time: float = None, which: str = "first"
    ) -> None:
        """
        Extend a trajectory from its current length to either total_length or total_time,
        depending on which is provided. The first initial mesh and its data will be duplicated.

        Args:
            total_length (int): total length of the new trajectory (in number of steps)
            total_time (float): total time of new trajectory (last timestamp)
            which (int): which frame to duplicate in extension (either 'first' or 'last')
        """
        if total_length is None and total_time is None:
            raise ValueError("Either total_length or total_time must be provided.")
        if total_length is not None and total_time is not None:
            if not np.isclose(total_length, int(total_time / self.timestep)):
                raise ValueError(
                    "Both total_length and total_time provided but don't match."
                )
        if total_length is not None:
            if total_length <= self.length:
                raise ValueError("total_length must be greater than current length.")
            num_additional_steps = total_length - self.length
            additional_times = [
                self.times[-1] + (i + 1) * self.timestep
                for i in range(num_additional_steps)
            ]
        else:  # total_time is not None
            if total_time <= self.times[-1]:
                raise ValueError("total_time must be greater than current last time.")
            num_additional_steps = int((total_time - self.times[-1]) / self.timestep)
            additional_times = [
                self.times[-1] + (i + 1) * self.timestep
                for i in range(num_additional_steps)
            ]
        if not (which in ["first", "last"]):
            raise ValueError(
                "The traj frame to duplicate must be chosen from either 'first' or 'last'."
            )
        else:
            which = 0 if which == "first" else -1

        additional_meshes = [
            self.meshes[which].copy() for _ in range(num_additional_steps)
        ]
        self.meshes.extend(additional_meshes)
        self.times.extend(additional_times)
        self.length = len(self.meshes)

    def build_wall_mask(self) -> np.ndarray:
        """
        Build a boolean mask for wall nodes based on the 'NodeType' field.
        Wall nodes here are defined as either NodeType.WALL_BOUNDARY or NodeType.OBJECT
        Returns:
            np.ndarray: Boolean array where True indicates a wall node.

        """
        if "NodeType" not in self.get_fields():
            raise ValueError("The mesh does not contain 'NodeType' field.")
        node_types = self.meshes[0].point_data["NodeType"]
        wall_mask = (node_types == NodeType.OBSTACLE) | (
            node_types == NodeType.WALL_BOUNDARY
        )
        return wall_mask

    def init_boundary_conditions(
        self,
        inlet_profile: str = "uniform",
        u_inf: float = 1.0,
        wall_bc: bool = False,
    ):
        """
        Initialize boundary conditions for the mesh.
        Sets the inlet velocity profile everywhere.
        Only works for empty (only one initial frame) trajectories.
        You can choose to also upload boundary conditions from a vtu file by using inlet_profile="from_vtu".

        Args:
            inlet_profile: str, type of inlet velocity profile ("uniform" or "abl", or "from_vtu").
            u_inf: float, free stream velocity for the inlet profile.
            wall_bc: bool, whether to apply a wall bc to the velocity field (on wall/obstacle nodes).
        """
        # sanity checks
        if self.length > 1:
            raise ValueError(
                "This trajectory is not empty (contains more than just an initial frame)."
            )
        # define bc
        if inlet_profile == "uniform":

            def inlet_velocity_map(y):
                return u_inf  # uniform inlet velocity everywhere

        elif inlet_profile == "abl":

            def inlet_velocity_map(y):
                A = 0.18  # rugosity constant
                y0c = 0.05  # roughness length
                return u_inf * A * np.log(1 + (y / y0c))

        elif inlet_profile == "from_vtu":
            if "Vitesse" not in self.get_fields():
                raise ValueError(
                    "The provided vtu does not contain velocity data under 'Vitesse'."
                )
            if "Pression" not in self.get_fields():
                # default to 0 pressure everywhere
                pressure = np.zeros((self.num_points, 1), dtype=float)
                self.meshes[0].point_data["Pression"] = pressure
            if wall_bc:
                wall_mask = self.build_wall_mask()
                velocity = self.meshes[0].point_data["Vitesse"]
                velocity[wall_mask, :] = 0.0  # set velocity to 0 at wall nodes
                self.meshes[0].point_data["Vitesse"] = velocity
            return

        else:
            raise ValueError(f"Unknown inlet profile: {inlet_profile}")

        # Initialize features
        velocity = np.zeros((self.num_points, 3))
        pressure = np.zeros((self.num_points, 1), dtype=float)
        for i, point in enumerate(self.points):
            _, y = point[0], point[1]
            velocity[i, 0] = inlet_velocity_map(y)  # set Vx according to profile

        if wall_bc:
            wall_mask = self.build_wall_mask()
            velocity[wall_mask, :] = 0.0  # set velocity to 0 at wall nodes

        self.meshes[0].point_data["Vitesse"] = velocity
        self.meshes[0].point_data["Pression"] = pressure
        return

    def apply_wall_bc(self, start_step: int = 1, end_step: int = None) -> None:
        """
        Apply wall boundary conditions to the velocity field on wall nodes
        (defined as those where nodetype is NodeType.OBSTACLE and NodeType.WALL_BOUNDARY)
        for all frames between start_step and end_step (inclusive).

        Args:
            start_step: int, starting step index to apply wall BC.
            end_step: int, ending step index to apply wall BC. If None, applies to last step.
        """
        if end_step is None:
            end_step = self.length - 1
        wall_mask = self.build_wall_mask()
        for step in range(start_step, end_step + 1):
            velocity = self.meshes[step].point_data["Vitesse"].copy()
            velocity[wall_mask, :] = 0.0  # set velocity to 0 at wall nodes
            self.meshes[step].point_data["Vitesse"] = velocity


def create_trajectory(
    path: str,
    parameters: dict,
    geometry_class: Geometry,
    geometry_args: dict,
    init_features: bool = True,
    output_name: str = "full_traj.xdmf",
) -> Trajectory:
    """
    Create the environment mesh,.
    This uses the geometry class to create the mesh and needs configuration parameters for the mesh generation.
    These include domain parameters, configuration meta parameters, and specific geometry parameters.
    By default, the trajectory's first frame will not have wall boundary conditions applied to the velocity field,
    these are applied only from step 1 onwards so that the BC match the ones seen in training of the GNN.

    Args:
        path: Path to save the mesh at (typically the environment path)
        parameters: Configuration parameters for geometry, mesh, and trajectory.
        See environment_config/panels.json for an example.
        geometry_class: Geometry class instance for mesh creation.
        geometry_args: Arguments for the geometry class (dim, number of objects, angles, etc.)
        init_features: Whether to initialize features (nodetype, levelset) in the mesh.
    Returns:
        The created full trajectory (Trajectory, of trajectory_length frames specified in parameters dict).
    """
    # create empty trajectory
    empty_trajectory = make_empty_trajectory(
        path=path,
        parameters=parameters,
        geometry_class=geometry_class,
        geometry_args=geometry_args,
        init_features=init_features,
    )
    # full trajectory parameters
    trajectory_length = parameters["traj_parameters"].get("trajectory_length", 600)
    inlet_profile_type = parameters["traj_parameters"].get("inlet_type", "uniform")
    inlet_velocity_amplitude = parameters["traj_parameters"].get("inlet_amplitude", 1.0)
    apply_wall_bc = parameters["traj_parameters"].get("apply_wall_bc", True)
    # extend to full trajectory
    full_trajectory = make_full_trajectory_from_empty(
        empty_trajectory=empty_trajectory,
        num_steps=trajectory_length,
        output_name=output_name,
        inlet_profile=inlet_profile_type,
        inlet_amplitude=inlet_velocity_amplitude,
        apply_wall_bc=apply_wall_bc,
        save=False,
    )
    # save
    full_trajectory.save(filename=output_name)
    return full_trajectory


def make_empty_trajectory(
    path: str,
    parameters: dict,
    geometry_class: Geometry,
    geometry_args: dict,
    init_features: bool = True,
) -> Trajectory:
    """
    Create an empty trajectory.
    This uses the geometry class to create the mesh and needs configuration parameters for the mesh generation.
    These include domain parameters, configuration meta parameters, and specific geometry parameters.

    Args:
        path: Path to save the mesh at (typically the environment path)
        parameters: Configuration parameters for geometry, mesh, and trajectory. \
        See environment_config/panels.json for an example.
        geometry_class: Geometry class instance for mesh creation.
        geometry_args: Arguments for the geometry class (dim, number of objects, angles, etc.)
        init_features: Whether to initialize features (nodetype, levelset) in the mesh.
    Returns:
        The created full trajectory (Trajectory, of trajectory_length frames specified in parameters dict).
    """
    dim = parameters.get("dim", 2)
    dt = parameters["traj_parameters"].get("dt", None)
    if dt is None:
        raise KeyError(
            "You must include a 'dt' value in 'traj_parameters' to indicate timestep between traj frames"
        )
    num_objects_param_name = ("num_" + geometry_class.__name__).lower()
    num_objects_blm = geometry_args.get(num_objects_param_name, None)

    # init cimlib env
    blm_adaptation = parameters["traj_parameters"].get("mesh_adapt", False)
    if init_features or blm_adaptation:
        cimlib_env = CimlibEnv(parameters=parameters, path=path)
        cimlib_env.prep()

        geometry = geometry_class(
            path=cimlib_env.dir, parameters_dict=parameters, **geometry_args
        )
        geometry.set_meshing_options()
        geometry.apply_box2params()

        _ = geometry.create_domain(save_mesh=True, dim_mesh=dim)
        _ = geometry.create_object(force_model="", save_mesh=True, dim_mesh=dim)

        geometry.finalize()

        convert_gmsh_to_mtc(
            input=os.path.join(cimlib_env.dir, "object.msh"),
            output=os.path.join(cimlib_env.dir, "object.t"),
            verbose=False,
        )
        convert_gmsh_to_mtc(
            input=os.path.join(cimlib_env.dir, "domain.msh"),
            output=os.path.join(cimlib_env.dir, "domain.t"),
            verbose=False,
        )
        move_meshes(
            output_directory=os.path.join(cimlib_env.dir, "meshes"),
            extensions=[".t"],
            source_directory=cimlib_env.dir,
        )
        move_meshes(
            output_directory=os.path.join(cimlib_env.dir, "meshes_GMSH"),
            extensions=[".msh", ".geo_unrolled", ".vtk"],
            source_directory=cimlib_env.dir,
        )

        if blm_adaptation and num_objects_blm is not None:
            cimlib_env.apply_blm_boxsize(num_objects=num_objects_blm)

        meshes, times = cimlib_env.run()
        cimlib_env.cleanup()
        trajectory = Trajectory(meshes=meshes, times=times, timestep=dt)
    else:
        geometry = geometry_class(
            path=path, parameters_dict=parameters, **geometry_args
        )
        geometry.set_meshing_options()

        _ = geometry.create_domain(save_mesh=True, dim_mesh=dim)

        geometry.finalize()
        try:
            meshes, times = msh_to_meshes(
                msh_file_path=os.path.join(geometry.path, "domain.msh"), time=0.0
            )
            trajectory = Trajectory(meshes=meshes, times=times, timestep=dt)
        except Exception as e:
            raise RuntimeError(
                f"Error loading mesh from {os.path.join(geometry.path, 'domain.msh')}"
            ) from e
    return trajectory


def make_full_trajectory_from_empty(
    empty_trajectory: Trajectory,
    num_steps: int = 600,
    output_name: str = "traj_full.xdmf",
    inlet_profile: str = "uniform",
    inlet_amplitude: int = 1.0,
    apply_wall_bc: bool = True,
    save: bool = True,
) -> Trajectory:
    """
    Create a full and boundary conditioned (inlet profile) trajectory from empty_trajectory for num_steps timesteps and
    saves as an xdmf file in desired location with desired name.

    Args:
        trajectory (Trajectory): initial trajectory to create from, should contain only a single frame.
        num_steps (int): number of timesteps in the trajectory.
        output_name (str): path to save the output XDMF file (full path with filename and extension).
        inlet_profile (str): inlet profile type ("uniform" or "abl").
        inlet_amplitude (float): freestream velocity (i.e. max amplitude of the inlet profile).
        apply_wall_bc (bool): whether to apply wall BC to velocity field on wall nodes from first step onwards \
        (defined as those where nodetype is NodeType.OBSTACLE and NodeType.WALL_BOUNDARY).
        save (bool): whether to save the trajectory to xdmf.

    Returns:
        The full trajectory (Trajectory) created.
    """
    # check
    if len(empty_trajectory) > 1:
        raise ValueError(
            "A non-empty trajectory was provided, please provide a single-frame trajectory"
        )
    # bc
    empty_trajectory.init_boundary_conditions(
        inlet_profile=inlet_profile, u_inf=inlet_amplitude, wall_bc=False
    )
    # verify all base features are present
    if not set(BASE_FEATURES).issubset(set(empty_trajectory.get_fields())):
        raise KeyError(
            "Trajectory is missing base features: "
            + str(set(BASE_FEATURES) - set(empty_trajectory.get_fields()))
        )
    # extend
    empty_trajectory.extend(total_length=num_steps, which="first")
    # wall bc
    if apply_wall_bc:
        empty_trajectory.apply_wall_bc(start_step=0, end_step=None)
    # save
    if save:
        empty_trajectory.save(filename=output_name)
    return empty_trajectory


def make_full_traj_from_vtu(
    vtu_path: str,
    timestep: float = 0.1,
    num_steps: int = 600,
    output_name: str = "traj_full.xdmf",
    inlet_profile: str = "from_vtu",
    inlet_amplitude: float = None,
    apply_wall_bc: bool = True,
    save: bool = True,
) -> Trajectory:
    """
    Create a full and boundary conditioned (inlet profile) trajectory from a single vtu for num_steps timesteps and
    saves as an xdmf file in desired location with desired name.
    Note that ideally, all boundary conditions should come from vtu. If vtu contains only mesh info, defaults to same as
    the `make_full_traj_from_empty` function.

    Args:
        trajectory (Trajectory): initial trajectory to create from, should contain only a single frame.
        num_steps: number of timesteps in the trajectory.
        output_name: path to save the output XDMF file (full path with filename and extension).
        inlet_profile: inlet profile type ("uniform" or "abl").
        inlet_amplitude: freestream velocity (i.e. max amplitude of the inlet profile).
        apply_wall_bc: whether to apply wall BC to velocity field on wall nodes from first step onwards \
        (defined as those where nodetype is NodeType.OBSTACLE and NodeType.WALL_BOUNDARY).
        save: whether to save the trajectory to xdmf.

    Returns:
        The full trajectory (Trajectory) created.
    """
    # create trajectory from vtu
    meshes, times = vtu_to_meshes(vtu_file_path=vtu_path, time=0.0)
    empty_trajectory = Trajectory(meshes=meshes, times=times, timestep=timestep)

    # check
    if len(empty_trajectory) > 1:
        raise ValueError(
            "Failed to create single-frame trajectory from vtu, please check the vtu file"
        )

    # bc
    if inlet_profile is None and inlet_amplitude is None:
        empty_trajectory.init_boundary_conditions(
            inlet_profile="from_vtu", wall_bc=False
        )
    else:  # override with provided inlet profile
        empty_trajectory.init_boundary_conditions(
            inlet_profile=inlet_profile,
            u_inf=inlet_amplitude,
            wall_bc=False,
        )
    # verify all base features are present
    if not set(BASE_FEATURES).issubset(set(empty_trajectory.get_fields())):
        raise KeyError(
            "Trajectory is missing base features: "
            + str(set(BASE_FEATURES) - set(empty_trajectory.get_fields()))
        )
    # extend
    empty_trajectory.extend(total_length=num_steps, which="first")
    # wall bc
    if apply_wall_bc:
        empty_trajectory.apply_wall_bc(start_step=0, end_step=None)
    # save
    if save:
        empty_trajectory.save(filename=output_name)
    return empty_trajectory
