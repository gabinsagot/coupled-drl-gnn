import meshio
import numpy as np

from graphdrl.environment.geometry import Panels
from graphdrl.environment.trajectory import (
    Trajectory,
    create_trajectory,
    make_empty_trajectory,
    make_full_trajectory_from_empty,
    BASE_FEATURES,
)
from graphdrl.utils.file_handling import adjust_driver_to_architecture
from graphdrl.utils.meshio_mesh import msh_to_meshes, vtu_to_meshes, xdmf_to_meshes


def small_params():
    return adjust_driver_to_architecture(
        {
            "geometry_parameters": {
                "origin": [0, 1.5, 0],
                "chord": 1.0,
                "span": 0.0,
                "thickness": 0.1,
                "spacing": 2.0,
            },
            "domain_parameters": {
                "dx": 50.0,
                "dy": 10.0,
                "dz": 0.0,
                "origin_x": -10.0,
                "origin_y": 0.0,
                "origin_z": 0.0,
            },
            "traj_parameters": {
                "dt": 0.1,
                "trajectory_length": 3,
                "Hbox123": [0.01, 0.1, 2.0],
                "mesh_adapt": False,
                "inlet_type": "uniform",
                "inlet_amplitude": 2.0,
                "driver": "environment_config/driver/ubuntu64_cfd_driver",
            },
        }
    )


def small_params_blm():
    return adjust_driver_to_architecture(
        {
            "geometry_parameters": {
                "origin": [0, 1.5, 0],
                "chord": 1.0,
                "span": 0.0,
                "thickness": 0.1,
                "spacing": 2.0,
            },
            "domain_parameters": {
                "dx": 50.0,
                "dy": 10.0,
                "dz": 0.0,
                "origin_x": -10.0,
                "origin_y": 0.0,
                "origin_z": 0.0,
            },
            "traj_parameters": {
                "dt": 0.1,
                "trajectory_length": 3,
                "Hbox123": [0.05, 1.0, 2.0],
                "mesh_adapt": True,
                "inlet_type": "uniform",
                "inlet_amplitude": 2.0,
                "driver": "environment_config/driver/ubuntu64_cfd_driver",
            },
        }
    )


def make_mesh_with_fields():
    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    cells = [("line", np.array([[0, 1]]))]
    point_data = {"Vitesse": np.zeros((2, 2)), "Pression": np.zeros((2, 1))}
    return meshio.Mesh(points=points, cells=cells, point_data=point_data)


def test_trajectory_init_and_getters():
    mesh = make_mesh_with_fields()
    meshes = [mesh, mesh]
    times = [0.0, 0.1]
    traj = Trajectory(meshes=meshes, times=times, timestep=0.1)
    assert len(traj) == 2
    assert traj.timestep == 0.1
    assert traj.get_fields() == ["Vitesse", "Pression"]
    assert traj.get_mesh_at_step(0) is meshes[0]
    assert traj.get_mesh_at_time(0.1) is meshes[1]


def test_trajectory_invalid_init():
    mesh = make_mesh_with_fields()
    try:
        Trajectory(meshes=[mesh], times=[0.0])
    except ValueError:
        # expected because timestep is required when only one frame
        pass
    else:
        raise AssertionError(
            "Expected ValueError for missing timestep with single frame"
        )


def test_extend_and_init_bc():
    mesh = make_mesh_with_fields()
    meshes = [mesh]
    times = [0.0]
    traj = Trajectory(meshes=meshes, times=times, timestep=0.1)
    # init boundary on single-frame traj should work
    traj.init_boundary_conditions(inlet_profile="uniform", u_inf=2.0)
    assert "Vitesse" in traj.meshes[0].point_data
    assert np.allclose(traj.meshes[0].point_data["Vitesse"][:, 0], 2.0)
    assert np.allclose(traj.meshes[0].point_data["Vitesse"][:, 1], 0.0)
    assert np.allclose(traj.meshes[0].point_data["Vitesse"][:, 2], 0.0)
    assert np.allclose(traj.meshes[0].point_data["Pression"][:, 0], 0.0)
    # extend
    traj.extend(total_length=3, which="first")
    assert len(traj) == 3


def test_trajectory_from_vtk(tmp_path):
    params = small_params()
    angles = [0.0, 30.0]
    panels = Panels(
        parameters_dict=params, angles=angles, num_panels=2, dim=2, path=str(tmp_path)
    )
    # init
    panels.set_meshing_options()
    # Check that files are created
    _ = panels.create_domain(save_mesh=True, dim_mesh=2)
    panels.finalize()
    # Check that the output files are created
    assert (tmp_path / "domain.msh").exists()
    assert (tmp_path / "domain.vtk").exists()
    # create trajectory from the vtk files
    meshes, times = vtu_to_meshes(str(tmp_path / "domain.vtk"))
    (tmp_path / "domain.*").unlink(missing_ok=True)
    traj = Trajectory(meshes=meshes, times=times, timestep=0.1)
    assert len(traj) == 1
    assert traj.get_fields() == []
    # init boundary conditions
    traj.init_boundary_conditions(inlet_profile="uniform", u_inf=2.0)
    assert "Vitesse" in traj.meshes[0].point_data
    assert "Pression" in traj.meshes[0].point_data
    assert np.allclose(traj.meshes[0].point_data["Vitesse"][:, 0], 2.0)
    assert np.allclose(traj.meshes[0].point_data["Vitesse"][:, 1], 0.0)
    assert np.allclose(traj.meshes[0].point_data["Vitesse"][:, 2], 0.0)
    assert np.allclose(traj.meshes[0].point_data["Pression"][:, 0], 0.0)
    # extend
    traj.extend(total_length=3, which="first")
    assert len(traj) == 3
    assert traj.get_fields() == ["Vitesse", "Pression"]
    assert np.allclose(traj.meshes[-1].point_data["Vitesse"][:, 0], 2.0)
    assert np.allclose(traj.meshes[-1].point_data["Vitesse"][:, 1], 0.0)
    assert np.allclose(traj.meshes[-1].point_data["Vitesse"][:, 2], 0.0)
    assert np.allclose(traj.meshes[-1].point_data["Pression"][:, 0], 0.0)


def test_trajectory_save(tmp_path):
    params = small_params()
    angles = [0.0, 30.0]
    panels = Panels(
        parameters_dict=params, angles=angles, num_panels=2, dim=2, path=str(tmp_path)
    )
    # init
    panels.set_meshing_options()
    # Check that files are created
    _ = panels.create_domain(save_mesh=True, dim_mesh=2)
    panels.finalize()
    # create trajectory from the msh files
    meshes, times = msh_to_meshes(str(tmp_path / "domain.msh"))
    traj = Trajectory(meshes=meshes, times=times, timestep=0.1)
    # init boundary conditions
    traj.init_boundary_conditions(inlet_profile="uniform", u_inf=2.0)
    # extend
    traj.extend(total_length=3, which="first")
    # check dim of meshes
    assert traj.meshes[0].points.shape[1] == 3
    assert traj.meshes[1].points.shape[1] == 3
    assert traj.meshes[2].points.shape[1] == 3
    assert traj.meshes[0].cells[0].type == "triangle"

    # save
    traj.save(filename=str(tmp_path / "traj.xdmf"))
    assert (tmp_path / "traj.xdmf").exists()
    assert (tmp_path / "traj.h5").exists()

    # read back
    meshes2, times2 = xdmf_to_meshes(str(tmp_path / "traj.xdmf"))
    assert len(meshes2) == 3
    assert len(times2) == 3
    assert np.allclose(times2, [0.0, 0.1, 0.2])
    assert np.allclose(meshes2[0].point_data["Vitesse"][:, 0], 2.0)
    assert np.allclose(meshes2[0].point_data["Vitesse"][:, 1], 0.0)
    assert np.allclose(meshes2[0].point_data["Pression"][:, 0], 0.0)

    (tmp_path / "traj.*").unlink(missing_ok=True)


def test_make_empty_trajectory_no_init_features(tmp_path):
    params = small_params()
    traj = make_empty_trajectory(
        path=str(tmp_path),
        parameters=params,
        geometry_class=Panels,
        geometry_args=dict(angles=[0.0, 30.0], num_panels=2, dim=2),
        init_features=False,
    )
    assert traj is not None
    assert len(traj) == 1
    assert traj.get_fields() == []
    assert traj.meshes[0].points.shape[1] == 3
    assert traj.meshes[0].cells[0].type == "triangle"


def test_make_empty_trajectory_with_init_features(tmp_path):
    params = small_params()
    traj = make_empty_trajectory(
        path=str(tmp_path),
        parameters=params,
        geometry_class=Panels,
        geometry_args=dict(angles=[0.0, 30.0], num_panels=2, dim=2),
        init_features=True,
    )
    assert traj is not None
    assert len(traj) == 1
    assert ({"NodeType", "LevelSetObject"}).issubset(set(traj.get_fields()))
    assert traj.meshes[0].points.shape[1] == 3


def test_make_empty_trajectory_with_blm(tmp_path):
    params = small_params_blm()
    traj = make_empty_trajectory(
        path=str(tmp_path),
        parameters=params,
        geometry_class=Panels,
        geometry_args=dict(angles=[0.0, 30.0], num_panels=2, dim=2),
        init_features=True,
    )
    assert traj is not None
    assert len(traj) == 1
    assert ({"NodeType", "LevelSetObject"}).issubset(set(traj.get_fields()))
    assert traj.meshes[0].points.shape[1] == 3


def test_make_full_trajectory_from_empty(tmp_path):
    params = small_params()
    traj = make_empty_trajectory(
        path=str(tmp_path),
        parameters=params,
        geometry_class=Panels,
        geometry_args=dict(angles=[0.0, 30.0], num_panels=2, dim=2),
        init_features=True,
    )
    assert traj is not None
    assert len(traj) == 1
    assert ({"NodeType", "LevelSetObject"}).issubset(set(traj.get_fields()))

    traj = make_full_trajectory_from_empty(
        empty_trajectory=traj,
        num_steps=3,
        output_name=str(tmp_path / "full_traj.xdmf"),
        inlet_profile="uniform",
        inlet_amplitude=2.0,
        apply_wall_bc=True,
    )
    assert traj is not None
    assert len(traj) == 3
    assert ({"NodeType", "LevelSetObject", "Vitesse", "Pression"}).issubset(
        set(traj.get_fields())
    )
    assert traj.meshes[0].points.shape[1] == 3
    assert (tmp_path / "full_traj.xdmf").exists()
    assert (tmp_path / "full_traj.h5").exists()

    # read back
    meshes2, times2 = xdmf_to_meshes(str(tmp_path / "full_traj.xdmf"))
    assert len(meshes2) == 3
    assert len(times2) == 3
    assert np.allclose(times2, [0.0, 0.1, 0.2])
    assert np.allclose(meshes2[0].point_data["Vitesse"][:, 0], 2.0)
    assert np.allclose(meshes2[0].point_data["Vitesse"][:, 1], 0.0)
    assert np.allclose(meshes2[0].point_data["Pression"][:, 0], 0.0)
    # test wall bc
    wall_mask = traj.build_wall_mask()
    assert wall_mask is not None
    assert np.allclose(meshes2[1].point_data["Vitesse"][wall_mask, 0], 0.0)

    (tmp_path / "full_traj.*").unlink(missing_ok=True)


def test_create_trajectory(tmp_path):
    params = small_params()
    traj = create_trajectory(
        path=str(tmp_path),
        parameters=params,
        geometry_class=Panels,
        geometry_args=dict(angles=[0.0, 30.0], num_panels=2, dim=2),
        init_features=True,
        output_name=str(tmp_path / "full_traj_bis.xdmf"),
    )
    assert traj is not None
    assert len(traj) == small_params()["traj_parameters"]["trajectory_length"]
    assert (set(BASE_FEATURES)).issubset(set(traj.get_fields()))
    assert traj.meshes[0].points.shape[1] == 3
    assert traj.meshes[0].cells[0].type == "triangle"
    assert (tmp_path / "full_traj_bis.xdmf").exists()
    assert (tmp_path / "full_traj_bis.h5").exists()

    # read back
    meshes2, times2 = xdmf_to_meshes(str(tmp_path / "full_traj_bis.xdmf"))
    assert len(meshes2) == 3
    assert len(times2) == 3
    assert np.allclose(times2, [0.0, 0.1, 0.2])
    assert ({"Vitesse", "Pression"}).issubset(set(traj.get_fields()))
    assert np.allclose(meshes2[0].point_data["Vitesse"][:, 0], 2.0)
    assert np.allclose(meshes2[0].point_data["Vitesse"][:, 1], 0.0)
    assert np.allclose(meshes2[0].point_data["Pression"][:, 0], 0.0)
    (tmp_path / "full_traj_bis.*").unlink(missing_ok=True)
    wall_mask = traj.build_wall_mask()
    assert wall_mask is not None
    assert np.allclose(meshes2[1].point_data["Vitesse"][wall_mask, 0], 0.0)


def test_create_trajectory_no_wall_bc(tmp_path):
    params = small_params()
    params["traj_parameters"]["apply_wall_bc"] = False
    traj = create_trajectory(
        path=str(tmp_path),
        parameters=params,
        geometry_class=Panels,
        geometry_args=dict(angles=[0.0, 30.0], num_panels=2, dim=2),
        init_features=True,
        output_name=str(tmp_path / "full_traj_bis.xdmf"),
    )
    assert traj is not None
    assert len(traj) == small_params()["traj_parameters"]["trajectory_length"]
    assert (set(BASE_FEATURES)).issubset(set(traj.get_fields()))
    assert traj.meshes[0].points.shape[1] == 3
    assert traj.meshes[0].cells[0].type == "triangle"
    assert (tmp_path / "full_traj_bis.xdmf").exists()
    assert (tmp_path / "full_traj_bis.h5").exists()

    # read back
    meshes2, times2 = xdmf_to_meshes(str(tmp_path / "full_traj_bis.xdmf"))
    assert len(meshes2) == 3
    assert len(times2) == 3
    assert np.allclose(times2, [0.0, 0.1, 0.2])
    assert ({"Vitesse", "Pression"}).issubset(set(traj.get_fields()))
    assert np.allclose(meshes2[0].point_data["Vitesse"][:, 0], 2.0)
    assert np.allclose(meshes2[0].point_data["Vitesse"][:, 1], 0.0)
    assert np.allclose(meshes2[0].point_data["Pression"][:, 0], 0.0)
    (tmp_path / "full_traj_bis.*").unlink(missing_ok=True)
    wall_mask = traj.build_wall_mask()
    assert wall_mask is not None
    assert np.allclose(meshes2[1].point_data["Vitesse"][wall_mask, 0], 2.0)
