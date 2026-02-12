from graphdrl.environment.geometry import Panels


def small_params():
    return {
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
        "traj_parameters": {"Hbox123": [0.01, 0.1, 2.0], "mesh_adapt": False},
    }


def test_panels_objects_and_meshdicts(tmp_path):
    params = small_params()
    angles = [0.0, 30.0]
    panels = Panels(
        parameters_dict=params, angles=angles, num_panels=2, dim=2, path=str(tmp_path)
    )
    # objects_dict should have two panels
    assert len(panels.objects_dict) == 2
    assert panels.n_panels == 2
    mesh_dict = panels.create_mesh_dicts(panels.objects_dict)
    assert "panel1" in mesh_dict and "panel2" in mesh_dict
    origins = panels.objects_origins(panels.objects_dict)
    assert isinstance(origins, list) and len(origins) == 2
    dom_dims = panels.get_domain_dimensions()
    assert dom_dims == [50.0, 10.0, 0.0]


def test_panels_mesh_outputs(tmp_path):
    params = small_params()
    angles = [0.0, 30.0]
    panels = Panels(
        parameters_dict=params, angles=angles, num_panels=2, dim=2, path=str(tmp_path)
    )
    # init
    panels.set_meshing_options()
    # Check that files are created
    _ = panels.create_domain(save_mesh=True, dim_mesh=2)
    _ = panels.create_object(force_model="", save_mesh=True, dim_mesh=2)
    panels.finalize()
    # Check that the output files are created
    assert (tmp_path / "domain.msh").exists()
    assert (tmp_path / "domain.vtk").exists()
    assert (tmp_path / "object.msh").exists()
    assert (tmp_path / "object.vtk").exists()
    # cleanup
    (tmp_path / "domain.*").unlink(missing_ok=True)
    (tmp_path / "object.*").unlink(missing_ok=True)
