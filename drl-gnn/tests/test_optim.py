import os

from graphdrl.utils.file_handling import adjust_driver_to_architecture
from optim_config.panels import panels
from tests.test_gnn import mock_meta, small_params


def test_optim_panels(tmp_path):
    panel = panels(path=str(tmp_path))
    assert panel is not None

    # check init panels
    assert panel.name == "panels"
    assert hasattr(panel, "action_type")
    assert hasattr(panel, "n_panels")
    assert hasattr(panel, "x_values")
    assert hasattr(panel, "act_size")
    assert hasattr(panel, "obs_size")
    assert hasattr(panel, "obs")
    assert panel.episode == 0

    # check case params
    assert hasattr(panel, "case")
    assert panel.case is not None
    assert hasattr(panel.case, "params")
    assert panel.case.params is not None
    assert hasattr(panel.case, "geometry_class")
    assert panel.case.geometry_class is not None
    panel.case._sanity_checks()

    # check gnn params
    assert hasattr(panel, "gnn_parameters")
    assert hasattr(panel.gnn_parameters, "model_path")
    assert panel.gnn_parameters.model_path is not None
    assert hasattr(panel.gnn_parameters, "no_edge_feature")
    assert hasattr(panel.gnn_parameters, "traj_config_path")
    assert panel.gnn_parameters.traj_config_path is not None
    assert hasattr(panel.gnn_parameters, "predict_config_path")
    assert panel.gnn_parameters.predict_config_path is not None

    # check reward params
    assert hasattr(panel, "reward_parameters")
    assert panel.reward_parameters is not None
    panel.reward_parameters._sanity_checks()


def test_write_actions(tmp_path):
    panel = panels(path=str(tmp_path))
    panel.setup_env_folder(ep=0)
    actions = [0.0] * panel.n_panels
    panel.write_actions(actions, ep=0)
    assert (tmp_path / "actions.log").exists()
    with open(str(tmp_path / "actions.log"), "r") as f:
        lines = f.readlines()
    assert len(lines) == 2  # header + 1 line
    assert all(float(val) == 0.0 for val in lines[1].strip().split("\t"))


def test_setup_env_folder(tmp_path):
    panel = panels(path=str(tmp_path))
    panel.setup_env_folder(ep=0)
    assert hasattr(panel, "output_path")
    assert panel.output_path is not None
    assert os.path.exists(os.path.join(panel.output_path, "predict_config.json"))
    assert os.path.exists(os.path.join(panel.output_path, "traj_config.json"))


def test_create_env_trajectory(tmp_path):
    panel = panels(path=str(tmp_path))
    panel.case.params["traj_parameters"]["mesh_adapt"] = False
    panel.case.needs_cimlib_init = True
    panel.case.params = adjust_driver_to_architecture(panel.case.params)
    panel.setup_env_folder(ep=0)
    panel.create_env_trajectory(actions=[0.0] * panel.n_panels, ep=0)
    assert os.path.exists(os.path.join(panel.output_path, f"{panel.name}_0.xdmf"))
    assert os.path.exists(os.path.join(panel.output_path, f"{panel.name}_0.h5"))


def test_gnn_prediction(tmp_path):
    panel = panels(path=str(tmp_path))
    panel.case.params["traj_parameters"]["mesh_adapt"] = False
    panel.case.needs_cimlib_init = True
    panel.case.params = adjust_driver_to_architecture(panel.case.params)
    # overwrite gnn params meta
    mock_meta(tmp_path)
    small_params(tmp_path)
    panel.gnn_parameters.traj_config_path = str(tmp_path / "mock_meta.json")
    panel.gnn_parameters.predict_config_path = str(tmp_path / "mock_predict.json")
    panel.gnn_parameters.no_edge_feature = True
    panel.gnn_parameters.model_path = os.path.abspath(
        "environment_config/models/mock_model.ckpt"
    )
    # setup and predict
    panel.setup_env_folder(ep=0)
    panel.create_env_trajectory(actions=[0.0] * panel.n_panels, ep=0)
    panel.gnn_prediction(strict_load=False)
    assert os.path.exists(os.path.join(panel.output_path, "graph_0.xdmf"))
    assert os.path.exists(os.path.join(panel.output_path, "graph_0.h5"))
