import shutil
import subprocess

import json
import meshio
import numpy as np

from graphdrl.utils.meshio_mesh import meshes_to_xdmf


def simple_xdmf(tmp_path):
    """Create a simple xdmf file with one mesh, all base features, and 2 timesteps."""
    path = str(tmp_path / "traj_0.xdmf")
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    cells = [("triangle", np.array([[0, 1, 2]]))]
    vitesse = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]])
    pression = np.array([[0.0], [0.0], [0.0]])
    levelset = np.array([[-0.1], [-0.1], [0.0]])
    nodetype = np.array([[1], [6], [0]])
    mesh = meshio.Mesh(
        points=points,
        cells=cells,
        point_data={
            "Vitesse": vitesse,
            "Pression": pression,
            "LevelSet": levelset,
            "NodeType": nodetype,
        },
    )
    meshes = [mesh] * 3
    times = [0.0, 0.1, 0.2]
    meshes_to_xdmf(filename=path, meshes=meshes, timestep=times, verbose=False)


def mock_meta(tmp_path):
    meta = {
        "dt": 0.1,
        "features": {
            "cells": {"type": "static", "shape": [2, -1, 3], "dtype": "int32"},
            "mesh_pos": {"type": "static", "shape": [2, -1, 3], "dtype": "float32"},
            "Vitesse": {"type": "dynamic", "shape": [2, -1, 3], "dtype": "float32"},
            "Pression": {"type": "dynamic", "shape": [2, -1, 1], "dtype": "float32"},
            "LevelSet": {"type": "static", "shape": [2, -1, 1], "dtype": "float32"},
            "NodeType": {"type": "static", "shape": [2, -1, 1], "dtype": "float32"},
        },
        "field_names": [
            "cells",
            "mesh_pos",
            "Vitesse",
            "Pression",
            "LevelSet",
            "NodeType",
        ],
        "trajectory_length": 2,
    }
    path = str(tmp_path / "mock_meta.json")
    with open(path, "w") as f:
        json.dump(meta, f)


def small_params(tmp_path):
    meta = {
        "dataset": {
            "extension": "xdmf",
            "xdmf_folder": str(tmp_path),
            "meta_path": str(tmp_path / "mock_meta.json"),
            "khop": 1,
            "new_edges_ratio": 0,
        },
        "model": {
            "type": "transformer",
            "message_passing_num": 15,
            "hidden_size": 128,
            "node_input_size": 7,
            "output_size": 3,
            "edge_input_size": 0,
            "num_heads": 4,
        },
        "index": {
            "feature_index_start": 0,
            "feature_index_end": 7,
            "output_index_start": 0,
            "output_index_end": 3,
            "node_type_index": 7,
        },
        "transformations": {
            "preprocessing": {
                "noise": 0.0,
                "noise_index_start": [0],
                "noise_index_end": [3],
                "masking": 0,
            },
            "world_pos_parameters": {
                "use": False,
                "world_pos_index_start": 0,
                "world_pos_index_end": 3,
            },
        },
    }
    path = str(tmp_path / "mock_predict.json")
    with open(path, "w") as f:
        json.dump(meta, f)


def test_gnn_import(tmp_path):
    """Test that graphphysics can be imported."""
    import importlib

    module = importlib.import_module("graphphysics")
    # Ensure the module was actually imported and is usable
    assert module is not None


def test_gnn_setup_and_prediction(tmp_path):
    """Test GNN setup and a mock prediction run."""
    # Assuming testing is not done on full env with DGL etc
    # so we load checkpoint with strict=False
    # so that missing keys are ignored (prediction quality is of no concern here)

    # Create simple xdmf and meta files
    simple_xdmf(tmp_path)
    assert (tmp_path / "traj_0.xdmf").exists()
    assert (tmp_path / "traj_0.h5").exists()

    mock_meta(tmp_path)
    assert (tmp_path / "mock_meta.json").exists()

    small_params(tmp_path)
    assert (tmp_path / "mock_predict.json").exists()

    shutil.copy(
        "environment_config/models/mock_model.ckpt", str(tmp_path / "mock_model.ckpt")
    )
    assert (tmp_path / "mock_model.ckpt").exists()

    # run prediction via subprocess
    predict_command = [
        "python",
        "-m",
        "graphphysics.predict",
        f"--predict_parameters_path={tmp_path / 'mock_predict.json'}",
        f"--model_path={tmp_path / 'mock_model.ckpt'}",
        "--no_edge_feature",
        f"--prediction_save_path={tmp_path}",
        "--no_strict_load",  # just during testing
    ]
    try:
        subprocess.run(
            predict_command,
            check=True,
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Subprocess failed with return code {e.returncode}. " f"stderr: {e.stderr}"
        ) from e
    assert (tmp_path / "graph_0.xdmf").exists()
    assert (tmp_path / "graph_0.h5").exists()

    # Clean up generated files
    (tmp_path / "traj_0.xdmf").unlink()
    (tmp_path / "graph_0.xdmf").unlink()
    (tmp_path / "graph_0.h5").unlink()
    (tmp_path / "mock_meta.json").unlink()
    (tmp_path / "mock_predict.json").unlink()
    (tmp_path / "mock_model.ckpt").unlink()
