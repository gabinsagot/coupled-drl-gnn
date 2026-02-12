import meshio
import numpy as np

from graphdrl.utils.meshio_mesh import meshes_to_xdmf
from graphdrl.utils.reward import (
    compute_reward_case,
    compute_time_avg_velocity_components,
    create_reward_box_mask,
    create_single_box_mask,
    compute_velocity_rms_fluctuation,
)


def create_meshes():
    # create a case with known reward
    mesh_points = np.array(
        [
            [1.5, 1.5, 0.0],  # point 1: inside box
            [1.8, 1.3, 0.0],  # point 2: inside box
            [0.5, 0.5, 0.0],  # point 3: outside box
            [3.0, 3.0, 0.0],  # point 4: outside box
            [2.5, 0.5, 0.0],  # point 5: outside box
        ]
    )
    cells = [("triangle", [[0, 1, 2], [1, 2, 3], [2, 3, 4]])]

    velocity_field_t0 = np.array(
        [
            [1.0, 0.0],  # point 1
            [0.0, 1.0],  # point 2
            [1.0, 1.0],  # point 3
            [0.5, 1.0],  # point 4
            [0.0, 0.0],  # point 5
        ]
    )
    velocity_field_t1 = np.array(
        [
            [0.0, 1.0],  # point 1
            [1.0, 0.0],  # point 2
            [1.0, 0.0],  # point 3
            [0.5, 3.0],  # point 4
            [0.0, 0.0],  # point 5
        ]
    )
    velocity_field_t2 = np.array(
        [
            [0.5, 0.5],  # point 1
            [0.5, 0.5],  # point 2
            [1.0, -1.0],  # point 3
            [0.5, -1.0],  # point 4
            [0.0, 1.5],  # point 5
        ]
    )
    meshes = []
    times = [0.0, 0.1, 0.2]
    # Create xdmf file with 3 timesteps
    for velocity_field in [velocity_field_t0, velocity_field_t1, velocity_field_t2]:
        mesh = meshio.Mesh(
            points=mesh_points, cells=cells, point_data={"velocity": velocity_field}
        )
        meshes.append(mesh)
    return meshes, times


def create_case(tmp_path):
    meshes, times = create_meshes()
    meshes_to_xdmf(
        filename=str(tmp_path / "test_reward_case.xdmf"), meshes=meshes, timestep=times
    )


def create_boxes():
    return {"box1": {"x_min": 1.0, "y_min": 1.0, "dx": 1.0, "dy": 1.0, "weight": 1.0}}


def test_create_single_box_mask():
    meshes, _ = create_meshes()
    boxes = create_boxes().get("box1")
    box_mask = create_single_box_mask(mesh=meshes[0], box_dict=boxes)
    expected_mask = np.array([1, 1, 0, 0, 0], dtype=bool)
    assert np.array_equal(box_mask, expected_mask)


def test_create_reward_box_mask():
    meshes, _ = create_meshes()
    boxes = {
        "box1": {"x_min": 1.0, "y_min": 1.0, "dx": 1.0, "dy": 1.0, "weight": 1.0},
        "box2": {"x_min": 0.0, "y_min": 0.0, "dx": 2.0, "dy": 2.0, "weight": 1.0},
    }
    box_mask = create_reward_box_mask(mesh=meshes[0], box_dict=boxes)
    expected_mask = np.array([1, 1, 1, 0, 0], dtype=bool)
    assert np.array_equal(box_mask, expected_mask)


def test_compute_time_avg_velocity_components():
    meshes, _ = create_meshes()
    expected_avg_vx = np.array([0.5, 0.5, 1.0, 0.5, 0.0])
    expected_avg_vy = np.array([0.5, 0.5, 0.0, 1.0, 0.5])
    avg_vx, avg_vy = compute_time_avg_velocity_components(
        meshes=meshes, velocity_field="velocity", start_step=0, mask=None
    )
    print(avg_vx, avg_vy)
    assert np.allclose(avg_vx, expected_avg_vx)
    assert np.allclose(avg_vy, expected_avg_vy)


def test_compute_velocity_rms_fluctuation():
    meshes, _ = create_meshes()
    boxes = create_boxes()
    box_mask = create_reward_box_mask(mesh=meshes[0], box_dict=boxes)
    # theoretical rms fluctuation for point1,2 is 0.57735027
    expected_rms_fluctuation = 0.57735027
    rms_fluctuation = compute_velocity_rms_fluctuation(
        meshes=meshes,
        velocity_field="velocity",
        start_step=0,
        mask_fluc=box_mask,
        mask_avg=None,
    )
    assert np.isclose(rms_fluctuation, expected_rms_fluctuation)


def test_compute_reward_case(tmp_path):
    create_case(tmp_path)
    boxes = create_boxes()
    reward = compute_reward_case(
        case_xdmf_path=str(tmp_path / "test_reward_case.xdmf"),
        box_dict=boxes,
        velocity_field="velocity",
        start_step=0,
        verbose=False,
    )
    # theoretical rms fluctuation for point1,2 is 0.57735027
    # reward is -rms_fluctuation
    expected_reward = -0.57735027
    assert np.isclose(reward, expected_reward)
    (tmp_path / "test_reward_case.*").unlink(missing_ok=True)
