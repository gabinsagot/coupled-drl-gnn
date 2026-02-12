from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import os
from typing import Dict, List, Tuple

import meshio
import numpy as np
import pandas as pd
from tqdm import tqdm

from graphdrl.environment.trajectory import Trajectory
from graphdrl.utils.meshio_mesh import xdmf_to_meshes


def compute_boundary_edges(triangles: np.ndarray) -> np.ndarray:
    """Find edges belonging to only one triangle (vectorized, no Python Counter)."""
    edges = np.vstack(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ]
    )
    edges_sorted = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    return boundary_edges  # (B,2)


def build_object_containers(
    n_obj: int, x0: float, y0: float, chord: float, spacing: float
) -> Dict[int, Dict[str, float]]:
    """
    Build a dictionary of box coordinates for each panel object.

    Args:
        n_obj (int): number of objects (panels).
        x0 (float): x origin of first panel center of rotation.
        y0 (float): y origin of first panel center of rotation.
        chord (float): chord length of each panel.
        spacing (float): spacing between panel rotation centers along the x-axis.

    Returns:
        object_containers: dict of dict of box dimensions containing each object
            "obj_id" (int): {"x_min": float,"y_min": float,"dx": float,"dy": float}
        where obj_id is a unique int id for the object. Must start from 0 and be consecutive.
    """
    object_containers = {}
    for i in range(n_obj):
        object_containers[i] = {
            "x_min": x0 + i * spacing - (chord * 0.625),  # slight margin
            "y_min": y0 - (chord * 0.625),
            "dx": chord * 1.25,  # slight margin
            "dy": chord * 1.25,
        }
    return object_containers


def create_object_masks(
    mesh: meshio.Mesh,
    object_containers: Dict[int, Dict[str, float]],
) -> np.ndarray:
    """
    Create a mask for a single reward box in the mesh based on the box dimensions.

    Args:
        mesh (meshio.Mesh): meshio Mesh object (a single timestep snapshot).
        object_containers (dict): dict of dict of box dimensions containing each object
            "obj_id" (int): {
                "x_min": float,
                "y_min": float,
                "dx": float,
                "dy": float}
            }, where obj_id is a unique int id for the object. Must start from 0 and be consecutive.

    Returns:
        mask: boolean array of shape (num_objects,num_points,) (axis 0: objects, axis 1: points)
        where True indicates the point is inside the box and False otherwise.
    """
    points = mesh.points
    num_points = points.shape[0]
    num_objects = len(object_containers)
    masks = np.zeros((num_objects, num_points), dtype=bool)
    for box_id, box in object_containers.items():
        # Extract box parameters
        x0 = box["x_min"]
        y0 = box["y_min"]
        dx = box["dx"]
        dy = box["dy"]
        # Create a mask for the points inside the box
        masks[box_id] = (
            (points[:, 0] >= x0)
            & (points[:, 0] <= x0 + dx)
            & (points[:, 1] >= y0)
            & (points[:, 1] <= y0 + dy)
        )
    return masks


def tri_gradients(
    coords: np.ndarray, triangles: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute triangle area and gradients of barycentric basis functions (vectorized)."""
    x = coords[:, 0]
    y = coords[:, 1]
    p0, p1, p2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]

    x0, y0 = x[p0], y[p0]
    x1, y1 = x[p1], y[p1]
    x2, y2 = x[p2], y[p2]

    detJ = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    inv_det = 1.0 / detJ
    area = 0.5 * np.abs(detJ)

    # gradients of barycentric basis functions: gx = d(phi)/dx, gy = d(phi)/dy
    gx0 = (y1 - y2) * inv_det
    gx1 = (y2 - y0) * inv_det
    gx2 = (y0 - y1) * inv_det

    gy0 = (x2 - x1) * inv_det
    gy1 = (x0 - x2) * inv_det
    gy2 = (x1 - x0) * inv_det

    grads = np.empty((triangles.shape[0], 3, 2), dtype=coords.dtype)
    grads[:, 0, 0] = gx0
    grads[:, 1, 0] = gx1
    grads[:, 2, 0] = gx2
    grads[:, 0, 1] = gy0
    grads[:, 1, 1] = gy1
    grads[:, 2, 1] = gy2

    return area, grads


def nodal_gradient(
    coords: np.ndarray, triangles: np.ndarray, field: np.ndarray
) -> np.ndarray:
    """Compute nodal gradients by area-weighted averaging of triangle gradients.
    Uses vectorized accumulation (np.bincount) instead of Python loops or many np.add.at calls.
    """
    area, grads = tri_gradients(coords, triangles)  # area (T,), grads (T,3,2)
    val_tri = field[triangles]  # (T,3)

    # gradient inside each triangle: sum_j value_j * grad(phi_j) -> (T,2)
    grad_tri = np.einsum("tjk,tj->tk", grads, val_tri)

    N = coords.shape[0]

    # flatten per-triangle contributions to nodes
    tri_idx_flat = triangles.ravel()  # length 3*T
    area_rep = np.repeat(area, 3)  # length 3*T
    grad_rep = np.repeat(grad_tri, 3, axis=0)  # (3*T,2)
    contrib = grad_rep * area_rep[:, None]  # weighted contributions

    # accumulate with bincount (fast)
    node_grad_x = np.bincount(tri_idx_flat, weights=contrib[:, 0], minlength=N)
    node_grad_y = np.bincount(tri_idx_flat, weights=contrib[:, 1], minlength=N)
    node_area = np.bincount(tri_idx_flat, weights=area_rep, minlength=N)

    node_grad = np.column_stack([node_grad_x, node_grad_y]) / (
        node_area[:, None] + 1e-16
    )
    return node_grad  # (N,2)


def triangle_velocity_gradient(
    coords: np.ndarray, triangles: np.ndarray, vel: np.ndarray
) -> np.ndarray:
    """Compute grad(u) for each triangle (constant per triangle) using a single einsum."""
    _, grads = tri_gradients(coords, triangles)  # grads shape (T,3,2)
    vel_tri = vel[triangles]  # (T,3,2)

    # grad_u[t, k, l] = sum_j grads[t, j, k] * vel_tri[t, j, l]
    # yields shape (T,2,2) where rows correspond to [du, dv] and columns to [dx,dy]
    grad_u = np.einsum("tjk,tjl->tkl", grads, vel_tri)
    return grad_u


def compute_drag_lift(
    mesh: meshio.Mesh,
    mu: float = 1.0,
    feature_names: dict = {},
    object_mask: np.ndarray | None = None,
) -> Tuple[float, float]:
    """
    Compute drag and lift forces on the object defined by nodetype==1 in a 2D triangular mesh.

    Args:
        mesh: meshio.Mesh object containing the simulation data.
        mu: dynamic viscosity of the fluid.
        feature_names: optional dict mapping standard feature names to actual point data keys in the mesh.
            Supported keys: "pressure", "velocity", "levelset", "nodetype".
        object_mask: optional boolean mask array indicating nodes belonging to box containing object (N,).
    Returns:
        Tuple of (drag force Fx, lift force Fy).
    """

    def _extract_fields(mesh, feature_names):
        """Extract required fields from the mesh with given feature names."""
        coords = mesh.points[:, :2]
        triangles = mesh.cells_dict["triangle"]

        p = mesh.point_data[feature_names.get("pressure", "Pression")].reshape(-1)
        vel_key = feature_names.get("velocity", "Vitesse")
        if isinstance(vel_key, (list, tuple)) and len(vel_key) == 2:
            vx = np.asarray(mesh.point_data[vel_key[0]]).reshape(-1)
            vy = np.asarray(mesh.point_data[vel_key[1]]).reshape(-1)
            u = np.column_stack([vx, vy])
        else:
            arr = np.asarray(mesh.point_data[vel_key])
            if arr.ndim == 1:
                raise ValueError(
                    f"Velocity field '{vel_key}' is 1D; expected shape (N,2) or provide two keys in for 'velocity'."
                )
            u = arr[:, :2]
        phi = -mesh.point_data[feature_names.get("levelset", "LevelSetObject")].reshape(
            -1
        )
        nodetype = np.asarray(
            mesh.point_data[feature_names.get("nodetype", "NodeType")].reshape(-1)
        ).astype(int)

        return coords, triangles, p, u, phi, nodetype

    def _get_boundary_cache(triangles):
        """
        Get cached boundary information for the given triangles.
        """
        # caching based on triangles content
        if not hasattr(compute_drag_lift, "_boundary_cache"):
            compute_drag_lift._boundary_cache = {}
        key = (triangles.shape, triangles.dtype, hash(triangles.tobytes()))
        cache = compute_drag_lift._boundary_cache.get(key)
        if cache is None:
            boundary_edges = compute_boundary_edges(triangles)

            T = len(triangles)
            edges_all = np.vstack(
                [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
            )
            edges_all_sorted = np.sort(edges_all, axis=1)
            tri_idx = np.repeat(np.arange(T, dtype=np.int64), 3)

            keys = (edges_all_sorted[:, 0].astype(np.int64) << 32) | edges_all_sorted[
                :, 1
            ].astype(np.int64)
            unique_keys, idx_first = np.unique(keys, return_index=True)
            tri_for_key = tri_idx[idx_first]

            cache = {
                "boundary_edges": boundary_edges,
                "unique_keys": unique_keys,
                "tri_for_key": tri_for_key,
            }
            compute_drag_lift._boundary_cache[key] = cache
        return cache

    def _filter_boundary_edges_by_nodetype(boundary_edges, nodetype):
        """Keep only boundary edges where both nodes have nodetype==1."""
        i0_tmp, i1_tmp = boundary_edges[:, 0], boundary_edges[:, 1]
        mask = (nodetype[i0_tmp] == 1) & (nodetype[i1_tmp] == 1)
        if not np.any(mask):
            return np.empty((0, 2), dtype=boundary_edges.dtype)
        return boundary_edges[mask]

    def _filter_boundary_edges_by_object_mask(boundary_edges, object_mask):
        """Keep only boundary edges where both nodes are inside the object mask."""
        i0_tmp, i1_tmp = boundary_edges[:, 0], boundary_edges[:, 1]
        mask = object_mask[i0_tmp] & object_mask[i1_tmp]
        if not np.any(mask):
            return np.empty((0, 2), dtype=boundary_edges.dtype)
        return boundary_edges[mask]

    def _edge_geometry(coords, boundary_edges, phi):
        """Compute edge length, tangent, and normal vectors for boundary edges."""
        i0, i1 = boundary_edges[:, 0], boundary_edges[:, 1]
        p0, p1 = coords[i0], coords[i1]
        edge_vec = p1 - p0
        edge_len = np.linalg.norm(edge_vec, axis=1)
        # avoid division by zero in degenerate edges
        edge_len_safe = edge_len.copy()
        edge_len_safe[edge_len_safe == 0.0] = 1.0
        tangent = edge_vec / edge_len_safe[:, None]

        # normals oriented using level-set gradient
        n_cand = np.column_stack([-tangent[:, 1], tangent[:, 0]])
        grad_phi = nodal_gradient(coords, triangles, phi)
        grad_phi_edge = 0.5 * (grad_phi[i0] + grad_phi[i1])
        sign = np.sign(np.einsum("ij,ij->i", n_cand, grad_phi_edge))
        sign[sign == 0] = 1.0
        normals = n_cand * sign[:, None]

        return i0, i1, edge_len, tangent, normals

    def _pressure_traction(p, i0, i1, normals):
        """Compute pressure traction on edges."""
        p_edge = 0.5 * (p[i0] + p[i1])
        return -p_edge[:, None] * normals

    def _viscous_traction(coords, triangles, u, boundary_edges, cache, mu):
        """Compute viscous traction on edges."""
        # map each boundary edge to one adjacent triangle
        edges_sorted = np.sort(boundary_edges, axis=1)
        bkeys = (edges_sorted[:, 0].astype(np.int64) << 32) | edges_sorted[:, 1].astype(
            np.int64
        )

        if "key_to_tri" not in cache:
            cache["key_to_tri"] = dict(
                zip(cache["unique_keys"].tolist(), cache["tri_for_key"].tolist())
            )
        key_to_tri = cache["key_to_tri"]

        try:
            tri_for_edge = np.fromiter(
                (key_to_tri[k] for k in bkeys), dtype=np.int64, count=bkeys.shape[0]
            )
        except KeyError as e:
            missing = [int(k) for k in bkeys if k not in key_to_tri]
            raise KeyError(
                f"Boundary edge key(s) not found in mesh edge mapping: {missing[:10]}..."
            ) from e

        grad_u = triangle_velocity_gradient(coords, triangles, u)  # (T,2,2)
        S = mu * (
            grad_u[tri_for_edge] + np.swapaxes(grad_u[tri_for_edge], 1, 2)
        )  # (B,2,2)
        # normals will be provided by caller when multiplying
        return S, tri_for_edge

    # main function body
    coords, triangles, p, u, phi, nodetype = _extract_fields(mesh, feature_names)

    cache = _get_boundary_cache(triangles)
    boundary_edges = cache["boundary_edges"].copy()  # safe local copy before filtering

    boundary_edges = _filter_boundary_edges_by_nodetype(boundary_edges, nodetype)
    if object_mask is not None:
        boundary_edges = _filter_boundary_edges_by_object_mask(
            boundary_edges, object_mask
        )
    if boundary_edges.shape[0] == 0:
        return 0.0, 0.0

    i0, i1, edge_len, tangent, normals = _edge_geometry(coords, boundary_edges, phi)

    traction_p = _pressure_traction(p, i0, i1, normals)

    S, tri_for_edge = _viscous_traction(coords, triangles, u, boundary_edges, cache, mu)
    # traction_v[b,i] = S[b, i, j] * normals[b, j]
    traction_v = np.einsum("bij,bj->bi", S, normals)

    traction_total = traction_p + traction_v

    # integrate over edges
    F_edges = traction_total * edge_len[:, None]
    F_total = np.sum(F_edges, axis=0)

    return float(F_total[0]), float(F_total[1])


def compute_drag_lift_multiobject(
    mesh: meshio.Mesh,
    mu: float = 1.0,
    feature_names: dict = {},
    object_masks: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Compute drag and lift forces on multiple objects defined by nodetype==1 in a 2D triangular mesh.

    Args:
        mesh: meshio.Mesh object containing the simulation data.
        mu: dynamic viscosity of the fluid.
        feature_names: optional dict mapping standard feature names to actual point data keys in the mesh.
            Supported keys: "pressure", "velocity", "levelset", "nodetype".
        object_masks: optional boolean mask array indicating nodes belonging to \
        boxes containing each object (num_objects,N).
    Returns:
        pandas DataFrame with columns ['Fx', 'Fy', 'Object'] containing drag and lift forces for each object.
    """
    forces = []
    if object_masks is None:
        fx, fy = compute_drag_lift(
            mesh=mesh, mu=mu, feature_names=feature_names, object_mask=None
        )
        forces.append({"Fx": fx, "Fy": fy, "Object": ""})
    else:
        for obj_id, mask in enumerate(object_masks):
            fx, fy = compute_drag_lift(
                mesh=mesh, mu=mu, feature_names=feature_names, object_mask=mask
            )
            forces.append({"Fx": fx, "Fy": fy, "Object": obj_id})

    return pd.DataFrame(forces)


def compute_series_forces(
    meshes: List[meshio.Mesh],
    times: List[float] = None,
    mu: float = 1e-3,
    feature_names: Dict[str, any] = None,
    object_masks: np.ndarray | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Compute drag and lift forces for each mesh in the trajectory.
    Parallelized over meshes and objects if applicable.

    Args:
        meshes: list of meshio.Mesh objects representing the simulation at different time steps.
        times: optional list of time values corresponding to each mesh.
        mu: dynamic viscosity of the fluid.
        feature_names: optional dict mapping standard feature names to actual point data keys in the mesh.
        object_masks: optional boolean mask array indicating nodes belonging to \
        boxes containing each object (num_objects,N).
        verbose: whether to show progress bar.

    Returns:
        pandas DataFrame with columns ['Time', 'Fx', 'Fy', 'Object'] containing drag and lift forces for each object.
    """
    if times is None:
        times = list(range(len(meshes)))
    if feature_names is None:
        feature_names = {}

    n_jobs = min(
        object_masks.shape[0] if object_masks is not None else 2,
        max(1, (os.cpu_count() or 1)),
    )  # reasonable default cap

    if object_masks is None:
        # Parallel compute single-object drag/lift per mesh
        results = [None] * len(meshes)
        submit_fn = partial(
            compute_drag_lift, mu=mu, feature_names=feature_names, object_mask=None
        )
        with ThreadPoolExecutor(max_workers=n_jobs) as exe:
            futures = {exe.submit(submit_fn, mesh=m): i for i, m in enumerate(meshes)}
            pbar = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Computing drag and lift",
                disable=not verbose,
            )
            for fut in pbar:
                i = futures[fut]
                results[i] = fut.result()
        # build dataframe
        drag = [r[0] for r in results]
        lift = [r[1] for r in results]
        forces_df = pd.DataFrame(
            {"Time": times, "Fx": drag, "Fy": lift, "Object": [""] * len(meshes)}
        )
    else:
        # Parallel compute multi-object per mesh, collect DataFrames and concat once
        dfs: List[pd.DataFrame] = []
        submit_fn = partial(
            compute_drag_lift_multiobject,
            mu=mu,
            feature_names=feature_names,
            object_masks=object_masks,
        )
        with ThreadPoolExecutor(max_workers=n_jobs) as exe:
            futures = {exe.submit(submit_fn, mesh=m): i for i, m in enumerate(meshes)}
            pbar = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Computing drag and lift (multi-object)",
                disable=not verbose,
            )
            for fut in pbar:
                i = futures[fut]
                time = times[i]
                df_obj = fut.result()
                # insert time column at front
                df_obj.insert(0, "Time", time)
                dfs.append(df_obj)
        if dfs:
            forces_df = pd.concat(dfs, ignore_index=True)
        else:
            forces_df = pd.DataFrame(columns=["Time", "Fx", "Fy", "Object"])

    # stable sort and return
    forces_df = forces_df.sort_values(by=["Object", "Time"], ignore_index=True)
    return forces_df


def compute_forces_from_trajectory(
    trajectory: Trajectory,
    mu: float = 1e-3,
    feature_names: dict = {},
    start_step: int = None,
    end_step: int = None,
    object_containers: Dict[int, Dict[str, float]] | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Compute drag and lift forces from a Trajectory object.
    Args:
        trajectory: Trajectory object containing the mesh time series.
        mu: dynamic viscosity of the fluid.
        feature_names: optional dict mapping standard feature names to actual point data keys in the mesh.
        start_step: optional start index for time series slicing.
        end_step: optional end index for time series slicing.
        object_containers: optional dictionary mapping object IDs to their container box dimensions.
        verbose: whether to show progress bar.
    Returns:
        DataFrame containing [Time, Fx, Fy, Object] columns.
    """
    trajectory = trajectory.slice(start=start_step, end=end_step)
    # create object masks if containers provided
    object_masks = None
    if object_containers is not None:
        object_masks = create_object_masks(trajectory.meshes[0], object_containers)
    return compute_series_forces(
        meshes=trajectory.meshes,
        times=trajectory.times,
        mu=mu,
        feature_names=feature_names,
        object_masks=object_masks,
        verbose=verbose,
    )


def compute_forces_from_xdmf(
    case_xdmf_path: str,
    mu: float = 1e-3,
    feature_names: dict = {},
    start_step: int = None,
    end_step: int = None,
    object_containers: Dict[int, Dict[str, float]] | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Compute drag and lift forces from an XDMF file.
    Args:
        xdmf_path: path to the XDMF file.
        mu: dynamic viscosity of the fluid.
        feature_names: optional dict mapping standard feature names to actual point data keys in the mesh.
        start_step: optional start index for time series slicing.
        end_step: optional end index for time series slicing.
        object_containers: optional dictionary mapping object IDs to their container box dimensions.
        verbose: whether to show progress bar.
    Returns:
        pandas DataFrame with columns ['Time', 'Fx', 'Fy', 'Object'] containing drag and lift forces for each time step.
    """
    # Load the mesh from the XDMF file
    meshes, times = xdmf_to_meshes(xdmf_file_path=case_xdmf_path, verbose=verbose)
    # create object masks if containers provided
    object_masks = None
    if object_containers is not None:
        object_masks = create_object_masks(meshes[0], object_containers)
    return compute_series_forces(
        meshes=meshes[start_step:end_step],
        times=times[start_step:end_step],
        mu=mu,
        feature_names=feature_names,
        object_masks=object_masks,
        verbose=verbose,
    )


def plot_save_compare(
    forces_df: pd.DataFrame,
    out_base,
    out_dir,
    mu=1e-3,
    gt_path: str = None,
    show: bool = True,
    object_id: int | str = "",
    save_data: bool = False,
):
    """
    Plot results, optionally compare to ground truth CSV, save figure and a CSV of forces.
    Args:
        forces_df: DataFrame containing forces data
        mu: dynamic viscosity
        out_base: base name used to generate file names (without extension)
        out_dir: directory Path or string to save outputs
        gt_path: optional path to ground truth CSV (if None, no GT compared)
        show: whether to display the plot interactively
        object_id: object identifier to filter forces for plotting (int or str if all objects)
        save_data: whether to save the filtered/used part of forces DataFrame as CSV
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    import os

    forces_df = forces_df[forces_df["Object"] == object_id]
    times = forces_df["Time"].values
    drag = forces_df["Fx"].values
    lift = forces_df["Fy"].values

    out_dir = Path(out_dir)
    fig_path = out_dir / f"{out_base}_draglift{object_id}.png"
    csv_path = out_dir / f"{out_base}_forces.csv"

    gt_times = gt_drag = gt_lift = None
    if gt_path is not None and os.path.exists(gt_path):
        try:
            df = pd.read_csv(gt_path)
            gt_df = df[df["Object"] == object_id]
            if {"Temps", "Fx", "Fy"}.issubset(gt_df.columns):
                gt_times = gt_df["Temps"].values
                gt_drag = gt_df["Fx"].values
                gt_lift = gt_df["Fy"].values
                print(f"Loaded ground truth forces from {gt_path}")
            else:
                print(
                    f"Ground truth CSV found but required columns missing in {gt_path}"
                )
        except Exception as e:
            print(f"Failed to read ground truth CSV {gt_path}: {e}")

    plt.figure()
    plt.plot(
        times,
        drag,
        label="Fx",
        color="darkblue",
        alpha=0.7 if gt_times is None else 1,
    )
    plt.plot(
        times,
        lift,
        label="Fy",
        color="darkred",
        alpha=0.7 if gt_times is None else 1,
    )
    if gt_times is not None:
        plt.plot(
            gt_times,
            gt_drag,
            label="Fx (truth)",
            color="blue",
            linestyle="--",
            alpha=0.7,
        )
        plt.plot(
            gt_times,
            gt_lift,
            label="Fy (truth)",
            color="red",
            linestyle="--",
            alpha=0.7,
        )

    plt.title(f"Drag and Lift (mu={mu}) for {out_base}")
    # better y-limits
    try:
        candidates = [drag.min(), lift.min(), drag.max(), lift.max()]
        if gt_times is not None and gt_lift is not None and len(gt_lift) > 0:
            candidates += [
                gt_drag[100:].min(),
                gt_lift[100:].min(),
                gt_drag[100:].max(),
                gt_lift[100:].max(),
            ]
        ymin, ymax = min(candidates), max(candidates)
        if ymin == ymax:
            ymin -= 1.0
            ymax += 1.0
        plt.ylim((ymin, ymax))
    except Exception:
        pass

    plt.legend(ncols=2)
    plt.xlabel("Time")
    plt.ylabel("Force")
    plt.savefig(fig_path, dpi=200)
    print(f"Saved figure to {fig_path}")
    if show:
        plt.show()
    else:
        plt.close()

    # save forces as CSV (time, drag, lift)
    if save_data:
        forces_df.to_csv(csv_path, index=False)
        print(f"Saved forces to {csv_path}")


def main():
    import os
    from pathlib import Path

    args = _parser()
    xdmf_path = args.xdmf
    mu = args.mu
    verbose = args.verbose
    meshes, times = xdmf_to_meshes(xdmf_file_path=xdmf_path, verbose=verbose)

    out_dir = args.out_dir or Path(xdmf_path).parent
    out_dir = Path(out_dir)
    base = Path(xdmf_path).stem

    if "graph" in base:
        feature_names = {
            "velocity": ["x0", "x1"],
            "pressure": "x2",
            "levelset": "x3",
            "nodetype": "x6",
        }
    else:
        feature_names = {}

    object_containers = build_object_containers(
        n_obj=3, x0=0, y0=1.5, chord=2.0, spacing=4.0
    )
    object_masks = create_object_masks(meshes[0], object_containers)

    forces_df = compute_series_forces(
        meshes=meshes,
        times=times,
        mu=mu,
        feature_names=feature_names,
        verbose=verbose,
        object_masks=object_masks,
    )
    forces_df.to_csv(out_dir / f"{base}_forces.csv", index=False)

    # TODO: parse ground truth path?
    input_path = xdmf_path.replace("graph", "panels")
    gt_path = os.path.splitext(input_path)[0] + "_data.csv"
    if not Path(gt_path).exists():
        gt_path = None

    plot_save_compare(
        forces_df,
        base,
        out_dir,
        mu=mu,
        gt_path=gt_path,
        show=args.show,
        object_id=args.object_id,
        save_data=False,
    )


def _parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute drag and lift from an XDMF series.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("xdmf", help="Input XDMF timeseries file")
    parser.add_argument("--mu", type=float, default=1e-3, help="Dynamic viscosity.")
    parser.add_argument(
        "--no_show",
        dest="show",
        action="store_false",
        help="Do not display the plot",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Directory to write outputs.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose progress bars")
    parser.add_argument(
        "--object_id", type=int, default=0, help="Object ID to analyze."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
