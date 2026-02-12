import os
import sys

import meshio
import numpy as np


# Ensure project src is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def make_simple_mesh():
    """Create a tiny 2D meshio.Mesh with a velocity field for tests."""
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    # single triangle
    cells = [("triangle", np.array([[0, 1, 2]]))]
    # point_data: velocity vector Vitesse
    vitesse = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    point_data = {"Vitesse": vitesse}
    mesh = meshio.Mesh(points=points, cells=cells, point_data=point_data)
    return mesh


def mesh_file(tmp_path, mesh=None):
    mesh = mesh or make_simple_mesh()
    p = tmp_path / "simple.vtu"
    meshio.write(str(p), mesh)
    return str(p)
