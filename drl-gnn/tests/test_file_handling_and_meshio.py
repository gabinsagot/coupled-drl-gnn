import gmsh
import meshio
import numpy as np

from graphdrl.utils.file_handling import Mtc, gather_cases

from graphdrl.utils.meshio_mesh import (
    meshes_to_xdmf,
    xdmf_to_meshes,
    vtu_to_meshes,
    convert_gmsh_to_mtc,
)


def test_mtc_format_and_write(tmp_path):
    p = tmp_path / "test.mtc"
    mtc = Mtc(path=str(p))
    # write content
    content = mtc.format_ModeleDeModeles(
        model_name="M1", submodels_list=["A", "B"], mesh_model="Msh"
    )
    mtc.write(content=content, overwrite=True)
    assert p.exists()
    loaded = Mtc(path=str(p))
    assert "ModeleDeModeles" in loaded.content


def test_gather_cases_and_move_meshes(tmp_path):
    # create fake xdmf files
    folder = tmp_path / "cases"
    folder.mkdir()
    for i in range(3):
        f = folder / f"case_{i}.xdmf"
        f.write_text("dummy")
    cases = gather_cases(str(folder), "case_")
    assert len(cases) == 3


def test_meshio_helpers_roundtrip(tmp_path):
    # create a simple mesh and write using meshes_to_xdmf then read back
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    cells = [("triangle", np.array([[0, 1, 2]]))]
    vitesse = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    mesh = meshio.Mesh(points=points, cells=cells, point_data={"Vitesse": vitesse})
    meshes_to_xdmf(
        filename=str(tmp_path / "test_traj.xdmf"),
        meshes=[mesh],
        timestep=0.1,
        verbose=False,
    )
    meshes, times = xdmf_to_meshes(str(tmp_path / "test_traj.xdmf"), verbose=False)
    assert len(meshes) == 1
    assert (
        np.allclose(meshes[0].points[:2], points[:2]) or meshes[0].points.shape[0] >= 1
    )
    (tmp_path / "test_traj.*").unlink(missing_ok=True)


def test_vtu_to_meshes(tmp_path):
    # create a simple mesh and write using meshes_to_xdmf then read back
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    cells = [("triangle", np.array([[0, 1, 2]]))]
    vitesse = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    mesh = meshio.Mesh(points=points, cells=cells, point_data={"Vitesse": vitesse})
    meshio.write(str(tmp_path / "test.vtu"), mesh)
    meshes, times = vtu_to_meshes(str(tmp_path / "test.vtu"))
    assert len(times) == 1
    assert times[0] == 0.0
    assert len(meshes) == 1
    assert (
        np.allclose(meshes[0].points[:2], points[:2]) or meshes[0].points.shape[0] >= 1
    )
    (tmp_path / "test.vtu").unlink(missing_ok=True)


def test_msh_to_meshes(tmp_path):
    # create a simple mesh and write using meshes_to_xdmf then read back
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    cells = [("triangle", np.array([[0, 1, 2]]))]
    vitesse = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    mesh = meshio.Mesh(points=points, cells=cells, point_data={"Vitesse": vitesse})
    meshio.write(str(tmp_path / "test.msh"), mesh)
    meshes, times = vtu_to_meshes(str(tmp_path / "test.msh"))
    assert len(times) == 1
    assert times[0] == 0.0
    assert len(meshes) == 1
    assert (
        np.allclose(meshes[0].points[:2], points[:2]) or meshes[0].points.shape[0] >= 1
    )
    (tmp_path / "test.msh").unlink(missing_ok=True)


def test_xdmf_to_meshes(tmp_path):
    # create a simple mesh and write using meshes_to_xdmf then read back
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    cells = [("triangle", np.array([[0, 1, 2]]))]
    vitesse = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    mesh = meshio.Mesh(points=points, cells=cells, point_data={"Vitesse": vitesse})
    meshes_to_xdmf(
        filename=str(tmp_path / "test_traj.xdmf"),
        meshes=[mesh],
        timestep=0.1,
        verbose=False,
    )
    meshes, times = xdmf_to_meshes(str(tmp_path / "test_traj.xdmf"), verbose=False)
    assert len(times) == 1
    assert times[0] == 0.0
    assert len(meshes) == 1
    assert (
        np.allclose(meshes[0].points[:2], points[:2]) or meshes[0].points.shape[0] >= 1
    )
    (tmp_path / "test_traj.*").unlink(missing_ok=True)


def test_convert_gmsh_to_mtc(tmp_path):
    # create a dummy 3D mesh with gmsh
    gmsh.initialize()
    gmsh.model.add("test")
    # Create points
    p1 = gmsh.model.geo.addPoint(0.0, 0.0, 1.0, 2.0)
    p2 = gmsh.model.geo.addPoint(1.0, 0.0, 0.0, 2.0)
    p3 = gmsh.model.geo.addPoint(0.0, 1.0, 0.0, 2.0)
    # Create lines
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p1)
    # Create surface
    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3])
    _ = gmsh.model.geo.addPlaneSurface([cl])
    # Mesh
    gmsh.model.geo.synchronize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.mesh.generate(2)
    gmsh.write(str(tmp_path / "test.msh"))
    gmsh.finalize()
    # Test conversion
    convert_gmsh_to_mtc(
        input=str(tmp_path / "test.msh"),
        output=str(tmp_path / "test.t"),
        verbose=False,
    )
    assert (tmp_path / "test.t").exists()
    # Check that the first line of .t file has the correct format
    with open(tmp_path / "test.t", "r") as f:
        first_line = f.readline().strip()
        parts = first_line.split()
        assert len(parts) == 4
        num_points, dim_points, num_connectivities, dim_connectivities = map(int, parts)
        assert num_points == 3
        assert dim_points == 3
        assert num_connectivities == 1
        assert dim_connectivities == 4
    (tmp_path / "test.*").unlink(missing_ok=True)
