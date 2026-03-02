import os
from typing import Dict, List, Union

import gmsh
import numpy as np

from utils import apply_Center2, get_predefined_airfoil, morphed_airfoil


class Geometry:
    def __init__(self, dim: int, path: str = "./", verbose: bool = False):
        """Initialize the Geometry class with given parameters.

        : param dim: (int) Dimension of the geometry.
        : param path: (str) Path to the directory where the geometry will be saved
        and mesh generated. Should be a cfd directory, with BLM subdirectory.
        : param verbose: (bool) Whether to print messages of mesh generation info.
        """
        self.dim = dim
        self.cfd_path = os.path.abspath(path)
        self.verbose = verbose
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 1)  # print only errors

    def set_mesh_size(self, min_mesh_size: float = 0.1, max_mesh_size: float = 1):
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_mesh_size)

    def set_meshing_options(
        self,
        mesh_size_points: int = 0,
        mesh_size_curvature: int = 100,
        extend_from_boundary: int = 1,
    ):
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", mesh_size_points)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", mesh_size_curvature)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", extend_from_boundary)
        gmsh.option.setNumber("Mesh.Algorithm", 5)  # Delaunay 5, Frontal-Delaunay 6

    def finalize(self):
        gmsh.finalize()

    def create_rectangle(
        self,
        rect_dict: dict,
        model_name: str = "Rectangle",
        force_model: str = "",
        save_mesh: bool = False,
        dim_mesh: int = 2,
        mesh_size: float = 0.1,
    ) -> dict:
        """
        Create a rectangle (2D) or box (3D) object in GMSH OCC module from rect_dict parameters.

        Args:
            rect_dict (dict): Dictionary containing the rectangle parameters. Keys include:
                - chord (float): Chord length.
                - thickness (float): Thickness.
                - span (float): Span.
                - angle (float): Angle of attack.
                - x (float): Shift in x.
                - y (float): Shift in y.
                - z (float): Shift in z.
            model_name (str): Name of the model.
            force_model (str): Force model name.
            save_mesh (bool): Whether to save the mesh.
            dim_mesh (int): Dimension of the mesh.
            mesh_size (float): Size of the mesh.

        Returns:
            dict: Dictionary containing model name, entities, and entity names.
        """
        # Rectangle params
        chord = rect_dict["chord"]
        thickness = rect_dict["thickness"]
        span = rect_dict["span"]
        # center of mass centered in 0
        x_0 = -chord * 0.5
        y_0 = -thickness * 0.5
        z_0 = -span * 0.5
        # Create rectangle
        if force_model != "":
            gmsh.model.setCurrent(name=force_model)
        else:
            gmsh.model.add(model_name)
        if dim_mesh == 2:
            rectangle = gmsh.model.occ.addRectangle(
                x=x_0,
                y=y_0,
                z=z_0,
                dx=chord,
                dy=thickness,
            )
        elif dim_mesh == 3:
            rectangle = gmsh.model.occ.addBox(
                x=x_0,
                y=y_0,
                z=z_0,
                dx=chord,
                dy=thickness,
                dz=span,
            )
        else:
            raise ValueError("dim_mesh must be 2 or 3")
        gmsh.model.occ.synchronize()
        # rotate
        tilt = np.deg2rad(rect_dict["angle"])
        origin_rot = [0, 0, 0]
        ax_rot = [0, 0, 1]
        gmsh.model.occ.rotate(
            dimTags=[(dim_mesh, rectangle)],
            x=origin_rot[0],
            y=origin_rot[1],
            z=origin_rot[2],
            ax=ax_rot[0],
            ay=ax_rot[1],
            az=ax_rot[2],
            angle=tilt,
        )
        gmsh.model.occ.synchronize()
        # translate
        gmsh.model.occ.translate(
            dimTags=[(dim_mesh, rectangle)],
            dx=rect_dict["x"],
            dy=rect_dict["y"],
            dz=rect_dict["z"],
        )
        gmsh.model.occ.synchronize()
        # create entity name
        gmsh.model.setEntityName(dim=dim_mesh, tag=rectangle, name="rectangle")
        # save
        if save_mesh:
            # create boundary layer
            dist_field = 1
            gmsh.model.mesh.field.add(fieldType="Distance", tag=dist_field)
            gmsh.model.mesh.field.setNumbers(
                tag=dist_field,
                option="CurvesList",
                values=[gmsh.model.getBoundary([(2, rectangle)], oriented=False)[0][1]],
            )
            gmsh.model.mesh.field.setNumber(dist_field, "Sampling", 200)
            # create threshold field
            thresh_field = 2
            gmsh.model.mesh.field.add(fieldType="Threshold", tag=thresh_field)
            gmsh.model.mesh.field.setNumber(thresh_field, "InField", dist_field)
            gmsh.model.mesh.field.setNumber(
                thresh_field, "SizeMin", thickness / 4
            )  # fine at boundary
            gmsh.model.mesh.field.setNumber(
                thresh_field, "SizeMax", thickness / 2
            )  # coarse inside
            gmsh.model.mesh.field.setNumber(
                thresh_field, "DistMin", thickness / 3
            )  # transition zone
            gmsh.model.mesh.field.setNumber(thresh_field, "DistMax", 2 * thickness)
            # apply mesh size field
            gmsh.model.mesh.field.setAsBackgroundMesh(thresh_field)
            # save
            if self.verbose:
                print("saving %s mesh..." % model_name)
            gmsh.write(os.path.join(self.cfd_path, "%s.geo_unrolled" % model_name))
            with open(
                os.path.join(self.cfd_path, "%s.geo_unrolled" % model_name), "r+"
            ) as file:
                content = file.read()
                file.seek(0, 0)
                file.write('SetFactory("OpenCASCADE");\n' + content)
            try:
                gmsh.model.mesh.generate(dim_mesh)
                gmsh.write(os.path.join(self.cfd_path, "%s.msh" % model_name))
                gmsh.write(os.path.join(self.cfd_path, "%s.vtk" % model_name))
            except Exception as e:
                print(f"Error generating rectangle mesh: {e}")
                raise
        # dict of entities
        entity_dict = {
            "model": model_name,
            "volume": rectangle if dim_mesh == 3 else None,
            "surface": rectangle if dim_mesh == 2 else None,
            "entities": gmsh.model.getEntities(dim=dim_mesh),
            "entity_names": [
                gmsh.model.getEntityName(dim=entity[0], tag=entity[1])
                for entity in gmsh.model.getEntities(dim=dim_mesh)
            ],
        }
        return entity_dict

    def create_disk(
        self,
        disk_dict: dict,
        model_name: str = "Disk",
        force_model: str = "",
        save_mesh: bool = False,
        dim_mesh: int = 2,
        mesh_size: float = 0.1,
    ) -> dict:
        """
        Create a disk with given parameters.

        Args:
            disk_dict (dict): Dictionary of disk properties.
            model_name (str): Name of the model.
            force_model (str): Force model name.
            save_mesh (bool): Whether to save the mesh.
            dim_mesh (int): Dimension of the mesh.
            mesh_size (float): Size of the mesh.

        Returns:
            dict: Dictionary of disk entities.
        """
        if dim_mesh != 2:
            raise ValueError(
                f"Dimension {dim_mesh} not supported for disk object, only dim=2."
            )
        radius = disk_dict["radius"]
        x_0, y_0, z_0 = 0, 0, 0
        if force_model != "":
            gmsh.model.setCurrent(name=force_model)
        else:
            gmsh.model.add(model_name)
        model = gmsh.model.getCurrent()
        disk = gmsh.model.occ.addDisk(xc=x_0, yc=y_0, zc=z_0, rx=radius, ry=radius)
        gmsh.model.occ.synchronize()
        gmsh.model.occ.translate(
            dimTags=[(2, disk)], dx=disk_dict["x"], dy=disk_dict["y"], dz=disk_dict["z"]
        )
        gmsh.model.occ.synchronize()
        gmsh.model.setEntityName(dim=2, tag=disk, name="disk")
        if save_mesh:
            # create boundary layer
            dist_field = 1
            gmsh.model.mesh.field.add(fieldType="Distance", tag=dist_field)
            gmsh.model.mesh.field.setNumbers(
                tag=dist_field,
                option="CurvesList",
                values=[gmsh.model.getBoundary([(2, disk)], oriented=False)[0][1]],
            )
            gmsh.model.mesh.field.setNumber(dist_field, "Sampling", 200)
            # create threshold field
            thresh_field = 2
            gmsh.model.mesh.field.add(fieldType="Threshold", tag=thresh_field)
            gmsh.model.mesh.field.setNumber(thresh_field, "InField", dist_field)
            gmsh.model.mesh.field.setNumber(
                thresh_field, "SizeMin", radius / 50
            )  # fine at boundary
            gmsh.model.mesh.field.setNumber(
                thresh_field, "SizeMax", radius / 2
            )  # coarse inside
            gmsh.model.mesh.field.setNumber(
                thresh_field, "DistMin", radius / 20
            )  # transition zone
            gmsh.model.mesh.field.setNumber(thresh_field, "DistMax", radius / 5)
            # apply mesh size field
            gmsh.model.mesh.field.setAsBackgroundMesh(thresh_field)
            # save
            if self.verbose:
                print("saving %s mesh..." % model_name)
            gmsh.write(os.path.join(self.cfd_path, "%s.geo_unrolled" % model_name))
            with open(
                os.path.join(self.cfd_path, "%s.geo_unrolled" % model_name), "r+"
            ) as file:
                content = file.read()
                file.seek(0, 0)
                file.write('SetFactory("OpenCASCADE");\n' + content)
            try:
                gmsh.model.mesh.generate(dim_mesh)
                gmsh.write(os.path.join(self.cfd_path, "%s.msh" % model_name))
                gmsh.write(os.path.join(self.cfd_path, "%s.vtk" % model_name))
            except Exception as e:
                print(f"Error generating disk mesh: {e}")
                raise
        # dict of entities
        entity_dict = {
            "model": model,
            "entities": gmsh.model.getEntities(dim=2),
            "entity_names": [
                gmsh.model.getEntityName(dim=entity[0], tag=entity[1])
                for entity in gmsh.model.getEntities(dim=2)
            ],
        }
        return entity_dict

    def create_triangle(
        self,
        tri_dict: dict,
        model_name: str = "Triangle",
        force_model: str = "",
        save_mesh: bool = False,
        dim_mesh: int = 2,
        mesh_size: float = 0.1,
    ) -> dict:
        """
        Create a triangle object in GMSH OCC module from tri_dict parameters.

        Args:
            tri_dict (dict): Dictionary containing the triangle parameters. Keys include:
                - base (float): Base length.
                - height (float): Height.
                - angle (float): Angle.
                - x (float): Shift in x.
                - y (float): Shift in y.
                - z (float): Shift in z.
            model_name (str): Name of the model.
            force_model (str): Force model name.
            save_mesh (bool): Whether to save the mesh.
            dim_mesh (int): Dimension of the mesh.
            mesh_size (float): Size of the mesh.

        Returns:
            dict: Dictionary containing model name, entities, and entity names.
        """
        if dim_mesh != 2:
            raise ValueError(
                f"Dimension {dim_mesh} not supported for triangle object, only dim=2."
            )
        # triangle params
        base = tri_dict["base"]
        height = tri_dict["height"]
        # center of mass centered in 0
        x_0 = -base * 0.5
        y_0 = -height * 0.5
        z_0 = 0.0
        # create triangle
        if force_model != "":
            gmsh.model.setCurrent(name=force_model)
        else:
            gmsh.model.add(model_name)
        # vertices
        p1 = gmsh.model.occ.addPoint(x=x_0, y=y_0, z=z_0)
        p2 = gmsh.model.occ.addPoint(x=x_0 + base, y=y_0, z=z_0)
        p3 = gmsh.model.occ.addPoint(x=x_0 + base * 0.5, y=y_0 + height, z=z_0)
        # edges
        l1 = gmsh.model.occ.addLine(startTag=p1, endTag=p2)
        l2 = gmsh.model.occ.addLine(startTag=p2, endTag=p3)
        l3 = gmsh.model.occ.addLine(startTag=p3, endTag=p1)
        # surface
        curve_loop = gmsh.model.occ.addCurveLoop(curveTags=[l1, l2, l3])
        triangle = gmsh.model.occ.addPlaneSurface(wireTags=[curve_loop])
        gmsh.model.occ.synchronize()
        # rotate
        tilt = np.deg2rad(tri_dict["angle"])
        origin_rot = [0, 0, 0]
        ax_rot = [0, 0, 1]
        gmsh.model.occ.rotate(
            dimTags=[(2, triangle)],
            x=origin_rot[0],
            y=origin_rot[1],
            z=origin_rot[2],
            ax=ax_rot[0],
            ay=ax_rot[1],
            az=ax_rot[2],
            angle=tilt,
        )
        gmsh.model.occ.synchronize()
        # translate
        gmsh.model.occ.translate(
            dimTags=[(2, triangle)],
            dx=tri_dict["x"],
            dy=tri_dict["y"],
            dz=tri_dict["z"],
        )
        gmsh.model.occ.synchronize()
        # create entity
        gmsh.model.setEntityName(dim=2, tag=triangle, name=f"{model_name}")
        # save
        if save_mesh:
            if self.verbose:
                print("saving %s mesh..." % model_name)
            gmsh.write(os.path.join(self.cfd_path, "%s.geo_unrolled" % model_name))
            with open(
                os.path.join(self.cfd_path, "%s.geo_unrolled" % model_name), "r+"
            ) as file:
                content = file.read()
                file.seek(0, 0)
                file.write('SetFactory("OpenCASCADE");\n' + content)
            try:
                gmsh.model.mesh.generate(dim_mesh)
                gmsh.write(os.path.join(self.cfd_path, "%s.msh" % model_name))
                gmsh.write(os.path.join(self.cfd_path, "%s.vtk" % model_name))
            except Exception as e:
                print(f"Error generating triangle mesh: {e}")
                raise
        # dict of entities
        entity_dict = {
            "model": model_name,
            "entities": gmsh.model.getEntities(dim=2),
            "entity_names": [
                gmsh.model.getEntityName(dim=entity[0], tag=entity[1])
                for entity in gmsh.model.getEntities(dim=2)
            ],
        }
        return entity_dict

    def create_bluff(
        self,
        bluff_dict: dict,
        model_name: str = "Bluff",
        force_model: str = "",
        save_mesh: bool = False,
        dim_mesh: int = 2,
        mesh_size: float = 0.1,
    ) -> dict:
        """
        Create a bluff object in GMSH OCC module from bluff_dict parameters.

        Args:
            bluff_dict (dict): Dictionary containing the bluff parameters. Keys include:
                - d1 (float): distance of first point
                - d2 (float): distance of second point
                - d3 (float): distance of third point
                - d4 (float): distance of fourth point
                - alpha (float): angle of the bluff points orientation
                - angle (float): Angle of attack.
                - x (float): Shift in x.
                - y (float): Shift in y.
                - z (float): Shift in z.
            model_name (str): Name of the model.
            force_model (str): Force model name.
            save_mesh (bool): Whether to save the mesh.
            dim_mesh (int): Dimension of the mesh.
            mesh_size (float): Size of the mesh.

        Returns:
            dict: Dictionary containing model name, entities, and entity names.
        """
        if dim_mesh != 2:
            raise ValueError(
                f"Dimension {dim_mesh} not supported for bluff object, only dim=2."
            )
        # Bluff params
        d1, d2, d3, d4 = (
            bluff_dict["d1"],
            bluff_dict["d2"],
            bluff_dict["d3"],
            bluff_dict["d4"],
        )
        alpha = np.deg2rad(bluff_dict["alpha"])
        p1 = [-d1 * np.cos(alpha), d2 * np.sin(alpha), 0]
        p2 = [-d2 * np.cos(alpha), -d2 * np.sin(alpha), 0]
        p3 = [d3 * np.cos(alpha), -d3 * np.sin(alpha), 0]
        p4 = [d4 * np.cos(alpha), d4 * np.sin(alpha), 0]
        cloud_points = [p1, p2, p3, p4]

        # Create bluff
        if force_model != "":
            gmsh.model.setCurrent(name=force_model)
        else:
            gmsh.model.add(model_name)

        # define points
        point_ids = []
        for coord in cloud_points:
            point = gmsh.model.occ.addPoint(x=coord[0], y=coord[1], z=coord[2])
            point_ids.append(point)
        gmsh.model.occ.synchronize()

        # define splines
        upper_spline = gmsh.model.occ.addBSpline(
            pointTags=[point_ids[1], point_ids[2], point_ids[3]]
        )
        lower_spline = gmsh.model.occ.addBSpline(
            pointTags=[point_ids[3], point_ids[0], point_ids[1]]
        )
        curve_loop = gmsh.model.occ.addCurveLoop(curveTags=[upper_spline, lower_spline])
        bluff = gmsh.model.occ.addPlaneSurface(wireTags=[curve_loop])
        gmsh.model.occ.synchronize()

        # rotate
        tilt = np.deg2rad(bluff_dict["angle"])
        origin_rot = [0, 0, 0]
        ax_rot = [0, 0, 1]
        gmsh.model.occ.rotate(
            dimTags=[(2, bluff)],
            x=origin_rot[0],
            y=origin_rot[1],
            z=origin_rot[2],
            ax=ax_rot[0],
            ay=ax_rot[1],
            az=ax_rot[2],
            angle=tilt,
        )
        gmsh.model.occ.synchronize()

        # translate
        gmsh.model.occ.translate(
            dimTags=[(2, bluff)],
            dx=bluff_dict["x"],
            dy=bluff_dict["y"],
            dz=bluff_dict["z"],
        )
        gmsh.model.occ.synchronize()

        # create entity name
        gmsh.model.setEntityName(dim=2, tag=bluff, name="bluff")
        # save
        if save_mesh:
            # create boundary layer
            dist_field = 1
            gmsh.model.mesh.field.add(fieldType="Distance", tag=dist_field)
            gmsh.model.mesh.field.setNumbers(
                tag=dist_field,
                option="CurvesList",
                values=[gmsh.model.getBoundary([(2, bluff)], oriented=False)[0][1]],
            )
            gmsh.model.mesh.field.setNumber(dist_field, "Sampling", 200)
            # create threshold field
            thresh_field = 2
            min_dist = min([d1, d2, d3, d4])
            gmsh.model.mesh.field.add(fieldType="Threshold", tag=thresh_field)
            gmsh.model.mesh.field.setNumber(thresh_field, "InField", dist_field)
            gmsh.model.mesh.field.setNumber(
                thresh_field, "SizeMin", min_dist / 20
            )  # fine at boundary
            gmsh.model.mesh.field.setNumber(
                thresh_field, "SizeMax", min_dist / 2
            )  # coarse inside
            gmsh.model.mesh.field.setNumber(
                thresh_field, "DistMin", min_dist / 20
            )  # transition zone
            gmsh.model.mesh.field.setNumber(thresh_field, "DistMax", min_dist / 5)
            # apply mesh size field
            gmsh.model.mesh.field.setAsBackgroundMesh(thresh_field)
            if self.verbose:
                print("saving %s mesh..." % model_name)
            gmsh.write(os.path.join(self.cfd_path, "%s.geo_unrolled" % model_name))
            with open(
                os.path.join(self.cfd_path, "%s.geo_unrolled" % model_name), "r+"
            ) as file:
                content = file.read()
                file.seek(0, 0)
                file.write('SetFactory("OpenCASCADE");\n' + content)
            try:
                gmsh.model.mesh.generate(dim_mesh)
                gmsh.write(os.path.join(self.cfd_path, "%s.msh" % model_name))
                gmsh.write(os.path.join(self.cfd_path, "%s.vtk" % model_name))
            except Exception as e:
                print(f"Error generating bluff mesh: {e}")
                raise
        # dict of entities
        entity_dict = {
            "model": model_name,
            "entities": gmsh.model.getEntities(dim=2),
            "entity_names": [
                gmsh.model.getEntityName(dim=entity[0], tag=entity[1])
                for entity in gmsh.model.getEntities(dim=2)
            ],
        }
        return entity_dict

    def create_airfoil(
        self,
        airfoil_dict: Dict,
        model_name: str = "Airfoil",
        force_model: str = "",
        save_mesh: bool = False,
        dim_mesh: int = 2,
    ) -> Dict:
        """Create an airfoil geometry using spline curves and gmsh occ kernel.

        Args:
            airfoil_dict (dict): Dictionary containing airfoil parameters:
                - "points": Array of (x, y) coordinates
                - "angle": Angle of attack in degrees
                - "x", "y", "z": Position offsets
            model_name (str): Name for the airfoil model.
            force_model (str): Existing model name to add to. If empty, creates new model.
            save_mesh (bool): Whether to generate and save the mesh.
            dim_mesh (int): Dimension for mesh generation (2 or 3).

        Returns:
            dict: Dictionary containing airfoil entities and metadata.
        """
        if force_model != "":
            gmsh.model.setCurrent(name=force_model)
        else:
            gmsh.model.add(model_name)

        current_model = gmsh.model.getCurrent()

        points = np.array(airfoil_dict["points"], dtype=float)
        npoints = points.shape[0]

        # ensure closed loops
        if not np.allclose(points[0], points[-1], atol=1e-10):
            points = np.vstack([points, points[0:1]])  # add first point

        # define points
        point_ids = []
        for pt in points:
            pt_id = gmsh.model.occ.addPoint(pt[0], pt[1], 0)
            point_ids.append(pt_id)

        gmsh.model.occ.synchronize()

        # leading edge
        x_coords = points[:, 0]
        leading_edge_idx = np.argmin(x_coords)

        # upper and lower B-splines
        upper_spline = gmsh.model.occ.addBSpline(
            pointTags=point_ids[0 : leading_edge_idx + 1]
        )
        lower_spline = gmsh.model.occ.addBSpline(pointTags=point_ids[leading_edge_idx:])

        # surface
        curve_loop_id = gmsh.model.occ.addCurveLoop([upper_spline, lower_spline])
        airfoil_surface = gmsh.model.occ.addPlaneSurface([curve_loop_id])

        gmsh.model.occ.synchronize()

        # impose mesh size at points and curves before deleting them
        thickness = float(np.max(points[:, 1]) - np.min(points[:, 1]))
        chord = float(np.max(points[:, 0]) - np.min(points[:, 0]))
        uniform_size = max(1e-3, max(thickness, chord) / (npoints * 2))
        targets = [(0, pid) for pid in point_ids]
        targets.extend([(1, upper_spline), (1, lower_spline)])
        gmsh.model.mesh.setSize(targets, uniform_size)

        # delete points and curves to keep only surface in final vtk
        for pid in point_ids:
            gmsh.model.occ.remove([(0, pid)])
        gmsh.model.occ.remove([(1, upper_spline)])
        gmsh.model.occ.remove([(1, lower_spline)])
        gmsh.model.occ.synchronize()

        # rotate (angle of attack)
        angle_rad = np.deg2rad(airfoil_dict["angle"])
        if angle_rad != 0:
            gmsh.model.occ.rotate(
                dimTags=[(2, airfoil_surface)],
                x=0,
                y=0,
                z=0,
                ax=0,
                ay=0,
                az=1,
                angle=angle_rad,
            )
            gmsh.model.occ.synchronize()

        # translate
        gmsh.model.occ.translate(
            dimTags=[(2, airfoil_surface)],
            dx=airfoil_dict["x"],
            dy=airfoil_dict["y"],
            dz=airfoil_dict["z"],
        )
        gmsh.model.occ.synchronize()

        # set entity name
        gmsh.model.setEntityName(dim=2, tag=airfoil_surface, name=model_name)
        gmsh.model.occ.synchronize()

        # generate and save mesh
        if save_mesh:
            # create box field for uniform mesh size inside airfoil
            box_field = gmsh.model.mesh.field.add(fieldType="Box")
            gmsh.model.mesh.field.setNumber(box_field, "VIn", uniform_size)
            gmsh.model.mesh.field.setNumber(box_field, "VOut", uniform_size * 10)
            gmsh.model.mesh.field.setNumber(
                box_field, "XMin", np.min(points[:, 0]) - chord * 0.1
            )
            gmsh.model.mesh.field.setNumber(
                box_field, "XMax", np.max(points[:, 0]) + chord * 0.1
            )
            gmsh.model.mesh.field.setNumber(
                box_field, "YMin", np.min(points[:, 1]) - thickness * 0.1
            )
            gmsh.model.mesh.field.setNumber(
                box_field, "YMax", np.max(points[:, 1]) + thickness * 0.1
            )
            gmsh.model.mesh.field.setNumber(box_field, "ZMin", -0.1)
            gmsh.model.mesh.field.setNumber(box_field, "ZMax", 0.1)
            # apply mesh size field
            gmsh.model.mesh.field.setAsBackgroundMesh(box_field)
            if self.verbose:
                print("saving %s mesh..." % model_name)
            gmsh.write(os.path.join(self.cfd_path, "%s.geo_unrolled" % model_name))
            with open(
                os.path.join(self.cfd_path, "%s.geo_unrolled" % model_name), "r+"
            ) as file:
                content = file.read()
                file.seek(0, 0)
                file.write('SetFactory("OpenCASCADE");\n' + content)
            try:
                gmsh.model.mesh.generate(dim_mesh)
                gmsh.write(os.path.join(self.cfd_path, "%s.msh" % model_name))
                gmsh.write(os.path.join(self.cfd_path, "%s.vtk" % model_name))
            except Exception as e:
                print(f"Error generating airfoil mesh: {e}")
                raise

        # return entity info
        entity_dict = {
            "model": current_model,
            "surface": airfoil_surface,
            "entities": gmsh.model.getEntities(dim=2),
            "entity_names": [
                gmsh.model.getEntityName(dim=entity[0], tag=entity[1])
                for entity in gmsh.model.getEntities(dim=2)
            ],
        }
        return entity_dict


class Airfoil(Geometry):
    """Create airfoil geometries using splines and generate body-fitted meshes."""

    def __init__(
        self,
        parameters_dict: Dict,
        airfoil_points_list: Union[List[np.ndarray], List[str]],
        chords: List[float],
        thicknesses: List[float] | List[List[float]],
        angles: List[float],
        centers_x: List[float],
        centers_y: List[float],
        num_airfoils: int,
        dim: int,
        cambers: None | List[List[float]] = None,
        path: str = "./cfd_bank/cfd_airfoil/",
    ):
        """Initialize the Airfoil class with given parameters.

        Args:
            parameters_dict (dict): Dictionary containing configuration parameters.
                Should include keys like:
                - "case": Case name
                - "domain_parameters": Dict with "dx", "dy", "dz", "origin_x", "origin_y", "origin_z"
                - "cfd_parameters": Dict with mesh options
            airfoil_points_list (Union[List[np.ndarray],List[str]]): List of point arrays or str type (ex: "NACA0012")\
                Each array has shape (n_points, 2) with (x, y) coordinates.\
                If str type contains "morph" then it will disable the thickness scaling, \
                look for camber and thickness parameters as type List[List[float]] inside parameters_dict.
            chords (List[float]): List of chord lengths for each airfoil.
            thicknesses (List[float] | List[List[float]]): List of maximum thickness values for each airfoil, \
                or List of List of thickness distribution for morphing airfoils.
            angles (List[float]): List of angle of attack values (in degrees) for each airfoil.
            centers_x (List[float]): List of x-coordinates for airfoil centers.
            centers_y (List[float]): List of y-coordinates for airfoil centers.
            num_airfoils (int): Number of airfoils to create.
            dim (int): Dimension of the geometry (2 or 3).
            path (str): Base path for geometry output files.
        """
        super().__init__(dim, path)
        self.params = parameters_dict

        self.airfoil_points_list: List[np.ndarray] = self._handle_airfoil_points_list(
            airfoil_points_list=airfoil_points_list,
            thicknesses=thicknesses,
            cambers=cambers,
        )

        self.chords = chords
        self.thicknesses = thicknesses
        self.cambers = cambers
        self.angles = angles
        self.centers_x = centers_x
        self.centers_y = centers_y
        self.n_airfoils = num_airfoils
        self.name = self.params.get("case", "airfoil")

        self.apply_airfoil_scaling(
            disable_thickness_scaling=(
                isinstance(self.thicknesses[0], list)
                or "morph" in airfoil_points_list[0].lower()
            )
        )

        self.objects_dict = self.create_objects_dict()
        self.mesh_dict = self.create_mesh_dicts(self.objects_dict)
        self.origins = self.objects_origins(self.objects_dict)

    def _handle_airfoil_points_list(
        self,
        airfoil_points_list: Union[List[np.ndarray], List[str]],
        thicknesses: Union[List[float], List[List[float]]],
        cambers: Union[None, List[List[float]]] = None,
    ) -> List[np.ndarray]:
        """Process airfoil_points_list: either it is a list of np.ndarrays containing points coords,
        or a list of strings representing airfoil types: "NACAxxxx" or "morphed".
        If NACAxxx it will get the predefined airfoil points,
        if morphed it will generate the morphed airfoil using cambers and thicknesses.
        Runs sanity checks.

        Args:
            airfoil_points_list (Union[List[np.ndarray],List[str]]): List of point arrays or str type.
            thicknesses (Union[List[float], List[List[float]]]): List of maximum thickness values for each airfoil, \
                or List of List of thickness distribution for morphing airfoils.
            cambers (Union[None, List[List[float]]], optional): List of List of camber distributions \
                for morphing airfoils. Defaults to None because only needed for morphed airfoils.
        Returns:
            List[np.ndarray]: Processed list of airfoil point arrays.
        """
        if isinstance(airfoil_points_list, list) and all(
            isinstance(pts, np.ndarray) for pts in airfoil_points_list
        ):
            if min([airfoil_pts.shape[0] for airfoil_pts in airfoil_points_list]) < 4:
                raise ValueError("Each airfoil must have at least 4 points.")
            else:
                airfoil_points_list_ = airfoil_points_list

        elif isinstance(airfoil_points_list, list) and all(
            isinstance(pts, str) for pts in airfoil_points_list
        ):
            if "morph" in airfoil_points_list[0].lower():
                # morphed airfoil
                if cambers is None or not (isinstance(thicknesses[0], list)):
                    raise ValueError(
                        "For morphed airfoils, cambers and thicknesses must be provided as List of List of floats."
                    )
                else:
                    airfoil_points_list_ = []
                    for i, pts in enumerate(airfoil_points_list):
                        if "morph" not in pts.lower():
                            raise ValueError(
                                "All airfoil types in airfoil_points_list must be 'morphed' when using "
                                "morphed airfoils. Mixed NACA and morphed types is not implemented."
                            )
                        morphed_airfoil_points = morphed_airfoil(
                            camber_parameters=cambers[i],
                            thickness_parameters=thicknesses[i],
                        )
                        airfoil_points_list_.append(morphed_airfoil_points)
            else:
                # NACA airfoil
                airfoil_points_list_ = [
                    get_predefined_airfoil(airfoil_name=pts, n_points=20)
                    for pts in airfoil_points_list
                ]
        else:
            raise ValueError(
                "airfoil_points_list must be a list of numpy arrays or a list of strings (either NACA or morphed)."
            )
        return airfoil_points_list_

    def apply_airfoil_scaling(self, disable_thickness_scaling: bool = False):
        """Scale airfoil points according to specified chords and thicknesses."""
        for i in range(self.n_airfoils):
            points = self.airfoil_points_list[i]
            chord = self.chords[i]
            thickness = self.thicknesses[i]
            # current dimensions
            current_chord = np.max(points[:, 0]) - np.min(points[:, 0])
            current_thickness = np.max(points[:, 1]) - np.min(points[:, 1])
            # scaling factors
            scale_x = chord / current_chord if current_chord > 0 else 1.0
            scale_y = thickness / current_thickness if current_thickness > 0 else 1.0
            # scale points
            scaled_points = points.copy()
            scaled_points[:, 0] *= scale_x
            if not disable_thickness_scaling:
                scaled_points[:, 1] *= scale_y
            self.airfoil_points_list[i] = scaled_points

    def create_objects_dict(self) -> Dict:
        """Create a dictionary of airfoil objects with their properties.

        Returns:
            dict: Dictionary with keys like "airfoil1", "airfoil2", etc.
                  Each value contains airfoil-specific parameters.
        """
        objects_dict = {}
        for i in range(self.n_airfoils):
            objects_dict[f"airfoil{i+1}"] = {
                "points": self.airfoil_points_list[i],
                "angle": self.angles[i],
                "x": self.centers_x[i],
                "y": self.centers_y[i],
                "z": 0,
            }
        return objects_dict

    def create_mesh_dicts(self, objects_dict: Dict) -> Dict:
        """Create a dictionary of mesh properties for the airfoils.

        Args:
            objects_dict (dict): Dictionary of airfoil objects.

        Returns:
            dict: Dictionary with mesh properties for each airfoil.
        """
        meshdict = {}
        for i in range(self.n_airfoils):
            airfoil_points = objects_dict[f"airfoil{i+1}"]["points"]
            npoints = airfoil_points.shape[0]
            # characteristic size from chord and thickness
            chord = np.max(airfoil_points[:, 0]) - np.min(airfoil_points[:, 0])
            thickness = np.max(airfoil_points[:, 1]) - np.min(airfoil_points[:, 1])
            char_size = max(chord, thickness)
            # Uniform mesh size driven on/in airfoil
            uniform_size = max(1e-3, char_size / (npoints * 2))

            meshdict[f"airfoil{i+1}"] = {
                "mesh_in": uniform_size,
                "mesh_out": char_size,
            }
        return meshdict

    def objects_origins(self, objects_dict: Dict) -> List:
        """Get the origins of the airfoil objects.

        Args:
            objects_dict (dict): Dictionary of airfoil objects.

        Returns:
            list: List of origin coordinates [x, y, z] for each airfoil.
        """
        origins = []
        for i in range(self.n_airfoils):
            origins.append(
                [
                    objects_dict[f"airfoil{i+1}"]["x"],
                    objects_dict[f"airfoil{i+1}"]["y"],
                    objects_dict[f"airfoil{i+1}"]["z"],
                ]
            )
        return origins

    def get_domain_dimensions(self) -> List[float]:
        """Get the dimensions of the domain.

        Returns:
            list: List of domain dimensions [dx, dy, dz].
        """
        domain_params = self.params.get("domain_parameters", {})
        dx = domain_params.get("dx", 10)
        dy = domain_params.get("dy", 10)
        dz = domain_params.get("dz", 1)
        return [dx, dy, dz]

    def auto_mesh_options(self):
        """Automatically set global mesh sizes from airfoil point spacing."""
        # all points of all airfoils
        all_points = np.vstack(
            [self.objects_dict[k]["points"] for k in self.objects_dict]
        )
        diffs = np.diff(all_points, axis=0)
        edge_lengths = np.linalg.norm(diffs, axis=1)
        min_edge = float(np.min(edge_lengths)) if len(edge_lengths) else 1e-2
        chord = (
            float(np.max(all_points[:, 0]) - np.min(all_points[:, 0]))
            if len(all_points)
            else 1.0
        )

        mesh_in_values = [self.mesh_dict[k]["mesh_in"] for k in self.mesh_dict]
        mesh_out_values = [self.mesh_dict[k]["mesh_out"] for k in self.mesh_dict]
        base_size = min(mesh_in_values) if mesh_in_values else min_edge
        far_size = max(mesh_out_values) if mesh_out_values else chord

        # coarse domain, fine on object boundaries
        min_mesh = max(1e-3, base_size)
        max_mesh = max(min_mesh * 200, far_size * 4)

        # global mesh size
        self.set_mesh_size(min_mesh_size=min_mesh, max_mesh_size=max_mesh)
        self.set_meshing_options(
            mesh_size_points=1, mesh_size_curvature=100, extend_from_boundary=1
        )

    def get_domain_origin(self) -> List[float]:
        """Get the origin of the domain.

        Returns:
            list: List of domain origin coordinates [x0, y0, z0].
        """
        domain_params = self.params.get("domain_parameters", {})
        x0 = domain_params.get("origin_x", -5)
        y0 = domain_params.get("origin_y", -5)
        z0 = domain_params.get("origin_z", 0)
        return [x0, y0, z0]

    def find_box2_params(self) -> Dict:
        """Find the parameters for the boundary layer mesh (BLM) box (Box2).
        This box encloses all airfoil objects, with a margin that is
        extended downstream and laterally, for wake resolution.

        Returns:
            dict: Dictionary with "Center2" and "Box2" parameters.
        """
        # Calculate box containing all airfoils with margin
        x_coords = []
        y_coords = []

        for i in range(self.n_airfoils):
            points = self.objects_dict[f"airfoil{i+1}"]["points"]
            center_x = self.objects_dict[f"airfoil{i+1}"]["x"]
            center_y = self.objects_dict[f"airfoil{i+1}"]["y"]
            angle_deg = self.objects_dict[f"airfoil{i+1}"]["angle"]

            # rotate points counterclockwise by angle for more accurate bounding box
            angle_rad = np.deg2rad(angle_deg)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)

            rotated_points = np.array(
                [
                    points[:, 0] * cos_a - points[:, 1] * sin_a,
                    points[:, 0] * sin_a + points[:, 1] * cos_a,
                ]
            ).T

            x_coords.extend(rotated_points[:, 0] + center_x)
            y_coords.extend(rotated_points[:, 1] + center_y)

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        y_middle = 0.5 * (y_min + y_max)

        char_size = min(x_max - x_min, y_max - y_min)
        margin = max(3 * char_size, 2)

        min_box2_xdim = 0.5 * self.get_domain_dimensions()[0]
        min_box2_ydim = 0.2 * self.get_domain_dimensions()[1]

        box2_ydim = abs(y_max - y_min) + max(2 * margin, min_box2_ydim)
        box2_xdim = max(abs(x_max - x_min) + 3 * margin, min_box2_xdim)

        box2_lowleft_corner = [
            x_min - margin,
            y_middle - box2_ydim / 2,
        ]

        box2_params = {"Center2": box2_lowleft_corner, "Box2": [box2_xdim, box2_ydim]}
        return box2_params

    def apply_box2params(self):
        """
        Apply the box2 parameters.
        """
        if self.params["cfd_parameters"]["mesh_adapt"]:
            box2_params = self.find_box2_params()
            box2_path = os.path.join(self.cfd_path, "BLM", "Center2.txt")
            try:
                apply_Center2(Center2=box2_params["Center2"], Center2_path=box2_path)
                apply_Center2(
                    Center2=box2_params["Box2"],
                    Center2_path=box2_path.replace("Center2", "Box2"),
                )
            except Exception as e:
                print(f"Error applying BLM box2 parameters: {e}")
                raise

    def create_object(
        self, force_model: str = "", save_mesh: bool = False, dim_mesh: int = 2
    ) -> Dict:
        """Create all airfoil objects in a single model.

        Args:
            force_model (str): Existing model name to add to.
            save_mesh (bool): Whether to generate and save the mesh.
            dim_mesh (int): Dimension for mesh generation.

        Returns:
            dict: Dictionary containing all airfoil entities.
        """
        if force_model != "":
            gmsh.model.setCurrent(name=force_model)
        else:
            model_name = self.name
            gmsh.model.add(model_name)

        current_model = gmsh.model.getCurrent()

        # create all airfoils
        airfoils = {}
        for i in range(len(self.objects_dict)):
            airfoil_name = f"airfoil{i+1}"
            airfoils[airfoil_name] = self.create_airfoil(
                self.objects_dict[airfoil_name],
                model_name=airfoil_name,
                force_model=current_model,
                save_mesh=False,
                dim_mesh=2,
            )
            gmsh.model.setEntityName(
                dim=2, tag=airfoils[airfoil_name]["entities"][0][1], name=airfoil_name
            )
            gmsh.model.occ.synchronize()

            # impose uniform size on airfoil boundary curves and points
            uniform_size = self.mesh_dict[airfoil_name]["mesh_in"]
            boundary_curves = gmsh.model.getBoundary(
                dimTags=[airfoils[airfoil_name]["entities"][0]],
                combined=False,
                oriented=False,
                recursive=False,
            )
            curve_tags = [(1, c[1]) for c in boundary_curves if c[0] == 1]
            point_tags = []
            for c in curve_tags:
                pts = gmsh.model.getBoundary(
                    [c], combined=False, oriented=False, recursive=False
                )
                point_tags.extend([(0, p[1]) for p in pts if p[0] == 0])
            gmsh.model.mesh.setSize(curve_tags + point_tags, uniform_size)

        # generate mesh with field sizing
        if save_mesh:
            box_meshfields = []

            for i in range(self.n_airfoils):
                box_meshfield = gmsh.model.mesh.field.add(fieldType="Box")

                points = self.objects_dict[f"airfoil{i+1}"]["points"]
                char_size = min(
                    np.max(points[:, 0]) - np.min(points[:, 0]),
                    np.max(points[:, 1]) - np.min(points[:, 1]),
                )

                center_x = self.objects_dict[f"airfoil{i+1}"]["x"]
                center_y = self.objects_dict[f"airfoil{i+1}"]["y"]

                interbox = {
                    "x_min": center_x - 2 * char_size,
                    "x_max": center_x + 2 * char_size,
                    "y_min": center_y - 2 * char_size,
                    "y_max": center_y + 2 * char_size,
                    "z_min": 0,
                    "z_max": 0,
                }

                gmsh.model.mesh.field.setNumber(
                    tag=box_meshfield,
                    option="VIn",
                    value=self.mesh_dict[f"airfoil{i+1}"]["mesh_in"],
                )
                gmsh.model.mesh.field.setNumber(
                    tag=box_meshfield,
                    option="VOut",
                    value=self.mesh_dict[f"airfoil{i+1}"]["mesh_out"],
                )
                gmsh.model.mesh.field.setNumber(
                    tag=box_meshfield, option="XMin", value=interbox["x_min"]
                )
                gmsh.model.mesh.field.setNumber(
                    tag=box_meshfield, option="XMax", value=interbox["x_max"]
                )
                gmsh.model.mesh.field.setNumber(
                    tag=box_meshfield, option="YMin", value=interbox["y_min"]
                )
                gmsh.model.mesh.field.setNumber(
                    tag=box_meshfield, option="YMax", value=interbox["y_max"]
                )
                gmsh.model.mesh.field.setNumber(
                    tag=box_meshfield, option="ZMin", value=interbox["z_min"]
                )
                gmsh.model.mesh.field.setNumber(
                    tag=box_meshfield, option="ZMax", value=interbox["z_max"]
                )
                box_meshfields.append(box_meshfield)

            if box_meshfields:
                combined_field = gmsh.model.mesh.field.add("Min")
                gmsh.model.mesh.field.setNumbers(
                    combined_field, "FieldsList", box_meshfields
                )
                gmsh.model.mesh.field.setAsBackgroundMesh(combined_field)

            try:
                gmsh.model.mesh.generate(dim_mesh)

                gmsh.write(os.path.join(self.cfd_path, "object.msh"))
                gmsh.write(os.path.join(self.cfd_path, "object.vtk"))
            except Exception as e:
                print(f"Error generating combined mesh: {e}")
                raise

        # return entity info
        obj_entities_dict = {
            "model": current_model,
            "entities": gmsh.model.getEntities(dim=2),
            "entity_names": [
                gmsh.model.getEntityName(dim=entity[0], tag=entity[1])
                for entity in gmsh.model.getEntities(dim=2)
            ],
        }
        return obj_entities_dict

    def create_each_object(self, save_mesh: bool = True) -> List[Dict]:
        """Create each airfoil separately, mesh and save each individually.

        Args:
            save_mesh (bool): Whether to save each airfoil's mesh.

        Returns:
            list: List of dictionaries, each containing airfoil entities.
        """
        airfoils_dicts = []
        for i in range(self.n_airfoils):
            airfoil_name = f"airfoil{i+1}"
            airfoil_dict = self.create_airfoil(
                airfoil_dict=self.objects_dict[airfoil_name],
                model_name=f"airfoil{i}",
                force_model="",
                save_mesh=False,
                dim_mesh=self.dim,
            )

            # impose uniform size on airfoil boundary curves and points
            uniform_size = self.mesh_dict[airfoil_name]["mesh_in"]
            boundary_curves = gmsh.model.getBoundary(
                dimTags=[airfoil_dict["entities"][0]],
                combined=False,
                oriented=False,
                recursive=False,
            )
            curve_tags = [(1, c[1]) for c in boundary_curves if c[0] == 1]
            point_tags = []
            for c in curve_tags:
                pts = gmsh.model.getBoundary(
                    [c], combined=False, oriented=False, recursive=False
                )
                point_tags.extend([(0, p[1]) for p in pts if p[0] == 0])
            gmsh.model.mesh.setSize(curve_tags + point_tags, uniform_size)

            # generate mesh with field sizing
            if save_mesh:
                box_meshfield = gmsh.model.mesh.field.add(fieldType="Box")

                points = self.objects_dict[airfoil_name]["points"]
                char_size = min(
                    np.max(points[:, 0]) - np.min(points[:, 0]),
                    np.max(points[:, 1]) - np.min(points[:, 1]),
                )

                center_x = self.objects_dict[airfoil_name]["x"]
                center_y = self.objects_dict[airfoil_name]["y"]

                interbox = {
                    "x_min": center_x - 2 * char_size,
                    "x_max": center_x + 2 * char_size,
                    "y_min": center_y - 2 * char_size,
                    "y_max": center_y + 2 * char_size,
                    "z_min": 0,
                    "z_max": 0,
                }

                gmsh.model.mesh.field.setNumber(
                    tag=box_meshfield,
                    option="VIn",
                    value=self.mesh_dict[airfoil_name]["mesh_in"],
                )
                gmsh.model.mesh.field.setNumber(
                    tag=box_meshfield,
                    option="VOut",
                    value=self.mesh_dict[airfoil_name]["mesh_out"],
                )
                gmsh.model.mesh.field.setNumber(
                    tag=box_meshfield, option="XMin", value=interbox["x_min"]
                )
                gmsh.model.mesh.field.setNumber(
                    tag=box_meshfield, option="XMax", value=interbox["x_max"]
                )
                gmsh.model.mesh.field.setNumber(
                    tag=box_meshfield, option="YMin", value=interbox["y_min"]
                )
                gmsh.model.mesh.field.setNumber(
                    tag=box_meshfield, option="YMax", value=interbox["y_max"]
                )
                gmsh.model.mesh.field.setNumber(
                    tag=box_meshfield, option="ZMin", value=interbox["z_min"]
                )
                gmsh.model.mesh.field.setNumber(
                    tag=box_meshfield, option="ZMax", value=interbox["z_max"]
                )
                gmsh.model.mesh.field.setAsBackgroundMesh(box_meshfield)

                try:
                    gmsh.model.mesh.generate(self.dim)
                    gmsh.write(os.path.join(self.cfd_path, f"airfoil{i}.msh"))
                    gmsh.write(os.path.join(self.cfd_path, f"airfoil{i}.vtk"))
                except Exception as e:
                    print(f"Error generating airfoil{i} mesh: {e}")
                    raise

            airfoils_dicts.append(airfoil_dict)
        return airfoils_dicts

    def create_domain(self, save_mesh: bool = True, dim_mesh: int = 2) -> Dict:
        """Create the fluid domain around the airfoils.

        Args:
            save_mesh (bool): Whether to generate and save the domain mesh.
            dim_mesh (int): Dimension for mesh generation.

        Returns:
            dict: Dictionary containing domain entities.
        """
        model_name = "domain"
        gmsh.model.add(model_name)

        # create airfoils first
        airfoils_dict = self.create_object(force_model="", save_mesh=False, dim_mesh=2)
        entities = airfoils_dict["entities"]
        entity_names = airfoils_dict["entity_names"]

        # create domain
        domain_origin = self.get_domain_origin()
        domain_dimensions = self.get_domain_dimensions()

        fluid_domain = gmsh.model.occ.addRectangle(
            x=domain_origin[0],
            y=domain_origin[1],
            z=0,
            dx=domain_dimensions[0],
            dy=domain_dimensions[1],
        )
        gmsh.model.occ.synchronize()

        # cut airfoils from domain
        fluid_domain_tag = (2, fluid_domain)

        for entity, entityname in zip(entities, entity_names):
            try:
                out, _ = gmsh.model.occ.cut(
                    objectDimTags=[fluid_domain_tag],
                    toolDimTags=[entity],
                    removeObject=True,
                    removeTool=True,  # remove the airfoil tool from domain model
                )
                if out:
                    fluid_domain_tag = out[0]
            except Exception as e:
                if self.verbose:
                    print(f"Note: Could not cut {entityname}: {e}")

        gmsh.model.occ.synchronize()
        gmsh.model.setEntityName(
            dim=fluid_domain_tag[0], tag=fluid_domain_tag[1], name=model_name
        )

        # create mesh with fine boundaries on airfoils only
        # get airfoil boundaries and sizes
        airfoil_sizes = [
            self.mesh_dict[f"airfoil{i+1}"]["mesh_in"] for i in range(self.n_airfoils)
        ]
        boundary_curves = gmsh.model.getBoundary(
            dimTags=[fluid_domain_tag], combined=False, oriented=False, recursive=False
        )

        # for 2D: 4 outer curves (domain) + N airfoil curves
        all_curve_tags = [c[1] for c in boundary_curves if c[0] == 1]
        n_expected_outer = 4
        if len(all_curve_tags) >= n_expected_outer + self.n_airfoils:
            outer_curve_tags = set(all_curve_tags[:n_expected_outer])
            airfoil_curves = all_curve_tags[n_expected_outer:]
        else:  # fallback
            airfoil_curves = (
                all_curve_tags[-self.n_airfoils :] if all_curve_tags else []
            )
            outer_curve_tags = (
                set(all_curve_tags[: -self.n_airfoils])
                if len(all_curve_tags) > self.n_airfoils
                else set(all_curve_tags)
            )

        # apply coarse size to outer domain curves first
        if outer_curve_tags:
            mesh_out_values = [self.mesh_dict[k]["mesh_out"] for k in self.mesh_dict]
            coarse_outer = (
                max(mesh_out_values)
                if mesh_out_values
                else max(self.get_domain_dimensions()) / 5
            )
            outer_curves_dimtags = [(1, c) for c in outer_curve_tags]
            outer_points = []
            for c in outer_curves_dimtags:
                try:
                    pts = gmsh.model.getBoundary(
                        [c], combined=False, oriented=False, recursive=False
                    )
                    outer_points.extend([(0, p[1]) for p in pts if p[0] == 0])
                except Exception as e:
                    if self.verbose:
                        print(f"Note: Could not get outer boundary points: {e}")
            if outer_curves_dimtags or outer_points:
                gmsh.model.mesh.setSize(
                    outer_curves_dimtags + outer_points, coarse_outer
                )

        # apply fine size to inner domain airfoil curves
        if airfoil_curves:
            for idx, curve_tag in enumerate(airfoil_curves):
                size = airfoil_sizes[min(idx, len(airfoil_sizes) - 1)]
                curve_dimtag = (1, curve_tag)
                try:
                    pts = gmsh.model.getBoundary(
                        [curve_dimtag], combined=False, oriented=False, recursive=False
                    )
                    point_tags = [(0, p[1]) for p in pts if p[0] == 0]
                    gmsh.model.mesh.setSize([curve_dimtag] + point_tags, size)
                except Exception as e:
                    if self.verbose:
                        print(f"Note: Could not get airfoil boundary points: {e}")

        # generate and save mesh
        if save_mesh:
            gmsh.write(os.path.join(self.cfd_path, "%s.geo_unrolled" % model_name))
            with open(
                os.path.join(self.cfd_path, "%s.geo_unrolled" % model_name), "r+"
            ) as file:
                content = file.read()
                file.seek(0, 0)
                file.write('SetFactory("OpenCASCADE");\n' + content)
            try:
                gmsh.model.mesh.generate(dim_mesh)
                gmsh.write(os.path.join(self.cfd_path, f"{model_name}.msh"))
                gmsh.write(os.path.join(self.cfd_path, f"{model_name}.vtk"))
                if self.verbose:
                    print("Domain mesh saved")
            except Exception as e:
                print(f"Error generating domain mesh: {e}")
                raise

        domain_dict = {
            "model": model_name,
            "surface": fluid_domain_tag[1],
            "entities": gmsh.model.getEntities(dim=2),
            "entity_names": [
                gmsh.model.getEntityName(dim=entity[0], tag=entity[1])
                for entity in gmsh.model.getEntities(dim=2)
            ],
        }
        return domain_dict
