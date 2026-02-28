import os
import sys
import gmsh
import subprocess
import numpy as np
import random as rd
import matplotlib.pyplot as plt

from graphdrl.environment.idw import *


class Geometry:
    def __init__(self, dim: int, path: str = "./", verbose: bool = False):
        """Initialize the Geometry class with given parameters.

        : param dim: (int) Dimension of the geometry.
        : param path: (str) Path to the directory where the geometry will be saved
        and mesh generated. Should be a cfd directory, with BLM subdirectory.
        : param verbose: (bool) Whether to print messages of mesh generation info.
        """
        self.dim = dim
        self.path = os.path.abspath(path)
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
            gmsh.write(os.path.join(self.path, "%s.geo_unrolled" % model_name))
            with open(
                os.path.join(self.path, "%s.geo_unrolled" % model_name), "r+"
            ) as file:
                content = file.read()
                file.seek(0, 0)
                file.write('SetFactory("OpenCASCADE");\n' + content)
            try:
                gmsh.model.mesh.generate(dim_mesh)
                gmsh.write(os.path.join(self.path, "%s.msh" % model_name))
                gmsh.write(os.path.join(self.path, "%s.vtk" % model_name))
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
            gmsh.write(os.path.join(self.path, "%s.geo_unrolled" % model_name))
            with open(
                os.path.join(self.path, "%s.geo_unrolled" % model_name), "r+"
            ) as file:
                content = file.read()
                file.seek(0, 0)
                file.write('SetFactory("OpenCASCADE");\n' + content)
            try:
                gmsh.model.mesh.generate(dim_mesh)
                gmsh.write(os.path.join(self.path, "%s.msh" % model_name))
                gmsh.write(os.path.join(self.path, "%s.vtk" % model_name))
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
            gmsh.write(os.path.join(self.path, "%s.geo_unrolled" % model_name))
            with open(
                os.path.join(self.path, "%s.geo_unrolled" % model_name), "r+"
            ) as file:
                content = file.read()
                file.seek(0, 0)
                file.write('SetFactory("OpenCASCADE");\n' + content)
            try:
                gmsh.model.mesh.generate(dim_mesh)
                gmsh.write(os.path.join(self.path, "%s.msh" % model_name))
                gmsh.write(os.path.join(self.path, "%s.vtk" % model_name))
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
                - d1 (float): first distance of first point
                - d2 (float): second distance of first point
                - d3 (float): distance of second point
                - d4 (float): distance of third point
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
            gmsh.write(os.path.join(self.path, "%s.geo_unrolled" % model_name))
            with open(
                os.path.join(self.path, "%s.geo_unrolled" % model_name), "r+"
            ) as file:
                content = file.read()
                file.seek(0, 0)
                file.write('SetFactory("OpenCASCADE");\n' + content)
            try:
                gmsh.model.mesh.generate(dim_mesh)
                gmsh.write(os.path.join(self.path, "%s.msh" % model_name))
                gmsh.write(os.path.join(self.path, "%s.vtk" % model_name))
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


class Panels(Geometry):
    """
    Class inherited from Geometry to create a geometry with multiple panels.
    """

    def __init__(
        self,
        parameters_dict: dict,
        angles: list[float],
        num_panels: int,
        dim: int,
        path: str = "./",
    ):
        """
        Initialize the Panels class with given parameters. Parameters are read from a
        dictionary formatted in the following way:
        {
            "geometry_parameters": {
                "origin": [x0, y0, z0],
                "chord": chord,
                "span": span,
                "thickness": thickness,
                "spacing": spacing
            },
            "domain_parameters": {
                "dx": dx,
                "dy": dy,
                "dz": dz,
                "origin_x": x0_domain,
                "origin_y": y0_domain,
                "origin_z": z0_domain
            },
            "traj_parameters": {
                "Hbox123": [h_min, h_inter, h_max],
                "mesh_adapt": True/False
            }
        }
        """
        super().__init__(dim, path)
        self.params = parameters_dict
        self.angles = angles
        self.origin = self.params["geometry_parameters"]["origin"]
        self.chord = self.params["geometry_parameters"]["chord"]
        self.span = self.params["geometry_parameters"]["span"]
        self.thickness = self.params["geometry_parameters"]["thickness"]
        self.spacing = self.params["geometry_parameters"]["spacing"]
        self.n_panels = num_panels

        self.objects_dict = self.create_objects_dict()
        self.mesh_dict = self.create_mesh_dicts(self.objects_dict)
        self.origins = self.objects_origins(self.objects_dict)
        self.name = "panels"

    def create_objects_dict(self) -> dict:
        """
        Create a dictionary of panel objects with their properties.

        Returns:
            dict: Dictionary of objects.
        """
        objects_dict = {}
        for i in range(self.n_panels):
            objects_dict[f"panel{i+1}"] = {
                "angle": self.angles[i],
                "chord": self.chord,
                "span": self.span,
                "thickness": self.thickness,
                "x": self.origin[0] + i * self.spacing,
                "y": self.origin[1],
                "z": self.origin[2],
            }
        return objects_dict

    def create_mesh_dicts(self, objects_dict: dict) -> dict:
        """
        Create a dictionary of mesh properties for the panels.

        Args:
            objects_dict (dict): Dictionary of objects.

        Returns:
            dict: Dictionary of mesh properties.
        """
        meshdict = {}
        for i in range(self.n_panels):
            meshdict[f"panel{i+1}"] = {
                "mesh_object": objects_dict[f"panel{i+1}"]["thickness"] / 2,
                "mesh_in": objects_dict[f"panel{i+1}"]["chord"] / 20,
                "mesh_out": objects_dict[f"panel{i+1}"]["chord"] / 2,
            }
        return meshdict

    def objects_origins(self, objects_dict: dict) -> list:
        """
        Get the origins of the panel objects.

        Args:
            objects_dict (dict): Dictionary of objects.

        Returns:
            list: List of origins.
        """
        origins = []
        for i in range(self.n_panels):
            panel_origin = [
                self.origin[0] + i * self.spacing,
                self.origin[1],
                self.origin[2],
            ]
            origins.append(panel_origin)
        return origins

    def get_domain_dimensions(self) -> list:
        """
        Get the dimensions of the domain.

        Returns:
            list: List of domain dimensions.
        """
        dx = self.params["domain_parameters"]["dx"]
        dy = self.params["domain_parameters"]["dy"]
        dz = self.params["domain_parameters"]["dz"]
        return [dx, dy, dz]

    def get_domain_origin(self) -> list:
        """
        Get the origin of the domain.

        Returns:
            list: List of domain origin coordinates.
        """
        x0 = self.params["domain_parameters"]["origin_x"]
        y0 = self.params["domain_parameters"]["origin_y"]
        z0 = self.params["domain_parameters"]["origin_z"]
        return [x0, y0, z0]

    def find_box2_params(self) -> dict:
        """
        Find the parameters for the box2.

        Returns:
            dict: Dictionary of box2 parameters.
        """
        x_min = self.origin[0] - 1.5 * self.chord
        y_min = self.origin[1] - 1.5 * self.chord
        dx = self.n_panels * self.spacing + 3 * self.chord
        dy = self.chord * 3
        box2_params = {"Center2": [x_min, y_min], "Box2": [dx, dy]}
        return box2_params

    def apply_box2params(self):
        """
        Apply the box2 parameters.
        """
        if self.params["traj_parameters"]["mesh_adapt"]:
            box2_params = self.find_box2_params()
            box2_path = os.path.join(self.path, "BLM", "Center2.txt")
            try:
                with open(box2_path, "w") as f:
                    f.write(" ".join(map(str, box2_params["Center2"])))
                with open(box2_path.replace("Center2", "BLMbox2.txt"), "w") as f:
                    f.write(" ".join(map(str, box2_params["Box2"])))
            except Exception as e:
                print(f"Error applying BLM box2 parameters: {e}")
                raise

    def create_object(
        self, force_model: str = "", save_mesh: bool = False, dim_mesh: int = 2
    ) -> dict:
        """
        Create the multi panels object with given parameters.

        Args:
            force_model (str): Force model name.
            save_mesh (bool): Whether to save the mesh.
            dim_mesh (int): Dimension of the mesh.

        Returns:
            dict: Dictionary of object entities.
        """
        if force_model != "":
            gmsh.model.setCurrent(name=force_model)
        else:
            model_name = self.name
            gmsh.model.add(model_name)
        current_model = gmsh.model.getCurrent()

        panels = {}
        for i in range(len(self.objects_dict)):
            panel_name = f"panel{i+1}"
            panels[panel_name] = self.create_rectangle(
                self.objects_dict[panel_name],
                model_name=panel_name,
                force_model=model_name,
                save_mesh=False,
                dim_mesh=2,
            )
            gmsh.model.setEntityName(
                dim=2, tag=panels[panel_name]["entities"][0][1], name=panel_name
            )
            gmsh.model.occ.synchronize()

        if save_mesh:
            box_meshfields = []
            for i in range(self.n_panels):
                box_meshfield = gmsh.model.mesh.field.add(fieldType="Box")
                interbox = {
                    "x_min": self.objects_dict[f"panel{i+1}"]["x"]
                    - self.objects_dict[f"panel{i+1}"]["chord"] / 2,
                    "x_max": self.objects_dict[f"panel{i+1}"]["x"]
                    + self.objects_dict[f"panel{i+1}"]["chord"] / 2,
                    "y_min": self.objects_dict[f"panel{i+1}"]["y"]
                    - self.objects_dict[f"panel{i+1}"]["chord"] / 2,
                    "y_max": self.objects_dict[f"panel{i+1}"]["y"]
                    + self.objects_dict[f"panel{i+1}"]["chord"] / 2,
                    "z_min": 0,
                    "z_max": 0,
                }
                gmsh.model.mesh.field.setNumber(
                    tag=box_meshfield,
                    option="VIn",
                    value=self.mesh_dict[f"panel{i+1}"]["mesh_object"],
                )
                gmsh.model.mesh.field.setNumber(
                    tag=box_meshfield,
                    option="VOut",
                    value=self.mesh_dict[f"panel{i+1}"]["mesh_in"],
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
            combined_field = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(
                combined_field, "FieldsList", box_meshfields
            )
            gmsh.model.mesh.field.setAsBackgroundMesh(combined_field)

            gmsh.write(os.path.join(self.path, "object.geo_unrolled"))
            try:
                gmsh.model.mesh.generate(dim_mesh)
                gmsh.write(os.path.join(self.path, "object.msh"))
                gmsh.write(os.path.join(self.path, "object.vtk"))
            except Exception as e:
                print(f"Error generating object mesh: {e}")
                raise
        # dict of entities
        obj_entities_dict = {
            "model": current_model,
            "entities": gmsh.model.getEntities(dim=2),
            "entity_names": [
                gmsh.model.getEntityName(dim=entity[0], tag=entity[1])
                for entity in gmsh.model.getEntities(dim=2)
            ],
        }
        return obj_entities_dict

    def create_domain(
        self, save_mesh: bool = True, dim_mesh: int = 2, boundary_layer: bool = False
    ) -> dict:
        """
        Create the panels domain with given parameters.

        Args:
            save_mesh (bool): Whether to save the mesh.
            dim_mesh (int): Dimension of the mesh.

        Returns:
            dict: Dictionary of domain entities.
        """
        model_name = "domain"
        gmsh.model.add(model_name)
        panels_dict = self.create_object(force_model="", save_mesh=False, dim_mesh=2)
        entities, entity_names = panels_dict["entities"], panels_dict["entity_names"]
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
        fluid_domain = [(2, fluid_domain)]
        for tool, toolname in zip(entities, entity_names):
            if self.verbose:
                print(
                    "cutting %s (tag %s) from fluid domain (tag %s)"
                    % (toolname, tool[1], fluid_domain[0][1])
                )
            fluid_domain = gmsh.model.occ.cut(
                objectDimTags=fluid_domain,
                toolDimTags=[tool],
                removeObject=True,
                removeTool=True,
            )[0]
        gmsh.model.occ.synchronize()
        gmsh.model.setEntityName(
            dim=fluid_domain[0][0], tag=fluid_domain[0][1], name=model_name
        )

        if boundary_layer:
            # get boundary and create boundary layer mesh field
            bl_meshsize = min(
                [
                    self.mesh_dict[f"panel{i}"]["mesh_object"] / 2
                    for i in range(1, self.n_panels + 1)
                ]
            )
            bl_thickness = max(
                [
                    self.objects_dict[f"panel{i}"]["thickness"] * 3
                    for i in range(1, self.n_panels + 1)
                ]
            )
            boundary = gmsh.model.getBoundary(
                dimTags=fluid_domain, combined=False, oriented=False, recursive=True
            )
            boundary_layer_field = gmsh.model.mesh.field.add(fieldType="BoundaryLayer")
            gmsh.model.mesh.field.setNumbers(
                tag=boundary_layer_field,
                option="CurvesList",
                values=[boundary[0][1]]
                + [
                    boundary[4 + i][1]
                    for i in range(len(boundary) - 4)
                    # boundary[4 + i][1] for i in range(self.n_panels * 4)
                ],  # objects come after 4 walls, panels are 4 curves
            )
            gmsh.model.mesh.field.setNumber(
                tag=boundary_layer_field,
                option="Size",
                value=bl_meshsize,
            )
            gmsh.option.setNumber(
                name="Mesh.BoundaryLayerFanElements",
                value=5,
            )  # for fans at sharp corners
            gmsh.model.mesh.field.setNumbers(
                tag=boundary_layer_field,
                option="FanPointsList",
                values=[boundary[4 + i][1] for i in range(self.n_panels * 4)],
            )
            gmsh.model.mesh.field.setNumber(
                tag=boundary_layer_field, option="Ratio", value=2
            )
            gmsh.model.mesh.field.setNumber(
                tag=boundary_layer_field,
                option="Thickness",
                value=bl_thickness,
            )
            gmsh.model.mesh.field.setAsBoundaryLayer(tag=boundary_layer_field)

        if save_mesh:
            if self.verbose:
                print("saving %s mesh..." % model_name)
            gmsh.write(os.path.join(self.path, "%s.geo_unrolled" % model_name))
            with open(
                os.path.join(self.path, "%s.geo_unrolled" % model_name), "r+"
            ) as file:
                content = file.read()
                file.seek(0, 0)
                file.write('SetFactory("OpenCASCADE");\n' + content)
            if not boundary_layer:
                box_meshfields = []
                ground_meshfield = gmsh.model.mesh.field.add(fieldType="Box")
                gmsh.model.mesh.field.setNumber(
                    tag=ground_meshfield,
                    option="VIn",
                    value=self.mesh_dict["panel1"]["mesh_in"],
                )
                gmsh.model.mesh.field.setNumber(
                    tag=ground_meshfield,
                    option="VOut",
                    value=self.mesh_dict["panel1"]["mesh_out"],
                )
                gmsh.model.mesh.field.setNumber(
                    tag=ground_meshfield, option="XMin", value=domain_origin[0]
                )
                gmsh.model.mesh.field.setNumber(
                    tag=ground_meshfield,
                    option="XMax",
                    value=domain_origin[0] + domain_dimensions[0],
                )
                gmsh.model.mesh.field.setNumber(
                    tag=ground_meshfield, option="YMin", value=domain_origin[1]
                )
                gmsh.model.mesh.field.setNumber(
                    tag=ground_meshfield, option="YMax", value=domain_origin[1] + 0.05
                )
                gmsh.model.mesh.field.setNumber(
                    tag=ground_meshfield, option="ZMin", value=domain_origin[2]
                )
                gmsh.model.mesh.field.setNumber(
                    tag=ground_meshfield, option="ZMax", value=domain_origin[2]
                )
                box_meshfields.append(ground_meshfield)
                for i in range(self.n_panels):
                    box_meshfield = gmsh.model.mesh.field.add(fieldType="Box")
                    interbox = {
                        "x_min": self.objects_dict[f"panel{i+1}"]["x"]
                        - self.objects_dict[f"panel{i+1}"]["chord"] / 2,
                        "x_max": self.objects_dict[f"panel{i+1}"]["x"]
                        + self.objects_dict[f"panel{i+1}"]["chord"] / 2,
                        "y_min": self.objects_dict[f"panel{i+1}"]["y"]
                        - self.objects_dict[f"panel{i+1}"]["chord"] / 2,
                        "y_max": self.objects_dict[f"panel{i+1}"]["y"]
                        + self.objects_dict[f"panel{i+1}"]["chord"] / 2,
                        "z_min": 0,
                        "z_max": 0,
                    }
                    gmsh.model.mesh.field.setNumber(
                        tag=box_meshfield,
                        option="VIn",
                        value=self.mesh_dict[f"panel{i+1}"]["mesh_in"],
                    )
                    gmsh.model.mesh.field.setNumber(
                        tag=box_meshfield,
                        option="VOut",
                        value=self.mesh_dict[f"panel{i+1}"]["mesh_out"],
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

                combined_field = gmsh.model.mesh.field.add("Min")
                gmsh.model.mesh.field.setNumbers(
                    combined_field, "FieldsList", box_meshfields
                )
                gmsh.model.mesh.field.setAsBackgroundMesh(combined_field)
            try:
                gmsh.model.mesh.generate(dim_mesh)
                gmsh.write(os.path.join(self.path, "%s.msh" % model_name))
                gmsh.write(os.path.join(self.path, "%s.vtk" % model_name))
            except Exception as e:
                print(f"Error generating domain mesh: {e}")
                raise
        # dict of entities
        domain_dict = {
            "model": model_name,
            "surface": fluid_domain[0][1],
            "entities": gmsh.model.getEntities(dim=2),
            "entity_names": [
                gmsh.model.getEntityName(dim=entity[0], tag=entity[1])
                for entity in gmsh.model.getEntities(dim=2)
            ],
        }
        return domain_dict

    def create_each_object(
        self,
        save_mesh: bool = True,
    ) -> list[dict]:
        """
        Create each single panel object separately,
        mesh and save each of them.

        Args:
            save_mesh (bool): Whether to save each object's mesh.

        Returns:
            dict: Dictionary of each panel entities.
        """
        panel_dicts = []
        for i in range(self.n_panels):
            panel_dict = self.create_rectangle(
                self.objects_dict[f"panel{i+1}"],
                model_name=f"panel{i}",
                force_model="",
                save_mesh=save_mesh,
                dim_mesh=self.dim,
            )
            panel_dicts.append(panel_dict)
        return panel_dicts

#####  FOIL CLASS FOR FOIL OPTIMIZATIONS  ######

# helper: orientation test
def _orient(a, b, c):
    return (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])

# helper: check if point c lies on segment ab
def _on_segment(a, b, c):
    return (min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and
            min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))

# check if two segments (a1-a2) and (b1-b2) intersect
def _segments_intersect(a1, a2, b1, b2):
    o1 = _orient(a1, a2, b1)
    o2 = _orient(a1, a2, b2)
    o3 = _orient(b1, b2, a1)
    o4 = _orient(b1, b2, a2)

    if o1 == 0 and _on_segment(a1, a2, b1):
        return True
    if o2 == 0 and _on_segment(a1, a2, b2):
        return True
    if o3 == 0 and _on_segment(b1, b2, a1):
        return True
    if o4 == 0 and _on_segment(b1, b2, a2):
        return True

    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)

# check if polygon defined by pts has any self-intersections
def _is_self_intersecting(pts):
    n = len(pts)
    if n < 4:
        return False
    edges = [(i, (i + 1) % n) for i in range(n)]
    for i, (a1_i, a2_i) in enumerate(edges):
        for j, (b1_i, b2_i) in enumerate(edges):
            # skip same edge or adjacent edges (they share a node)
            if abs(i - j) <= 1 or (i == 0 and j == n - 1) or (j == 0 and i == n - 1):
                continue
            if _segments_intersect(pts[a1_i], pts[a2_i], pts[b1_i], pts[b2_i]):
                return True
    return False


class Foil:

    def __init__(self, number_of_points, chord_length_multiplier, thickness_multiplier,
                 work_dir: str = "", name : str = "object", suffix: str = ""):
        """
        work_dir: base directory for this instance (e.g., geometry/mesh/{ep})
                  Files will be written under work_dir/{txt,geo,msh,t}.
                  If None, falls back to original script_dir/{txt_files,geo_files,msh_files,t_files}.
        suffix:   string appended to file basenames (e.g., '_17') to ensure uniqueness.
        """
        self.msh_size = 0.01
        self.type = "spline"
        self.number_of_points = number_of_points
        self.chord_length = chord_length_multiplier
        self.thickness_multiplier = thickness_multiplier
        self.points = np.array(self.generate_airfoil_points())
        self.origin = self.points[:, :1].argmin()  # Furthest left point
        self.surface = self.compute_surface()
        self.name = name  # base name without suffix
        self.suffix = suffix
        # ---- new: episode-local work area
        self.work_dir = work_dir  # e.g., geometry/mesh/{ep}
        self._init_dirs()

    # ---- new: directory resolver
    def _init_dirs(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        script_dir = self.script_dir
        if self.work_dir is None:
            # backward compatible layout
            self.txt_dir = os.path.join(script_dir, "txt_files")
            self.geo_dir = os.path.join(script_dir, "geo_files")
            self.msh_dir = os.path.join(script_dir, "msh_files")
            self.t_dir   = os.path.join(script_dir, "t_files")
            self.tmp_msh_dir = os.path.join(script_dir, "tmp_msh_files")
        else:
            # episode-local, parallel-safe layout
            self.txt_dir = os.path.join(self.work_dir, "txt")
            self.geo_dir = os.path.join(self.work_dir, "geo")
            self.msh_dir = os.path.join(self.work_dir, "msh")
            self.t_dir   = os.path.join(self.work_dir, "t")
            self.tmp_msh_dir = os.path.join(self.work_dir, "tmp_msh")

        for d in (self.txt_dir, self.geo_dir, self.msh_dir, self.t_dir, self.tmp_msh_dir):
            os.makedirs(d, exist_ok=True)

    def _base(self):
        """Unified basename including optional suffix."""
        return f"{self.name}{self.suffix}"


    def generate_airfoil_points(self, random : bool = False):
        """
        Generates a list of number_of_points points randomly if random == true or a NACA 0010 profile if false.
        """
        points = []
        if random :
            for i in range(self.number_of_points-1):
                x = rd.uniform(0, self.chord_length)
                y = rd.uniform(-self.thickness_multiplier/2, self.thickness_multiplier/2)

                points.append([x, y])
            points.append([0, 0])
            points = np.ndarray(points)
            points = self.order_points()

            return points

        # If not random : NACA0010 profile (for splines), points already ordered from trailing edge top to trailing edge bottom
        points = np.array([
            [1.00000,  0.00105],
            [0.8, 0.02],
            [0.5, 0.043],
            [0.3, 0.05],
            [0.12, 0.042],
            [0.025, 0.022],
            [0.0,  0.0],
            [0.025, -0.022],
            [0.12, -0.042],
            [0.3, -0.05],
            [0.5, -0.043],
            [0.8, -0.02],
            [1.00000,  -0.00105],
            [1.00000,  0.00000]
            ])
        
        multiplier = np.full((len(points), 2), np.array([self.chord_length,self.thickness_multiplier]))
        points = np.array(np.multiply(points, multiplier))

        return points
    

    
    def compute_surface(self) :
        """ Computes the area of the polygon formed by the points stored in self.points"""
        points = np.array(self.points)
        points = points.reshape(-1,2)

        x = points[:,0]
        y = points[:,1]

        S1 = float(np.sum(x*np.roll(y,-1)))
        S2 = float(np.sum(y*np.roll(x,-1)))

        self.surface = 0.5*np.abs(S1-S2)

        return self.surface
        
    
    def order_points(self):
        """
        Orders the point of the airfoil, with increasing indices in the trigonometric direction, and starting from the rightmost point right above the x-axis.
        """
        # sort by polar angle (counter-clockwise)
        angles = [np.arctan2(p[1], p[0]) for p in self.points]

        order = np.argsort(angles)
        sorted_points = [self.points[i] for i in order]

        # find the "top-right" point: maximal x, break ties by maximal y
        start_idx = max(range(len(sorted_points)), key=lambda i: (sorted_points[i][0], sorted_points[i][1]))

        # rotate so the sequence starts at the top-right point
        self.points = sorted_points[start_idx:] + sorted_points[:start_idx]

    def plot(self):
        """
        Plots the airfoil using matplotlib.
        """
        x, y = zip(*self.points)
        plt.figure()
        plt.plot(x, y, 'o-')
        plt.title('Airfoil Shape')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.grid(True)
        plt.show()

    def apply_symmetrical_y_actions(self, actions):
        """ 
        Changes self.points according to the actions, for a symmetrical foil profile
        and changing only the y coordinate of the points
        Args :
            actions : np.array([x,y], ....)
        Actions arguments are ordered as the points (trigonometric order starting at the top of 
        leading edge for NACA0010 basic profil of the class Foil)
        """
        current_points = np.array(self.points)

        origin = self.origin
        #print("Origine : ", self.origin)

        # Get current points and extract their y coordinates for points bewteen trailing edge and origin-1
        """ action x = np.array([thickness_te/2, y1, y2, y3, y4, y5]) """
        new_y_s = actions
        new_points = np.array(current_points)
        if np.any(actions <= 0.0):
            raise ValueError("Actions cannot be negative ! (y_coord of upper surface, symmetrical foil points)")
        if len(new_y_s) > origin :
            raise ValueError("Action is too large, larger than the number of control points")
        if len(new_y_s) < origin :
            raise ValueError("Action is too short, shorter than the number of control points")
        try:
            for i in range(len(current_points[:origin,1:])):
                new_points[i, 1] = new_y_s[i]
                new_points[-(i+2), 1] = -new_y_s[i]
        except ValueError as e:
            print(f"Error: {e}")

        self.points = new_points
        return
    
    def apply_camber_thickness(self, actions):
        """
        Changes the airfoil points based actions structured as follows:
        - The first N actions are the y coordinates of the camber points 
        (except the leading and trailing edges which are fixed), whose x coordinates are know from the airfoil points
        - The N following actions are the thickness distribution, which is applied symmetrically in the horizontal direction at the camber points (except for the leading and trailing edges)
        - The last action is the angle of rotation

        Args :
            actions (np.array): The actions to apply to the airfoil.
        Returns :
            None
        """
        N = (len(actions)-1)//2 # N = 3
        camber = actions[:N+1] #camber is of length N+1 (4)
        thickness = actions[N+1:] #thickness is of length N (3)
        origin = self.origin


        points = self.points
        new_points = np.array(points)
        if np.any(thickness < 0.0):
            raise ValueError("Thickness cannot be negative !")
        if len(camber) > 4:
            raise ValueError("Camber is too large, larger than the number of control points")
        if len(camber) < 4:
            raise ValueError("Camber is too short, shorter than the number of control points")
        if len(thickness) > 3:
            raise ValueError("Thickness is too large, larger than the number of control points")
        if len(thickness) < 3:
            raise ValueError("Thickness is too short, shorter than the number of control points")
        try:
            for i in range(N+1):
                c = camber[i]
                if i == 0:
                    new_points[i+1, 1] += c
                    new_points[-(i+3), 1] += c
                else:
                    t = thickness[i-1]

                    new_points[i+1, 1] = c + t/2
                    new_points[-(i+3), 1] = c - t/2

        except ValueError as e:
            print(f"Error: {e}")

        self.points = new_points

        return 

    def apply_transform_point(self, 
                  point_ind : int,
                  transformation_parameters,
                  transformation : str = "new_coord",
                  constraint_parameter = 1.0,
                  constraint : str = "max_thickness"):
        """
        Applies a transformation to a specific point of the airfoil.
        Since we want to avoid entanglement, we need to be careful with how we transform points.
        Args: 
            point_ind (int): The index of the point to transform.
            transformation_parameters (tuple[float, float]): The parameters for the transformation.
            transformation (str): The type of transformation to apply 
                (among 'new_coord' which changes the point's coordinates, 'translation' which moves the point, and 'scaling' which scales the point).
        """
        if 0 <= point_ind < len(self.points):
            candidate = list(self.points)
            constraint_satisfied = False 
            # Work on a copy and validate the transformation won't create intersections
            
            #COMPUTE NEW POINT CANDIDATE
            if transformation == "new_coord":
                new_x, new_y = transformation_parameters
                candidate[point_ind] = (new_x, new_y)

            elif transformation == "translation":
                dx, dy = transformation_parameters
                candidate[point_ind] = (candidate[point_ind][0] + dx,
                                        candidate[point_ind][1] + dy)
            elif transformation == "scaling":
                sx, sy = transformation_parameters
                candidate[point_ind] = (candidate[point_ind][0] * sx,
                                        candidate[point_ind][1] * sy)
            else:
                raise ValueError("Unknown transformation type. Please select from new_coord, translation and scaling")
            
            #CHECK WHETHER NEW POINT CANDIDATE RESPECTS CONSTRAINTS OR NOT
            if constraint == "max_thickness":
                #checks whether the delta between the max of the y coordinates of all points and the min is above constraint_parameter
                y_max = max(self.points[1])
                y_min = min(self.points[1])
                delta = y_max - y_min
                # print(delta)

                constraint_satisfied = (delta > constraint_parameter)

            if not _is_self_intersecting(candidate) and constraint_satisfied:
                self.points = candidate
            else:
                print("Transformation would create self-intersections or violate constraints")
                return -10
                
                
            
        else:
            raise IndexError("Point index out of range")
        
        return 10


    def apply_translation(self, x, y):
        """
        Translates the whole airfoil by a given (x, y) vector.
        """
        shape = np.array(self.points).shape
        translation = np.array(np.full(shape, [x, y]))
        self.points = self.points + translation


    def apply_rotation(self, angle):
        """
        Rotates the whole airfoil by a given angle in radians around its center of mass.
        """
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        #Compute the coordinates of the center of mass
        x_center = np.mean([x for x, y in self.points])
        y_center = np.mean([y for x, y in self.points])
        rotated_points = []
        for x, y in self.points: #rotates around the center of mass
            x_rot = x_center + (x - x_center) * cos_angle - (y - y_center) * sin_angle
            y_rot = y_center + (x - x_center) * sin_angle + (y - y_center) * cos_angle
            rotated_points.append((x_rot, y_rot))
        self.points = rotated_points


    def get_geo(self):
        """
        Generates .geo file of the foil's geometry
        
        Returns : path (str) to the created .geo
        """
        geo_output = os.path.join(self.geo_dir, f"{self._base()}.geo_unrolled")
        try:
            gmsh.initialize(sys.argv)
            gmsh.option.setNumber("General.Verbosity", 2)
            gmsh.option.setNumber("General.Terminal", 1)
            gmsh.option.setNumber("General.AbortOnError", 1)  # 1 = raise on error

            gmsh.model.add("object")
            for i in range(len(self.points)):
                gmsh.model.geo.addPoint(self.points[i][0], self.points[i][1], 0, self.msh_size, i)
            
            # Generate the points for the spline connection and the trailing edge line
            spline_points = [i for i in range(len(self.points)-1)]
            gmsh.model.geo.add_spline([len(self.points)-2, len(self.points)-1, 0], 1)
            gmsh.model.geo.add_spline(spline_points, 2)

            gmsh.model.geo.addCurveLoop([1, 2], 1)
            gmsh.model.geo.addPlaneSurface([-1], 1)

            gmsh.model.geo.synchronize()
            gmsh.write(geo_output)
        except:
            raise RuntimeError("gmsh Python API was unable to build .geo file.")
        finally:
            gmsh.finalize()

        return geo_output

    def get_mesh_timeout(self, geo_input: str, timeout: int = 60) -> str:
        """
        Meshing via a separate Python interpreter that imports gmsh.
        Works inside daemonic processes. Enforces a hard timeout.
        """
        current_working_dir = os.getcwd()
        msh_output = os.path.join(current_working_dir, self.msh_dir, f"{self._base()}.msh")
        os.makedirs(self.msh_dir, exist_ok=True)

        geo_input_abs = os.path.join(current_working_dir, geo_input)
        if not os.path.isfile(geo_input_abs):
            raise FileNotFoundError(f".geo not found: {geo_input_abs}")

        # Remove stale output
        try:
            if os.path.exists(msh_output):
                os.remove(msh_output)
        except Exception:
            pass

        cmd = [sys.executable, "mesh_worker.py", geo_input_abs, msh_output]
        working_dir = self.script_dir
        try:
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
                cwd=working_dir,
            )

            if res.stdout:  # Only print if non-empty
                print(res.stdout, flush=True)
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Gmsh meshing timed out after {timeout}s")

        if (not os.path.isfile(msh_output)): # or os.path.getsize(msh_output) == 0:
            raise RuntimeError("Gmsh produced no .msh file")
        
        
        if res.returncode != 0:
            raise RuntimeError(f"Worker failed (rc={res.returncode})")
        return msh_output

    def convert_gmsh_to_mtc(self, input: str, output: str, verbose: bool = True) -> str:
        """
        Convert a gmsh mesh file to an mtc (.t) mesh file.

        Args:
        input (str): Path to the input gmsh mesh file.
        output (str): Path to the output mtc mesh file.
        verbose (bool): Print progress to stdout.
        """
        if verbose:
            print("Initialisation...\n")

        with open(input) as f:
            f.readline()
            version = f.readline().split()[0]
            if len(version) > 1:
                version = version.split(".")[0]
            if version != "4" and version != "2":
                raise ValueError("This version of gmsh isn't supported")

            flags = {"$Nodes": [], "$EndNodes": [], "$Elements": [], "$EndElements": []}

            connect_3d = []
            connect_2d = []
            connect_1d = []

            if verbose:
                print("Getting position flags...\n")

            t = f.readline()

            while t:
                t = t.strip("\t\n")
                if t.startswith("$"):
                    for i in range(len(list(flags.keys()))):
                        if t == list(flags.keys())[i]:
                            flags[t].append(f.tell())
                            break
                t = f.readline()

            if verbose:
                print("Treating connectivities...\n")

            if version == "4":
                for index in range(len(flags["$Elements"])):
                    f.seek(flags["$Elements"][index])

                    t = f.readline()  # line ignored (nb of elements)
                    t = f.readline()

                    while t and f.tell() != flags["$EndElements"][index]:
                        t = t.strip("\t\n").split()

                        if len(t) <= 1:
                            break

                        if t[2] != "2" and t[2] != "4":
                            for i in range(int(t[-1])):
                                f.readline()

                        if t[2] == "2":  # triangle
                            for i in range(int(t[-1])):
                                elem = f.readline().strip("\t\n").split()
                                lig = [int(elem[1]), int(elem[2]), int(elem[3])]
                                connect_2d.append(lig)

                        if t[2] == "4":  # tetrahedron
                            for i in range(int(t[-1])):
                                elem = f.readline().strip("\t\n").split()
                                lig = [
                                    int(elem[1]),
                                    int(elem[2]),
                                    int(elem[3]),
                                    int(elem[4]),
                                ]
                                connect_3d.append(lig)

                        t = f.readline()

            if version == "2":
                for index in range(len(flags["$Elements"])):
                    f.seek(flags["$Elements"][index])

                    t = f.readline()  # line ignored (nb of elements)
                    t = f.readline()

                    while t and f.tell() != flags["$EndElements"][index]:
                        t = t.split()

                        if len(t) <= 1:
                            break

                        if t[1] == "2":  # triangle
                            lig = [int(t[-3]), int(t[-2]), int(t[-1])]
                            connect_2d.append(lig)

                        if t[1] == "4":  # tetrahedron
                            lig = [int(t[-4]), int(t[-3]), int(t[-2]), int(t[-1])]
                            connect_3d.append(lig)

                        t = f.readline()

            # Correction for gmsh numbering
            connect_2d = np.array(connect_2d, dtype=int) - 1
            connect_3d = np.array(connect_3d, dtype=int) - 1

            if verbose:
                print("Verifying nodes and edges...")

            # nodes

            nodes = []

            if version == "4":
                for index in range(len(flags["$Nodes"])):
                    f.seek(flags["$Nodes"][index])
                    f.readline()  # line ignored (nb of nodes)

                    t = f.readline()

                    while t and f.tell() != flags["$EndNodes"][index]:
                        t = t.strip("\t\n").split()

                        if len(t) <= 1:
                            break

                        for i in range(int(t[-1])):
                            f.readline()

                        for i in range(int(t[-1])):
                            node = f.readline().strip("\t\n").split()
                            nodes.append([float(node[0]), float(node[1]), float(node[2])])

                        t = f.readline()

            if version == "2":
                for index in range(len(flags["$Nodes"])):
                    f.seek(flags["$Nodes"][index])
                    f.readline()  # line ignored (nb of nodes)

                    t = f.readline()

                    while t and f.tell() != flags["$EndNodes"][index]:
                        t = t.strip("\t\n").split()

                        if len(t) <= 1:
                            break

                        nodes.append([float(t[1]), float(t[2]), float(t[3])])

                        t = f.readline()

        nodes = np.array(nodes)

        dim = 3
        if len(connect_3d) == 0:
            if np.all(nodes[:, 0] == nodes[0, 0]):
                dim = 2
                nodes = nodes[:, 1:]
            elif np.all(nodes[:, 1] == nodes[0, 1]):
                dim = 2
                nodes = nodes[:, -1:1]
            elif np.all(nodes[:, 2] == nodes[0, 2]):
                dim = 2
                nodes = nodes[:, :2]
            else:
                dim = 2.5

        # Apparently Cimlib prefers normals looking down in 2D
        # If normals are still wrong after that, there may be foldovers in your mesh
        if dim == 2:
            if verbose:
                print("   - Checking normals")  # Actually only checking the first normal
            normal = np.cross(
                nodes[connect_2d[0][1]] - nodes[connect_2d[0][0]],
                nodes[connect_2d[0][2]] - nodes[connect_2d[0][0]],
            )
            if normal > 0:
                connect_2d = connect_2d[:, [0, 2, 1]]

        if verbose:
            print("   - Detecting edges")

        if dim == 3:
            del connect_2d

            tris1 = connect_3d[:, [0, 2, 1]]  # Order is very important !
            tris2 = connect_3d[:, [0, 1, 3]]
            tris3 = connect_3d[:, [0, 3, 2]]
            tris4 = connect_3d[:, [1, 2, 3]]

            tris = np.concatenate((tris1, tris2, tris3, tris4), axis=0)
            tris_sorted = np.sort(
                tris, axis=1
            )  # creates a copy, may be source of memory error
            tris_sorted, uniq_idx, uniq_cnt = np.unique(
                tris_sorted, axis=0, return_index=True, return_counts=True
            )
            connect_2d = tris[uniq_idx][uniq_cnt == 1]

        if dim == 2:
            lin1 = connect_2d[:, [0, 1]]  # Once again, order is very important !
            lin2 = connect_2d[:, [2, 0]]
            lin3 = connect_2d[:, [1, 2]]

            lin = np.concatenate((lin1, lin2, lin3), axis=0)
            lin_sorted = np.sort(
                lin, axis=1
            )  # creates a copy, may be source of memory error
            lin_sorted, uniq_idx, uniq_cnt = np.unique(
                lin_sorted, axis=0, return_index=True, return_counts=True
            )
            connect_1d = lin[uniq_idx][uniq_cnt == 1]

        if verbose:
            print("   - Detecting unused nodes")

        used_nodes = np.unique(
            np.concatenate((connect_3d.flatten(), connect_2d.flatten()))
        )  # sorted
        bools_keep = np.zeros(len(nodes), dtype=bool)
        bools_keep[used_nodes] = True

        if verbose:
            print("   - Deleting unused nodes and reindexing\n")

        nodes = nodes[bools_keep]
        new_indices = np.cumsum(bools_keep) - 1

        if dim == 3 or dim == 2.5:
            connect_3d = new_indices[connect_3d]
            connect_2d = new_indices[connect_2d]

        if dim == 2:
            connect_2d = new_indices[connect_2d]
            connect_1d = new_indices[connect_1d]

        nb_elems = len(connect_2d) + len(connect_3d)
        if dim == 2:
            nb_elems += len(connect_1d)
            if verbose:
                print("Nb elements 1d : " + str(len(connect_1d)))

        if verbose:
            print("Nb elements 2d : " + str(len(connect_2d)))
            print("Nb elements 3d : " + str(len(connect_3d)))
            print("Dimension : " + str(dim) + "\n")
            print("Writing .t file...")

        # Correction for mtc numbering
        connect_3d += 1
        connect_2d += 1
        if len(connect_1d) > 0:
            connect_1d += 1

        with open(output, "w") as fo:
            lig = (
                str(len(nodes))
                + " "
                + str(dim)
                + " "
                + str(nb_elems)
                + " "
                + str(dim + 1)
                + "\n"
            )
            if dim == 2.5:
                lig = str(len(nodes)) + " 3 " + str(nb_elems) + " 4\n"
            fo.write(lig)

            for node in nodes:
                fo.write("{0:.8g} {1:.8g}".format(node[0], node[1]))
                if dim == 3 or dim == 2.5:
                    fo.write(" {0:.8g}".format(node[2]))
                fo.write(" \n")

            for e in connect_3d:
                fo.write(
                    str(e[0]) + " " + str(e[1]) + " " + str(e[2]) + " " + str(e[3]) + " \n"
                )

            for e in connect_2d:
                if dim == 3 or dim == 2.5:
                    fo.write(str(e[0]) + " " + str(e[1]) + " " + str(e[2]) + " 0 \n")
                else:
                    fo.write(str(e[0]) + " " + str(e[1]) + " " + str(e[2]) + " \n")

            if dim == 2:
                for e in connect_1d:
                    fo.write(str(e[0]) + " " + str(e[1]) + " 0 \n")

        if verbose:
            print("Done.")
        return output

    def sync(self) -> str:
        geo_file = self.get_geo()
        self.get_mesh_timeout(geo_file)
        input = os.path.join(self.msh_dir, f"{self._base()}.msh")
        output = os.path.join(self.t_dir, f"{self._base()}.t")
        return self.convert_gmsh_to_mtc(input, output, False)

