import os

import gmsh
import numpy as np


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
