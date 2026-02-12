import os
import sys
import gmsh
import multiprocessing
import subprocess
import numpy as np
import random as rd
import matplotlib.pyplot as plt

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

        # If not random : NACA0010 profile (for splines), points already ordered
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
        N = (len(actions)-1)//2 # N = 4
        camber = actions[:N+1] #camber is of length N+1
        thickness = actions[N+1:-1] #thickness is of length N-1
        rotation = actions[-1]
        origin = self.origin


        points = self.points
        new_points = np.array(points)
        if np.any(thickness < 0.0):
            raise ValueError("Thickness cannot be negative !")
        if len(camber) > origin-1 :
            raise ValueError("Camber is too large, larger than the number of control points")
        if len(camber) < origin-1 :
            raise ValueError("Camber is too short, shorter than the number of control points")
        if len(thickness) > 3 :
            raise ValueError("Thickness is too large, larger than the number of control points")
        if len(thickness) < 3 :
            raise ValueError("Thickness is too short, shorter than the number of control points")
        try:
            for i in range(N+1):
                
                c = camber[i]
                if i == 0 or i==N:
                    new_points[i+1, 1] += c
                    new_points[-(i+3), 1] += c
                else:
                    t = thickness[i-1]

                    new_points[i+1, 1] = c + t/2
                    new_points[-(i+3), 1] = c - t/2

        except ValueError as e:
            print(f"Error: {e}")

        self.points = new_points
        rotation = np.pi * rotation / 180.0  #convert degrees to radians
        self.apply_rotation(rotation)

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
        msh_output = os.path.join(self.msh_dir, f"{self._base()}.msh")
        os.makedirs(self.msh_dir, exist_ok=True)

        if not os.path.isfile(geo_input):
            raise FileNotFoundError(f".geo not found: {geo_input}")

        # Remove stale output
        try:
            if os.path.exists(msh_output):
                os.remove(msh_output)
        except Exception:
            pass

        cmd = [sys.executable, "mesh_worker.py", geo_input, msh_output]
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

            print(res.stdout, flush=True)   # lands in job.out
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

