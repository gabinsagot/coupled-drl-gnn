import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.spatial import distance_matrix
from scipy.interpolate import splprep, splev
import shutil
import os
from pathlib import Path

def order_points(points):
    """
    Orders the point of the airfoil, with increasing indices in the trigonometric direction, and starting from the rightmost point.
    """
    points = np.array(points)
    # sort by polar angle (counter-clockwise)
    angles = [np.arctan2(p[1], p[0]) for p in points]
    order = np.argsort(angles)
    sorted_points = [points[i] for i in order]

    # find the "top-right" point: maximal x, break ties by maximal y
    start_idx = max(range(len(sorted_points)), key=lambda i: (sorted_points[i][0], sorted_points[i][1]))

    # rotate so the sequence starts at the top-right point
    points = sorted_points[start_idx:] + sorted_points[:start_idx]


def compute_init_mesh_displacements(init_cp, new_cp, init_foil_origin: int):
    """
    compute_init_mesh_displacements computes the initial control points displacement according
    to init_cp and new_cp
    
    Args :
        init_cp: control points of the undeformed foil
        new_cp: control points of the deformed foil
        init_foil_origin: index corresponding to the leading edge of the undeformed foil
    """
    origin = init_foil_origin                           # Locate the leading edge of the foil in the mesh
    init_displacements = np.zeros((len(init_cp),2))

    for i in range(len(init_cp[:origin,1:])):
        init_displacements[i] = new_cp[i] - init_cp[i]
        init_displacements[-(i+2)] = new_cp[-(i+2)] - init_cp[-(i+2)]
    init_displacements[origin] = new_cp[origin] - init_cp[origin]   # Add leading edge displacement
    init_displacements[-1] = new_cp[-1] - init_cp[-1]   # Add trailing edge displacement
    #print("compute_init_displacements_mesh : Displacements : ", len(init_displacements))
    return init_displacements

def artificial_cp_linear(init_foil, new_foil, n):
    """
    Returns the artificial control points linearly interpolated between control points of init_foil 
    and their displacements according to new_foil's geometry       
    Args :
        init_foil: Foil class object, undeformed foil
        new_foil: Foil class object, deformed foil
        n: number of artificial conotrol points to put between each original control point
    """
    init_acp = init_foil.create_acp(n)
    new_acp = new_foil.create_acp(n)
    
    acp_displacements = new_acp - init_acp

    # for i in range(len(acp_displacements)):
    #     print(f"Artificial control point {init_acp[i]} displaced by {acp_displacements[i]}")

    return init_acp, acp_displacements

def compute_spline_length(tck, u_a, u_b):
    """
    Computes the arc length of a spline between two parameter values.
    
    tck = [t, c, k] object retured by Scipy splprep function
    u_a and u_b: float parameters between which spline length is computed
    Returns the length of the spline between u_a and u_b points coordinates
    """
    u_fine = np.linspace(u_a, u_b, 100)
    x_s, y_s = splev(u_fine, tck)
    pts = np.array(np.vstack((x_s, y_s)).T)

    length = np.sum(np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1)))   
    return length

def artificial_cp_spline(init_foil, new_foil, density = 100, k=2):
    """
    Creates artificial control points (acp) along the spline of the foils according to 
    a spline interpolation with the points of the given foil.
    The number of acp created between two consecutive control points depends on density
    
    init_foil: Foil class object, initial foil
    new_foil: Foil class object, deformed foil
    density: int, density of artificial control points per unit length of spline
    k: int, degree of the spline interpolation (1=linear, ...)
    """
    new_foil_points = new_foil.points
    init_foil_points = init_foil.points

    acp_displacements = []
    init_acp = []

    ### Create spline interpolating every point except last one (trailing edge not on same spline)
    points_init = np.array(init_foil_points[0:-1])
    points_new = np.array(new_foil_points[0:-1])
    x_init = points_init[:,0]
    y_init = points_init[:,1]
    x_new = points_new[:,0]
    y_new = points_new[:,1]

    (tck_init, u_init) = splprep([x_init, y_init], s=0, k=k)
    (tck_new, u_new) = splprep([x_new, y_new], s=0, k=k)

    for i in range(len(points_init)-1):
        # Compute the length of the spline segment created between consecutive points
        new_point_i = u_new[i]
        new_point_ip1 = u_new[i+1]
        length_new = compute_spline_length(tck_new, new_point_i, new_point_ip1)
    
        # Create n artificial control points along the splines segment according to and new spline length density
        n = int(np.floor(2.0*length_new*density))

        init_point_i = u_init[i]
        init_point_ip1 = u_init[i+1]
                                
        new_acp_u = np.linspace(new_point_i, new_point_ip1, (n+2))[1:-1]    
        init_acp_u = np.linspace(init_point_i, init_point_ip1, (n+2))[1:-1]

        # Get the coordinates of the init and new artificial control points
        x_acp_new, y_acp_new = splev(new_acp_u, tck_new)
        x_acp_init, y_acp_init = splev(init_acp_u, tck_init)
        # Compute points displacements
        new_acp_partial = np.array(np.vstack((x_acp_new, y_acp_new)).T)
        init_acp_partial = np.array(np.vstack((x_acp_init, y_acp_init)).T)

        acp_displacements_partial = new_acp_partial - init_acp_partial
        acp_displacements.append(acp_displacements_partial)
        init_acp.append(init_acp_partial)

        # print(f"Loop for {i+1}th spline running")
        # print("Longueur spline new : ", length_new)
        # print("Nombre de acp à ajouter spline new : ", n)
        # print(new_point_i, "   ", new_point_ip1)
        # print("Paramètres 0 à 1 de spline new : ", new_acp_u)

        # print("init_acp {i+1}th spline : ", init_acp_partial)
        # print("new_acp {i+1}th spline : ", new_acp_partial)

    #print("Artificial control points displacement :", acp_displacements)            
    return init_acp, acp_displacements

def artificial_cp_bezier(init_foil, new_foil, density):
    new_foil_points = new_foil.points
    init_foil_points = init_foil.points

    ### Create spline interpolating every point except last one (trailing edge not on same spline)
    points_init = np.array(init_foil_points[0:-1])
    points_new = np.array(new_foil_points[0:-1])
    x_init = points_init[:,0]
    y_init = points_init[:,1]
    x_new = points_new[:,0]
    y_new = points_new[:,1]

    (tck_init, u_init) = splprep([x_init, y_init], s=0.002, k=3)
    #print("u_init is : ",u_init)
    (tck_new, u_new) = splprep([x_new, y_new], s=0.002, k=3)

    for i in range(len(points_init)-1):
        # Compute the length of the spline segment created between consecutive points
        new_point_i = u_new[i]
        new_point_ip1 = u_new[i+1]
        length_new = compute_spline_length(tck_new, new_point_i, new_point_ip1)
    
        # Create n artificial control points along the splines segment according to and new spline length density
        n = int(np.floor(2.0*length_new*density))

        init_point_i = u_init[i]
        init_point_ip1 = u_init[i+1]
                                
        new_acp_u = np.linspace(new_point_i, new_point_ip1, (n+2))[1:]
        init_acp_u = np.linspace(init_point_i, init_point_ip1, (n+2))[1:]

        # Get the coordinates of the init and new artificial control points
        x_acp_new, y_acp_new = splev(new_acp_u, tck_new)
        x_acp_init, y_acp_init = splev(init_acp_u, tck_init)
        # Compute points displacements
        new_acp_partial = np.array(np.vstack((x_acp_new, y_acp_new)).T)
        init_acp_partial = np.array(np.vstack((x_acp_init, y_acp_init)).T)

        acp_displacements_partial = new_acp_partial - init_acp_partial

        if i == 0:
            acp_displacements = acp_displacements_partial
            init_acp = init_acp_partial
        else :
            acp_displacements = np.concatenate([acp_displacements,acp_displacements_partial])
            init_acp = np.concatenate([init_acp,init_acp_partial])

    # Add the last point (middle of trailling edge) at the end
    init_cp_te = init_foil_points[-1]
    new_cp_te = new_foil_points[-1]

    acp_displacements_partial = np.array(new_cp_te) - np.array(init_cp_te)
    
    acp_displacements = np.concatenate([acp_displacements,[acp_displacements_partial]])
    init_acp = np.concatenate([init_acp,[init_cp_te]])

    #print("Artificial control points displacement :", acp_displacements)            
    return init_acp, acp_displacements

def stack(cp, acp):
    """
    Returns a np.ndarray control points in the same order as on the foil's spline,
    adding control points of the trailing edge at the end
    
    Args :
        cp : np.ndarray[[x, y]] of control points
        acp : np.ndarray[[[x, y], ...], ...] of artificial control points to insert between cps
    """
    stacked_cp = []

    for i in range(len(cp)-2):
        stacked_cp.append(cp[i])
        # Insert acp[i] between cp[i] and cp[i+1]
        for point in acp[i]:
            stacked_cp.append(point)
    stacked_cp.append(cp[-2]) 
    stacked_cp.append(cp[-1]) # Append the last control points

    stacked_cp = np.array(stacked_cp)
    return stacked_cp

def compute_idw_mesh(init_naca, end_naca, ep : int, base_folder : str, path_to_results : str, interp_type = "bezier", density = 100, p = 3):
    """
    Returns the path (str) to new domain.t deformed according to IDW
    
    Deforms the original NACA0010 mesh according to any deformation between init_naca and end_naca Foil objects.

    Args :
        init_naca : Foil object of the initial foil
        end_naca : Foil object of the deformed foil
        ep : int, episode number
        refine_type : str, "spline" or "linear", type of refinement interpolation for artificial control points
        density : int, density of artificial control points per unit length of spline (only for "spline" refine_type)
        p : int, power parameter for IDW

    """

    # Extract original mesh domain data
    original_mesh = extract_points(os.path.join(base_folder, "domain/domain_naca0010_12_4.t"))

    # Select control points, including artificial ones
    init_cp = init_naca.points
    init_cp = np.array(init_cp)
    order_points(init_cp)
    new_cp = end_naca.points
    new_cp = np.array(new_cp)

    try:
        if init_cp.shape != new_cp.shape:
            raise ValueError("Initial and new control points from Foil class must have the same shape.")
        
        else :# Compute init_displacements
            init_displacements = compute_init_mesh_displacements(init_cp, new_cp, init_naca.origin)
    except ValueError as e:
        print("Error in compute_init_mesh_displacements: ", e)

    if interp_type == "spline":
        init_acp, acp_displacements = artificial_cp_spline(init_naca, end_naca, density, k=2)
        # Intercaler chaque acp entre les cp initiaux
        foil_cp = stack(init_cp, init_acp)
        control_points = get_closest_point(foil_cp, original_mesh)
        control_points = np.array(control_points) # No need to order here thanks to stack method

        displacements = stack(init_displacements, acp_displacements)

    if interp_type == "bezier":
        init_acp, acp_displacements = artificial_cp_bezier(init_naca, end_naca, density = 100)
        foil_cp = init_acp
        control_points = get_closest_point(foil_cp, original_mesh)
        control_points = np.array(control_points)

        displacements = acp_displacements
        # print("All control points displacements with Bézier interpolation : ", displacements)

    if interp_type == "linear":
        init_acp, acp_displacements = artificial_cp_linear(init_naca, end_naca, n=2)

        foil_cp = np.vstack((init_cp, init_acp))
        control_points = get_closest_point(foil_cp, original_mesh)
        control_points = np.array(control_points)

        # Order control points by trigonometric angle
        displacements = np.vstack((init_displacements, acp_displacements))

    # Move the points in mesh data
    new_mesh = idw(original_mesh, control_points, displacements, p)
    # Write new .t file at the right location
    input_t_file_path = os.path.join(base_folder, "domain/domain_naca0010_12_4.t")
    output_t_file_path = os.path.join(base_folder, path_to_results, str(ep), "cfd", "meshes", "domain.t")
    replace_points(input_t_file_path, output_t_file_path, new_mesh)

    return foil_cp + displacements


def get_closest_point(points, mesh):
    """
    Finds in mesh the closest points from those in points

    Args: 
        point: np.ndarray of shape (M,2)
        mesh: np.ndarray of shape (N, 2)
    """
    closest_points = []

    for point in points :
        distances = np.linalg.norm(mesh - point, axis=1)
        closest_index = np.argmin(distances)
        closest_point = mesh[closest_index]
        closest_points.append(closest_point)

    # print("Points les + proches : ", closest_points)

    return closest_points

def extract_points(t_file : str):
    """
    Extract the points from a .t file

    Args:
        t_file: The .t file to extract points from.

    Returns:
        A np.ndarray of shape (N, 2) of points extracted from the .t file.
    """
    points = []
    with open(t_file, 'r') as f:
        for line in f:
            l = line.split()
            points.append(np.array([float(l[0]), float(l[1])],dtype=np.float64)) if len(l) == 2 else None
        points = np.array(points, dtype=np.float64)
    return points


def idw(mesh, control_points, init_displacements, p, take_edges=True):
    """
    Args :
        mesh : np.ndarray of shape (N, 2)
        control_points : np.ndarray of shape (M, 2)
        init_displacements : np.ndarray of shape (M, 2)
        p : power parameter for inverse-distance weighting

    Returns :
        new_mesh : np.ndarray of shape (N, 2)
    """
    start_time = time.perf_counter()

    H = max(mesh, key=lambda x: x[1])[1]  # hauteur
    l = max(mesh, key=lambda x: x[0])[0]  # largeur (max x)
    null = np.array([0.0, 0.0])

    
    def is_edge(point):
        x, y = point[0], point[1]
        # use isclose to avoid floating-point equality issues
        return np.isclose(x, 0.0) or np.isclose(y, 0.0) or np.isclose(x, l) or np.isclose(y, H)
    
    if take_edges:
        for point in mesh:
            if is_edge(point):
                control_points = np.vstack((control_points, point))
                init_displacements = np.vstack((init_displacements, null))

    distances = distance_matrix(mesh, control_points, threshold=int(1e8))
    #if there is a 0 in a line, it means the point is a control_point
    #In this case, we can use the control point's displacement directly
    # compute inverse-distance weights, handling zeros so that a row with a control point
    # becomes one-hot (1 for coincident control(s), 0 for others)
    with np.errstate(divide='ignore', invalid='ignore'):
        weights = 1.0 / (distances ** p)

    zero_mask = (distances == 0)
    if zero_mask.any():
        # clear any inf/NaN produced by division by zero
        weights[zero_mask] = 0.0
        # for rows that contain one or more exact matches, set the row to the mask (1.0 where match)
        rows_with_zero = zero_mask.any(axis=1)
        weights[rows_with_zero] = zero_mask[rows_with_zero].astype(np.float64)

    #make weights sum to 1
    weights /= weights.sum(axis=1, keepdims=True)

    displacements = weights @ init_displacements
    new_mesh = mesh + displacements
    #displacements is of shape (N, 2). We need to create a list of np.ndarrays of shape (2,)
    #print(displacements)

    end_time = time.perf_counter()
    print(f"IDW computation time: {end_time - start_time:.4f} seconds")

    return new_mesh            

def replace_points(input_t_file_path : str , output_t_file_path : str, new_points):
    """
    Changes the points of a .t file and replaces them with new_points
    Args:
        t_file: Path to the .t file to modify.
        new_points: np.ndarray of shape (N, 2) containing the new points.
    Returns: 
        str: path to the new t_file
    """
    #new_file = input_t_file_path.replace('.t', '_idw.t')
    new_file = output_t_file_path

    with open(input_t_file_path, 'r') as f_in, open(new_file, 'w') as f_out:

        lines = f_in.readlines()
        point_index = 0

        for i in range(len(lines)):

            l = lines[i].split()

            if len(l) == 2:
                
                np_point = new_points[point_index]
                lines[i] = f"{np_point[0]} {np_point[1]}\n"
                point_index += 1
            
        f_out.writelines(lines)

    return new_file


#TESTS
# TEST1 = 0
# if TEST1:
#     L = []
#     for i in range(10):
#         for j in range(10):
#             L.append(np.array([float(i), float(j)]))

#     plt.scatter(*zip(*L), s=10, color='gray', alpha=0.5) #transparent
#     M = []
#     M.extend((np.array([i,3]), np.array([0, -1])) for i in range(3, 8))
#     M.extend((np.array([3,j]), np.array([-1, 0])) for j in range(3, 8))
#     M.extend((np.array([7,j]), np.array([1, 0])) for j in range(3, 8))
#     M.extend([
#         (np.array([4,7]), np.array([0, 1])),
#         (np.array([5,7]), np.array([0, 1])),
#         (np.array([6,7]), np.array([0, 1])),
#     ])

#     shift = np.array([0, 1])
#     null = np.array([0.0, 0.0])
#     transformation = M

#     # avoid exact matches with control points (prevents duplicate appends in the current idw)
#     def is_control(pt):
#         return any(np.allclose(pt, cp) for cp, _ in transformation)

#     L = [pt for pt in L if not is_control(pt)]

#     # compute transformed points using IDW (power parameter p=2 chosen as example)
#     new_points = idw(L, transformation, p=4)

#     # plot control points (original) in red
#     ctrl_orig = [cp for cp, _ in transformation]
#     if ctrl_orig:
#         plt.scatter(*zip(*ctrl_orig), color='red', label='controls')

#     # plot transformed grid
#     if new_points:
#         plt.scatter(*zip(*new_points), s=10, color='blue', label='transformed')
#     ctrl_new = [cp + disp for cp, disp in transformation]
#     if ctrl_new:
#         plt.scatter(*zip(*ctrl_new), color='orange', label='ctrl after')

#     plt.legend()
#     plt.axis('equal')
#     plt.show()
    
# TEST2 = 0
# if TEST2:
#     i_1 = 16681
#     i_2 = 200
#     L = extract_points('idw/domain.t')
#     np.set_printoptions(precision=16, suppress=False)
#     print(f"In domain.t, line {i_1} = {L[i_1]}, line {i_2} = {L[i_2]}")

#     control_points = [L[i_1], L[i_2]]
#     init_displacements = [np.array([0.0, 0.5]), np.array([0.0, 0])]

    
#     new_points = idw(L, control_points, init_displacements, p=1)


#     print(".\n.\n.\nTransformation done.")
#     #print(new_points, len(new_points))
#     print(f"{len(new_points)} points. In new_points, line {i_1} = {new_points[i_1]}, line {i_2} = {new_points[i_2]}")
#     file = replace_points('idw/domain.t', new_points)
#     print(f".\n.\n.\nFile written: {file}")
#     check = extract_points('idw/domain_idw.t')
#     print(f"In {file}, line {i_1} = {check[i_1]}, line {i_2} = {check[i_2]}")

# TEST3 = 0
# if TEST3:
#     file = 'object'
#     L = extract_points('object.t')
#     print("Points extracted.")
#     i_1 = 200
#     i_2 = 100
#     print(f"In {file}.t, line {i_1} = {L[i_1]}, line {i_2} = {L[i_2]}")

#     null = np.array([0.0, 0.0])
#     shift = np.array([0.0, 0.06])
#     control_points = []
#     displ = []

#     control_points.append(L[i_1])


#     print(f"Control points : {len(control_points)}")

#     displ.append(shift)

#     print(f"Displacements : {len(displ)}")

#     control_points = np.array(control_points)
#     displ = np.array(displ)
#     new_points = idw(L, control_points, displ, p=3.5)

#     # show full precision in console

#     print(".\n.\n.\nTransformation done.")
#     #print(new_points, len(new_points))
#     print(f"{len(new_points)} points.") 

#     print(f"In new_points, line {i_1} = {new_points[i_1]}, line {i_2} = {new_points[i_2]}")
#     file = replace_points('object.t', 'object_idw.t', new_points)
    
#     print(f".\n.\n.\nFile written: {file}")
#     check = extract_points('object_idw.t')   
#     print(f"In {file}, line {i_1} = {check[i_1]}, line {i_2} = {check[i_2]}")

    # #move idw/limace.t into cfd_ex, and rename it test.t
    # src = Path(file)  # 'file' was set earlier to 'idw/limace_idw.t'
    # dest_dir = Path('../idw/cfd_ex/meshes')
    
    # dest = dest_dir / 'test.t'

    # try:
    #     shutil.move(str(src), str(dest))
    #     print(f"Moved '{src}' -> '{dest}'")
    # except Exception as e:
    #     print(f"Failed to move '{src}' -> '{dest}': {e}")


# TEST4=0
# if TEST4:
#     """
#     Takes a meshed airfoil with control points, moves a point to a corresponding location 
#     """
