import numpy as np
import time
import shutil
from scipy.spatial import distance_matrix
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import os

def vectorized_triangle_intersection_factor(Ti_batch, Tj_batch):
    """
    Vectorized penetration depth computation for batches of triangle pairs.
    Ti_batch: shape (M, 3, 2)
    Tj_batch: shape (M, 3, 2)
    Returns: shape (M,) - penetration depth for each pair
    """
    M = Ti_batch.shape[0]
    penetration_depths = np.full(M, np.inf)
    
    # Process all 6 edges (3 from each triangle) for all pairs at once
    for tri_batch, name in [(Ti_batch, 'A'), (Tj_batch, 'B')]:
        for edge_idx in range(3):
            p1 = tri_batch[:, edge_idx]                    # (M, 2)
            p2 = tri_batch[:, (edge_idx + 1) % 3]          # (M, 2)
            edge = p2 - p1                                  # (M, 2)
            
            edge_norm = np.linalg.norm(edge, axis=1, keepdims=True)  # (M, 1)
            edge_norm = np.where(edge_norm < 1e-15, 1.0, edge_norm)  # avoid division by zero
            
            # Normal to edge: rotate 90 degrees
            axis = np.stack([edge[:, 1], -edge[:, 0]], axis=1) / edge_norm  # (M, 2)
            
            # Project both triangle batches onto this axis
            minA, maxA = vectorized_project_triangle(Ti_batch, axis)  # (M,), (M,)
            minB, maxB = vectorized_project_triangle(Tj_batch, axis)  # (M,), (M,)
            
            # Check separation
            separated = (maxA < minB) | (maxB < minA)
            
            # Compute overlap
            overlap = np.minimum(maxA, maxB) - np.maximum(minA, minB)
            
            # Update penetration depths (only for non-separated pairs)
            overlap = np.where(separated, 0.0, overlap)
            penetration_depths = np.minimum(penetration_depths, overlap)
    
    # If any pair has 0 penetration (separated), set to 0
    penetration_depths = np.where(penetration_depths == np.inf, 0.0, penetration_depths)
    return penetration_depths


def vectorized_project_triangle(triangles, axes):
    """
    Project batch of triangles onto batch of axes.
    triangles: shape (M, 3, 2)
    axes: shape (M, 2)
    Returns: (min_proj, max_proj) each of shape (M,)
    """
    # Compute projections for all 3 vertices
    # triangles[:, i] has shape (M, 2)
    # axes has shape (M, 2)
    # dot product along last dimension
    
    proj0 = np.sum(triangles[:, 0] * axes, axis=1)  # (M,)
    proj1 = np.sum(triangles[:, 1] * axes, axis=1)  # (M,)
    proj2 = np.sum(triangles[:, 2] * axes, axis=1)  # (M,)
    
    projs = np.stack([proj0, proj1, proj2], axis=1)  # (M, 3)
    
    min_proj = np.min(projs, axis=1)  # (M,)
    max_proj = np.max(projs, axis=1)  # (M,)
    
    return min_proj, max_proj


def vectorized_segment_distances(Ti_batch, Tj_batch):
    """
    Compute minimum distance between all edge pairs for batches of triangles.
    Ti_batch: shape (M, 3, 2)
    Tj_batch: shape (M, 3, 2)
    Returns: shape (M,) - minimum distance for each triangle pair
    """
    M = Ti_batch.shape[0]
    min_dists = np.full(M, 1e10)
    
    # 3 edges in Ti × 3 edges in Tj = 9 edge pairs to check
    for i in range(3):
        p1 = Ti_batch[:, i]                    # (M, 2)
        p2 = Ti_batch[:, (i + 1) % 3]          # (M, 2)
        
        for j in range(3):
            q1 = Tj_batch[:, j]                # (M, 2)
            q2 = Tj_batch[:, (j + 1) % 3]      # (M, 2)
            
            dists = vectorized_segment_segment_distance(p1, p2, q1, q2)  # (M,)
            min_dists = np.minimum(min_dists, dists)
    
    return min_dists


def vectorized_segment_segment_distance(p1, p2, q1, q2):
    """
    Vectorized segment-segment distance.
    All inputs: shape (M, 2)
    Returns: shape (M,)
    """
    u = p2 - p1  # (M, 2)
    v = q2 - q1  # (M, 2)
    w = p1 - q1  # (M, 2)
    
    a = np.sum(u * u, axis=1)  # (M,)
    b = np.sum(u * v, axis=1)  # (M,)
    c = np.sum(v * v, axis=1)  # (M,)
    d = np.sum(u * w, axis=1)  # (M,)
    e = np.sum(v * w, axis=1)  # (M,)
    
    denom = a * c - b * b  # (M,)
    
    # Handle parallel segments
    parallel = denom < 1e-14

    M = p1.shape[0]
    s = np.zeros(M)
    t = np.zeros(M)
    
    # Non-parallel case
    non_parallel = ~parallel
    s[non_parallel] = (b[non_parallel] * e[non_parallel] - c[non_parallel] * d[non_parallel]) / denom[non_parallel]
    t[non_parallel] = (a[non_parallel] * e[non_parallel] - b[non_parallel] * d[non_parallel]) / denom[non_parallel]
    
    s = np.clip(s, 0, 1)
    t = np.clip(t, 0, 1)
    
    closest_p = p1 + s[:, None] * u  # (M, 2)
    closest_q = q1 + t[:, None] * v  # (M, 2)
    
    dist = np.linalg.norm(closest_p - closest_q, axis=1)  # (M,)
    
    # For parallel segments, compute endpoint distances
    if parallel.any():
        endpoints_dist = np.stack([
            np.linalg.norm(p1 - q1, axis=1),
            np.linalg.norm(p1 - q2, axis=1),
            np.linalg.norm(p2 - q1, axis=1),
            np.linalg.norm(p2 - q2, axis=1)
        ], axis=1).min(axis=1)  # (M,)
        
        dist[parallel] = endpoints_dist[parallel]
    
    return dist


def vectorized_signed_clearance(Ti_batch, Tj_batch):
    """
    Vectorized signed clearance for batches of triangle pairs.
    Ti_batch: shape (M, 3, 2)
    Tj_batch: shape (M, 3, 2)
    Returns: shape (M,)
    """
    penetration_depths = vectorized_triangle_intersection_factor(Ti_batch, Tj_batch)  # (M,)
    
    # Mask for intersecting pairs
    intersecting = penetration_depths > 0
    
    # For intersecting pairs, return negative penetration
    result = -penetration_depths
    
    # For non-intersecting pairs, compute minimum distance
    if (~intersecting).any():
        distances = vectorized_segment_distances(Ti_batch[~intersecting], Tj_batch[~intersecting])
        result[~intersecting] = distances
    
    return result

def orient2d(a, b, c):
    """Signed area (2x the area). <0 means inverted orientation."""
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def segment_segment_distance(p1, p2, q1, q2):
    """
    Computes the minimal distance between two line segments in 2D.
    Returns positive distance.
    """
    # Parameterize segments as p(s) = p1 + s*(p2-p1), q(t) = q1 + t*(q2-q1)

    u = p2 - p1
    v = q2 - q1
    w = p1 - q1

    a = np.dot(u,u)
    b = np.dot(u,v)
    c = np.dot(v,v)
    d = np.dot(u,w)
    e = np.dot(v,w)
    denom = a*c - b*b

    if denom < 1e-14:
        # Segments are nearly parallel -> just check endpoints → segment distances
        return min(
            np.linalg.norm(p1-q1),
            np.linalg.norm(p1-q2),
            np.linalg.norm(p2-q1),
            np.linalg.norm(p2-q2),
        )

    s = (b*e - c*d) / denom
    t = (a*e - b*d) / denom
    s = np.clip(s, 0, 1)
    t = np.clip(t, 0, 1)

    closest_p = p1 + s*u
    closest_q = q1 + t*v
    return np.linalg.norm(closest_p - closest_q)


def segments_intersect(p1, p2, q1, q2):
    """Check segment intersection using orientations."""
    o1 = np.sign(orient2d(p1, p2, q1))
    o2 = np.sign(orient2d(p1, p2, q2))
    o3 = np.sign(orient2d(q1, q2, p1))
    o4 = np.sign(orient2d(q1, q2, p2))

    return (o1 != o2) and (o3 != o4)

def triangle_intersect(tri_1, tri_2):
    """Check triangle-triangle intersection in 2D."""
    # p, q are 3x2 arrays of points
    edges_p = [(0,1), (1,2), (2,0)]
    edges_q = [(0,1), (1,2), (2,0)]

    # Check all edge-edge intersections
    for (i,j) in edges_p:
        for (k,l) in edges_q:
            if segments_intersect(tri_1[i], tri_1[j], tri_2[k], tri_2[l]):
                return True
    return False

def project_triangle(triangle, axis):
    """
    Args :
        - Triangle : ndarray (3,2)
        - Axis : ndarray (2), 2d vector
    Returns :
        - Tuple (min, max), projection of the triangle points onto axis.
    """
    min = np.dot(axis, triangle[0] - triangle[1])
    max = min
    for i in range(1,3):
        p = np.dot(axis, triangle[i] - triangle[(i+1) % 3])
        if p < min:
            min = p
        elif p > max:
            max = p

    return (min, max)

def triangle_intersection_factor(Ti, Tj, intersects):
    """
    Return a signed separation distance between triangles Ti and Tj.
      - if intersects=True, returns the penetration_depth computed with Separate Axis Theorem
      - else, returns 0.0
    """
    if intersects:
        penetration_depth = float('inf')
        penetration_axis = None

        # For edges of both triangles
        for triangle in (Ti, Tj):
            for i in range(3):
                p1 = triangle[i]
                p2 = triangle[(i + 1) % 3]
                edge = p2-p1

                # Get the normal to the edge
                axis = np.array([-edge[0], edge[1]])/np.linalg.norm(edge)

                # Project both triangles onto axis
                minA, maxA = project_triangle(Ti, axis)
                minB, maxB = project_triangle(Tj, axis)

                # Check for separation, just in case
                if maxA < minB or maxB < minA: return 0.0

                # If intersection : compute overlap
                overlap = min(maxA, maxB) - max(minA, minB)
                if overlap < penetration_depth:
                    penetration_depth = overlap
                    penetration_axis = axis
        return penetration_depth
    return 0.0

def triangle_intersection_factor_nointer(Ti, Tj):
    """
    Return a signed separation distance between triangles Ti and Tj.
      - if intersects=True, returns the penetration_depth computed with Separate Axis Theorem
      - else, returns 0.0
    """
    penetration_depth = float('inf')
    penetration_axis = None

    # For edges of both triangles
    for triangle in (Ti, Tj):
        for i in range(3):
            p1 = triangle[i]
            p2 = triangle[(i + 1) % 3]
            edge = p2-p1

            # Get the normal to the edge
            
            norm_edge = np.linalg.norm(edge)
            if norm_edge < 1e-15:
                continue
            axis = np.array([edge[1], -edge[0]]) / norm_edge

            # Project both triangles onto axis
            minA, maxA = project_triangle(Ti, axis)
            minB, maxB = project_triangle(Tj, axis)

            # If separated
            if maxA < minB or maxB < minA: return 0.0

            # If intersection : compute overlap
            overlap = min(maxA, maxB) - max(minA, minB)
            if overlap < penetration_depth:
                penetration_depth = overlap
                penetration_axis = axis
    return penetration_depth

def signed_clearance(Ti, Tj, intersects = None):
    """Returns distance bewteen triangles if no intersection, otherwise penetration depth"""
    if intersects != None:
        penetration_depth = triangle_intersection_factor(Ti, Tj, intersects)
    else :
        penetration_depth = triangle_intersection_factor_nointer(Ti, Tj)

    if penetration_depth > 0.0:
        return -penetration_depth
    
    # Else, compute the minimum distance bewteen triangles
    min_dist = 1e3
    for i in range(3):
        for j in range(3):
            dist = segment_segment_distance(Ti[i], Ti[(i+1)%3], Tj[j], Tj[(j+1)%3])
            if dist < min_dist :
                min_dist = dist     

    return min_dist

def select_triangles(points, triangles, roi):
    """
    Finds traingles fully located in the region of interest
    Returns 
    - Array containing the indexes ot the triangles located in the ROI, as given in triangles array
    """
    xmin, ymin, xmax, ymax = roi
    pts_tri = points[triangles]           # (M,3,2)
    inside = ((pts_tri[:,:,0] >= xmin) &
            (pts_tri[:,:,0] <= xmax) &
            (pts_tri[:,:,1] >= ymin) &
            (pts_tri[:,:,1] <= ymax))

    mask = inside.all(axis=1)   # Keep triangles where all 3 vertices are inside
    return np.where(mask)[0]

def detect_inverted(points, tris_idx, tris):
    """
    Returns the indexes of triangles that are inverted
    """
    # Get all triangles points as [[[x1, y1], [x2, y2], [x3, y3]], [...], ...]
    P = points[tris[tris_idx]]      # (K,3,2)
    # Signed area = cross product of edges
    v0 = P[:,1] - P[:,0]
    v1 = P[:,2] - P[:,0]
    area = v0[:,0]*v1[:,1] - v0[:,1]*v1[:,0]

    inverted_mask = area <= 0
    return tris_idx[inverted_mask]

def triangles_bboxes(points, tris_idx, tris):
    """
    Computes bouding boxes for each triangle in the tris_idx array
    Returns 
        Array containing the bounding boxes parameters for each traingle
    """
    P = points[tris[tris_idx]]      # (K,3,2)
    xmin = P[:,:,0].min(axis=1)
    ymin = P[:,:,1].min(axis=1)
    xmax = P[:,:,0].max(axis=1)
    ymax = P[:,:,1].max(axis=1)
    return np.stack([xmin, ymin, xmax, ymax], axis=1)  # (K,4)

def candidate_intersection_pairs(bboxes):
    """
    Bounding-box intersection test.
    Returns 
        - Array of candidate index pairs (i,j).
    """
    xmin = bboxes[:,0][:,None]
    ymin = bboxes[:,1][:,None]
    xmax = bboxes[:,2][:,None]
    ymax = bboxes[:,3][:,None]

    # NxN matrix of bounding boxes intersections
    inter = ~((xmax < xmin.T) |
            (xmin > xmax.T) |
            (ymax < ymin.T) |
            (ymin > ymax.T))

    # Keep upper triangular part only to avoid duplication
    iu, ju = np.triu_indices_from(inter, k=1)
    mask = inter[iu, ju]
    return np.vstack([iu[mask], ju[mask]]).T

def triangle_intersections(points, tris_idx, tris, candidate_pairs):
    """
    Vectorized triangle-triangle intersection detection.
    Only tests candidate pairs.
    Returns
        - Triangle indices from original tris_idx that need to be adjusted
    """
    if len(candidate_pairs) == 0:
        return []

    P = points[tris[tris_idx]]        # (K,3,2)
    A = P[candidate_pairs[:,0]]       # (C,3,2)
    B = P[candidate_pairs[:,1]]       # (C,3,2)

    # Edges (3 per triangle)
    A_edges = np.stack([A[:,[0,1]], A[:,[1,2]], A[:,[2,0]]], axis=1)  # (C,3,2,2)
    B_edges = np.stack([B[:,[0,1]], B[:,[1,2]], B[:,[2,0]]], axis=1)  # (C,3,2,2)

    # Reshape for broadcasting segment vs segment
    Ae = A_edges.reshape(-1,2,2)  # (3C,2,2)
    Be = B_edges.reshape(-1,2,2)  # (3C,2,2)

    # All pairwise Ae[i] with Be[i]
    p1 = Ae[:,0]
    p2 = Ae[:,1]
    q1 = Be[:,0]
    q2 = Be[:,1]

    # Vectorized orientation
    def orient(p, q, r):
        return (q[:,0]-p[:,0])*(r[:,1]-p[:,1]) - (q[:,1]-p[:,1])*(r[:,0]-p[:,0])

    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)

    seg_inter = (o1*o2 < 0) & (o3*o4 < 0)    # segment intersection
    # Collapse back to triangle pairs: reshape (3C) → (C,3)
    seg_inter = seg_inter.reshape(-1,3)
    tri_inter_mask = seg_inter.any(axis=1)   # (C,)

    return tris_idx[candidate_pairs[tri_inter_mask]].flatten()

def detect_tangled(points, tris, roi):
    tri_idx_roi = select_triangles(points, tris, roi)       # Select triangles inside ROI
    inverted = detect_inverted(points, tri_idx_roi, tris)   # Detect inverted triangles in ROI

    bboxes = triangles_bboxes(points, tri_idx_roi, tris)    # Get the bouding boxes for all triangles in ROI
    candidates_inter_pairs = candidate_intersection_pairs(bboxes)            # Get candidates for intersection (overlapping bounding boxes)
    intersecting = triangle_intersections(points, tri_idx_roi, tris, candidates_inter_pairs)    # Intersection test

    bad = np.unique(np.concatenate([inverted, intersecting]))  # Combine to get triangles that need action

    return bad, inverted, intersecting

def untangle_patch(points, tris, bad_triangles, roi, fixed_mask=None, first_solver=True):
    """
    Move only nodes belonging to the patch.
    fixed_mask: boolean array of nodes that must not move (foil, domain boundary...)
    """
    if len(bad_triangles) == 0:
        return points

    bad_nodes = sorted(set(tris[bad_triangles].flatten()))
    if fixed_mask is None:
        fixed_mask = np.zeros(len(points), dtype=bool)

    movable = [n for n in bad_nodes if not fixed_mask[n]]
    if len(movable) == 0:
        print("No movable nodes in patch, cannot further repair mesh.")
        return points

    # Initial vector
    x0 = points[movable].flatten()

    # Objective: keep nodes close to original positions
    def objective_1st_step(x):
        new_pts = points.copy()
        new_pts[movable] = x.reshape(-1,2)
        return np.sum((new_pts[movable] - points[movable])**6)
    

    tri_idx_roi = select_triangles(points, tris, roi)          # global triangle ids in ROI
    bboxes0      = triangles_bboxes(points, tri_idx_roi, tris) # bboxes in same order
    candidate_pairs = candidate_intersection_pairs(bboxes0)    # (li, lj), local into tri_idx_roi
    
    pairs_i = candidate_pairs[:,0]    # shape = (M,)
    pairs_j = candidate_pairs[:,1]    # shape = (M,)
    
    def objective_2nd_step(x):
        new_pts = points.copy()
        new_pts[movable] = x.reshape(-1,2)
        
        P_new = new_pts[tris[tri_idx_roi]]
        Pi = P_new[pairs_i]
        Pj = P_new[pairs_j]
        
        penalties = -vectorized_signed_clearance(Pi, Pj)
        intersection_penalty = np.sum((1e3*np.maximum(0, penalties))**2)
        
        # Regularizer
        dx = x - x0
        reg = 1e-10 * np.dot(dx, dx)
        
        objective_val = intersection_penalty + reg
        # print(f"Objective: {objective_val:.6f}, Intersections: {np.sum(penalties > 1e-12)}, Max penetration: {np.max(penalties):.6f}")
    
        return objective_val
    

    # Constraints: positivity of all triangle areas in the patch
    def make_area_constraints(ti):
        def cons(x):
            new_pts = points.copy()
            new_pts[movable] = x.reshape(-1,2)
            a,b,c = new_pts[tris[ti]]
            return orient2d(a,b,c)   # must be positive
        return {"type":"ineq", "fun":cons}

    # Optimizers
    if first_solver:
        # If lots of triangles to repair : use signed area constraint (faster)
        constraints = []

        # Collect triangles adjacent to patch nodes
        patch_tris = set()
        for ti in range(len(tris)):
            if len(set(tris[ti]).intersection(bad_nodes)) > 0:
                patch_tris.add(ti)
        for ti in patch_tris:
            constraints.append(make_area_constraints(ti))
        # Optimization on points displacement with signed area constraints 
        result = minimize(
            objective_1st_step, x0,
            constraints=constraints,
            method='SLSQP',
            options={'maxiter': 10, 'ftol': 1e-12, 'disp': False})
        
    else :
        # It few triangles to repair: optimize directly the sum of penetrations depth squared 
        result = minimize(
            objective_2nd_step, x0,
            method='L-BFGS-B',
            options={'maxiter': 5, 'ftol': 1e-10, 'gtol': 1e-12, 'disp': False})

    # if not result.success:
    #     print("Warning: optimizer did not fully converge:", result.message)

    # Update points
    new_pts = points.copy()
    new_pts[movable] = result.x.reshape(-1,2)
    return new_pts

def repair_mesh(points, tris, ep, roi = (0, 0, 12, 4), fixed_mask=None, iterations=3):
    pts = points.copy()

    bad_tri, inv_tri, inter_tri = detect_tangled(pts, tris, roi)
    #print(f"Iter 0: inverted={len(inv_tri)}, intersecting={len(inter_tri)}", flush=True)

    # At each iteration, detect intersecting triangles and run optimization to resolve intersections
    for it in range(iterations):
        if len(bad_tri) == 0:
            break

        old_pts = pts
        if it == 0:
            pts = untangle_patch(pts, tris, bad_tri, roi, fixed_mask=fixed_mask, first_solver=True)
        else :
            pts = untangle_patch(pts, tris, bad_tri, roi, fixed_mask=fixed_mask, first_solver=False)
        
        bad_tri, inv_tri, inter_tri = detect_tangled(pts, tris, roi)
        
        if np.linalg.norm(pts-old_pts) < 1e-8 and it != iterations-1:
            print("Solver could not move points anymore. Exiting mesh repair function.")
            break

    print(f"End of mesh repair {ep}: inverted = {len(inv_tri)}, intersecting = {len(inter_tri)}", flush=True)


    return pts

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

    (tck_init, u_init) = splprep([x_init, y_init], s=0, k=3)
    #print("u_init is : ",u_init)
    (tck_new, u_new) = splprep([x_new, y_new], s=0, k=3)

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

def compute_foil_points(init_naca, end_naca, ep : int, interp_type = "bezier", density = 100):
    # Select control points, including artificial ones
    init_cp = init_naca.points
    init_cp = np.array(init_cp)
    new_cp = end_naca.points
    new_cp = np.array(new_cp)

    if init_cp.shape != new_cp.shape:
        raise ValueError("Initial and new control points from Foil class must have the same shape.")

    if interp_type == "bezier":
        init_cp, displacements = artificial_cp_bezier(init_naca, end_naca, density = density)
        init_foil_cp = init_cp

    return init_foil_cp, init_foil_cp + displacements #type: ignore


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

def _parse_header(line):
    """
    Parse the first declaration line: n_points, dim, n_tris, nodes_per_tri.
    Returns (n_points, dim, n_tris, nodes_per_tri) as ints.
    """
    parts = line.strip().split()
    if len(parts) < 4:
        raise ValueError(f"Header must have at least 4 integers. Got: {line!r}")
    n_points, dim, n_tris, nodes_per_tri = map(int, parts[:4])
    if dim != 2:
        raise ValueError(f"Only 2D meshes supported (got {dim}D).")
    if nodes_per_tri != 3:
        raise ValueError(f"Expecting triangles with 3 nodes (got {nodes_per_tri}).")
    return n_points, dim, n_tris, nodes_per_tri

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

def _try_read_int_triplet(line):
    """
    Attempts to parse a line as exactly 3 integers. 
    Returns a tuple of 3 ints if successful, else None.
    """
    parts = line.strip().split()
    if len(parts) != 3:
        return None
    try:
        a, b, c = map(int, parts)
        return (a, b, c)
    except ValueError:
        return None

def extract_triangles(t_file : str):
    """
    Parse the triangle connectivity from the full mesh file.

    Parameters
    ----------
    path : str or Path
        Path to the mesh file.

    Returns
        tris : (n_tris, 3) ndarray of int containing points id for each triangle
            Zero-based triangle connectivity.

    Notes
    -----
    - The function reads the header to get counts, skips `n_points` coordinate
      lines, then collects exactly `n_tris` triplets.
    - It automatically converts to 0-based node indices if the file is 1-based.
    """

    with open(t_file, "r", encoding="utf-8") as f:
        # Header
        header_line = f.readline()
        if not header_line:
            raise ValueError("Empty file; cannot read header.")
        n_points, dim, n_tris, nodes_per_tri = _parse_header(header_line)

        # Skip the point lines (already parsed on your side, but we skip here)
        for _ in range(n_points):
            skipped = f.readline()
            if not skipped:
                raise ValueError(f"Unexpected EOF while skipping {n_points} point lines.")
        # Collect exactly n_tris triplets
        tris_raw = []
        for line in f:
            t = _try_read_int_triplet(line)
            if t is None:
                continue # Ignore lines that are not exactly 3 integers
            if t[-1] == 0:
                break
            tris_raw.append(t)
        triangles = np.array(tris_raw, dtype=int)

        # Detect indexing (0-based or 1-based) and normalize to 0-based
        min_idx = triangles.min()
        max_idx = triangles.max()
        if min_idx == 0:
            # 0-based; validate range
            if max_idx >= n_points:
                raise ValueError(
                    f"Triangle index {max_idx} out of range for 0-based indexing "
                    f"(n_points={n_points})."
                )
        else:
            # Assume 1-based
            if not (1 <= min_idx <= n_points and 1 <= max_idx <= n_points):
                raise ValueError("Triangle indices look inconsistent with 1-based scheme. "
                                f"Min={min_idx}, Max={max_idx}, n_points={n_points}.")
            triangles -= 1  # convert to 0-based

        return triangles


### Computing mesh displacements ###

def compute_idw_mesh(init_foil_cp, new_foil_cp, ep : int, original_domain_path : str, path_to_results : str, mesh_pts = np.array([]), density = 150, a=4, b=2, epsilon=1e-15, n=1, save_t_file=False, repair_msh=False):
    """
    Returns position of new control points of the mesh, to create new foil's geometry from these points
    
    Deforms the original NACA0010 mesh according to any deformation between init_naca and end_naca Foil objects.

    Args :
        init_naca : Foil object of the initial foil
        end_naca : Foil object of the deformed foil
        ep : int, episode number
        refine_type : str, "spline" or "linear", type of refinement interpolation for artificial control points
        density : int, density of artificial control points per unit length of spline (only for "spline" refine_type)
        p : int, power parameter for IDW

    """
    # Compute control points displacemnt from initial to new foil
    if init_foil_cp.shape != new_foil_cp.shape:
        raise ValueError("Initial and new control points from Foil class must have the same shape.")   
    displacements = new_foil_cp - init_foil_cp

    # Extract original mesh domain data
    mesh_path = os.path.join(original_domain_path)
    if mesh_pts.shape == np.array([]).shape:
        original_mesh = extract_points(mesh_path)
        triangles = extract_triangles(mesh_path)
    else:
        original_mesh = mesh_pts
        triangles = extract_triangles(mesh_path)

    mesh_control_points = np.array(get_closest_point(init_foil_cp, original_mesh))

    # Move the points in mesh data

    new_mesh_pts = multistep_idw(original_mesh, mesh_control_points, displacements, n=n, a=a, b=b, epsilon=epsilon)

    # Write new .t file at the right location
    if repair_msh:
        new_mesh_pts = repair_mesh(new_mesh_pts, triangles, ep, roi=(2.2, 1.3, 3.8, 2.7), iterations=2)
        # print("Mesh repaired.", flush=True)

    if save_t_file:
        input_t_file_path = os.path.join(original_domain_path)
        meshes_dir = os.path.join(path_to_results, "meshes")
        os.makedirs(meshes_dir, exist_ok=True)
        output_t_file_path = os.path.join(path_to_results, "meshes/domain.t")
        shutil.copyfile(input_t_file_path, output_t_file_path)

        replace_points(input_t_file_path, output_t_file_path, new_mesh_pts)
        # print(f"Domain .t saved {ep}.", flush=True)
    return new_mesh_pts, triangles


### Proper IDW functions ###

def multistep_idw(mesh, control_points, init_displacements, n, a=3, b=5, epsilon = 1e-15, take_edges=True):
    """
    Splits the init_displacements into n fractions, and performs n successive IDWs
    This allows for a more gradual transformation of the mesh, and might avoid compenetration. 

    Args: 
        mesh : np.ndarray of shape (N, 2)
        control_points : np.ndarray of shape (M, 2)
        init_displacements : np.ndarray of shape (M, 2)
        n : int, number of steps for the multistep IDW
        p : power parameter for inverse-distance weighting
    """
    partial_displacements = init_displacements / n
    for _ in range(n):
        # Perform IDW for this step
        mesh = idw(mesh, control_points, partial_displacements, a=a, b=b, epsilon = epsilon, alpha=0.5, L=1, take_edges=take_edges)
        control_points += partial_displacements
    return mesh

def idw(mesh, control_points, init_displacements, a=3, b=5, epsilon = 1e-15, alpha=0.5, L=1, take_edges=True):
    """
    Args :
        mesh : np.ndarray of shape (N, 2)
        control_points : np.ndarray of shape (M, 2)
        init_displacements : np.ndarray of shape (M, 2)
        p : power parameter for inverse-distance weighting

    Returns :
        new_mesh : np.ndarray of shape (N, 2)
    """
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
        weights = L**a / (distances**a + epsilon) + (alpha*L)**b / ((distances)**b + epsilon)

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
    # print("New .t created at ", output_t_file_path)
    return new_file