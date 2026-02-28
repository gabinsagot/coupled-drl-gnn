# coupled-drl-gnn

This repo contains the programms we used during the TR Fluide.

## Structure

airfoil-dataset/ : dataset generation from training, adapted from @Théodore_Michel to generated morphed foils. Does not correspond exactly to the way PBO generates geometries.
drl-cfd/ : version of PBO working in a CFD environment, with implemented foil morphing, IDW mesh morphing, mesh repairs (not ideal)...
drl-gnn/ : PBO-GNN coupling. Same implementation as drl-cfd/.


## What can be easily tweaked to conduct further studies ?

- Actions
The way actions are implemented can be easily changed. Actions are stored in an
array of floats called ``actions``. Its size is declared at the top of airfoil.py.
Actions are converted by ``convert_actions_to_physical_scale()`` before affecting the geometry.

The geometry is affected by the ``apply_xxxxxxxx``, declared in graphdrl/environment/geometry.py and ``Foil()`` class.

- Geometry creation
The shape of the foil is determined by the type of curves defined by the control points.
The type of curve can be adjusted within a range of Bézier curve types by tweaking the
paramter ``s``, set in ``artificial_cp_bezier()``, located graphdrl/environment/idw.py. Seting this parameter
to ``0`` leads to Spline exactly interpolating the control points moved by PBO. The further
from 0, the smoothest the shape (closer to Bézier).

- IDW mesh morphing
Morphing can be performed in several steps by setting the parameter ``n`` when calling
``compute_idw_mesh()``function in airfoil.py, within ``create_geometry()``.
If performing morphing on a BLM mesh, ou might want to repair the mesh (intersecting edges
caused by anitropism). This can be done by setting ``repair_msh`` to True in the same function.

- IDW base mesh to morph
Put the mesh to morph in environment_config/idw_base_mesh/. Rename it with the proper expected name.
