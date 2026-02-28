# coupled-drl-gnn

This repo contains the programms we used during the TR Fluide.

## Structure

airfoil-dataset/ : dataset generation from training, adapted from @Théodore_Michel to generated morphed foils. Does not correspond exactly to the way PBO generates geometries.
drl-cfd/ : version of PBO working in a CFD environment, with implemented foil morphing, IDW mesh morphing, mesh repairs (not ideal)...
drl-gnn/ : PBO-GNN coupling. Same implementation as drl-cfd/.
