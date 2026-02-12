// Gmsh project created on Thu Jan 08 13:23:06 2026
//+
Point(1) = {-0, 0, 0, 0.2};
//+
Point(2) = {12, 0, 0, 0.2};
//+
Point(3) = {12, 4, 0, 0.2};
//+
Point(4) = {0, 4, 0, 0.2};
//+
Line(1) = {4, 1};
//+
Line(2) = {1, 2};
//+
Line(3) = {2, 3};
//+
Line(4) = {3, 4};
//+
Curve Loop(1) = {1, 2, 3, 4};
//+
Plane Surface(1) = {1};
