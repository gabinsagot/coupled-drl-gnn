# Convertit le fichier .txt d'entrée en un fichier .geo pour Gmsh
"""Par exemple, entrée :
1.00000            0.7700000
"""
import os
import numpy as np
from shutil import copy

def convert_txt_to_geo(fichier_entree, h, type="line"):

    os.makedirs("geo", exist_ok=True)
    
    base_name = os.path.basename(fichier_entree)
    fichier_sortie = os.path.join("geo", base_name.replace(".txt", ".geo"))

    with open(fichier_entree, "r") as f_in, open(fichier_sortie, "w") as f_out:
        f_out.write("// Fichier .geo généré à partir de {}\n".format(fichier_entree))
        f_out.write("h = {};\n".format(h))
        point_id = 1
        zero = 0

        for line in f_in:
            x_str, y_str = line.split()
            x, y = float(x_str), float(y_str)
            if (x, y) == (0.0, 0.0):
                zero = point_id
            f_out.write("Point({}) = {{{}, {}, 0, h}};\n".format(point_id, x, y))
            point_id += 1

        if type == "line":
            # On relie les points par des lignes
            for i in range(1, point_id - 1):
                f_out.write("Line({}) = {{{}, {}}};\n".format(i, i, i + 1))
            # On crée une ligne fermée
            f_out.write("Line({}) = {{{}, 1}};\n".format(point_id - 1, point_id - 1))

        if type == "bezier":
            L_attaque = [zero - 1, zero, zero + 1]
            L_up = [i for i in range(1, zero)]
            L_down = [i for i in range(zero + 1, point_id)] + [1]

            L = [i for i in range(1, point_id)]

            f_out.write("BSpline(1) = " + str((L)).replace("[", "{").replace("]", "}") + ";\n")
            f_out.write("Line(2) = {{1, {}}};\n".format(point_id-1))

            """
            f_out.write("Bezier(1) = " + str((L_attaque)).replace("[", "{").replace("]", "}") + ";\n")
            f_out.write("Bezier(2) = " + str(L_up).replace("[", "{").replace("]", "}") + ";\n")
            f_out.write("Bezier(3) = " + str(L_down).replace("[", "{").replace("]", "}") + ";\n")

            f_out.write("Curve Loop(1) = {2, 1, 3};\n")
            """
        f_out.write("Curve Loop(1) = {-1, 2};\n")
        f_out.write("Plane Surface(1) = {1};\n")

    print("Fichier .geo généré : {}".format(fichier_sortie))
    return fichier_sortie


def rotate_and_mesh(fichier_entree, angle):
    os.makedirs("meshes", exist_ok=True)
    
    base_name = os.path.basename(fichier_entree).replace(".geo", "")
    fichier_sortie = os.path.join("geo", base_name + "_" + str(angle) + ".geo")
    
    angle_rad = angle * np.pi / 180
    copy(fichier_entree, fichier_sortie)
    
    msh_output = os.path.join("meshes", base_name + "_" + str(angle) + ".msh")
    
    with open(fichier_sortie, "a") as f_out:
        f_out.write("Rotate " + "{" + "{" + "0, 0, 1}, {0.5, 0, 0}, " + str(angle_rad) + "}  {\n Surface{1};\n}\n")
        f_out.write("Mesh 2;\n" + "Save \"{}\";".format(msh_output))
    
    os.system("gmsh " + fichier_sortie + " -2 -o " + msh_output)
    print("Fichier .msh généré : {}".format(msh_output))

'''
GMSH4MTC

Utilitaire de conversion des maillages gmsh (.msh) de format 2 et 4
2D et 3D en format mtc (.t).
Fonctionne aussi desormais avec les fichiers stl.

Fonctionne pour les maillages surfaciques (2D et 3D) et volumiques.
Uniquement pour les éléments triangulaires, tetrahedriques.

Peut être drag&drop sur le fichier à transformer (Windows).

maxime.renault@minesparis.psl.eu
06/2022

Inspiré du travail de Tommy Carozzani en 2010.
'''

import sys
import os
import numpy as np

def gmsh4mtc_single_step(input, output):

    print("#########################################################")
    print("# GMSH VERSION 2 to 4 MTC CONVERTER - CFL - MINES PARIS #")
    print("#########################################################")
    print("No mtc.exe version")
    print("")

    print("Initialisation...\n")

    with open(input) as f:
        f.readline()
        version = f.readline().split()[0]
        if len(version) > 1:
            version = version.split('.')[0]
        if version != '4' and version != '2':
            print("This version of gmsh isn't supported")
            input("Press enter to close...")
            sys.exit()

        flags = {"$Nodes": [],
                "$EndNodes": [],
                "$Elements": [],
                "$EndElements": []}

        connect_3d = []
        connect_2d = []
        connect_1d = []

        ###########

        print("Getting position flags...\n")

        t = f.readline()

        while t:
            t = t.strip("\t\n")
            if t.startswith("$"):
                for i in range(len(list(flags.keys()))):
                    if(t == list(flags.keys())[i]):
                        flags[t].append(f.tell())
                        break
            t = f.readline()

        ###########

        print("Treating connectivities...\n")

        if version == '4':
            for index in range(len(flags["$Elements"])):
                f.seek(flags["$Elements"][index])

                t = f.readline()    # line ignored (nb of elements)
                t = f.readline()

                while (t and f.tell() != flags["$EndElements"][index]):
                    t = t.strip("\t\n").split()

                    if(len(t) <= 1):
                        break
                    
                    if(t[2] != '2' and t[2] != '4'):
                        for i in range(int(t[-1])):
                            f.readline()
                    
                    if(t[2] == '2'):    # triangle
                        for i in range(int(t[-1])):
                            elem = f.readline().strip("\t\n").split()
                            lig = [int(elem[1]), int(elem[2]), int(elem[3])]
                            connect_2d.append(lig)

                    if(t[2] == '4'):    # tetrahedron
                        for i in range(int(t[-1])):
                            elem = f.readline().strip("\t\n").split()
                            lig = [int(elem[1]), int(elem[2]), int(elem[3]), int(elem[4])]
                            connect_3d.append(lig)

                    t = f.readline()

        if version == '2':
            for index in range(len(flags["$Elements"])):
                f.seek(flags["$Elements"][index])

                t = f.readline()    # line ignored (nb of elements)
                t = f.readline()
                
                while (t and f.tell() != flags["$EndElements"][index]):
                    t = t.split()
                    
                    if(len(t) <= 1):
                        break

                    if(t[1] == '2'):    # triangle
                        lig = [int(t[-3]), int(t[-2]), int(t[-1])]
                        connect_2d.append(lig)

                    if(t[1] == '4'):    # tetrahedron
                        lig = [int(t[-4]), int(t[-3]), int(t[-2]), int(t[-1])]
                        connect_3d.append(lig)
                        
                    t = f.readline()

        connect_2d = np.array(connect_2d)
        connect_3d = np.array(connect_3d)

        ###########

        print("Verifying nodes and edges...")

        # nodes

        nodes = []

        if version == '4':
            for index in range(len(flags["$Nodes"])):
                f.seek(flags["$Nodes"][index])
                f.readline()    # line ignored (nb of nodes)

                t = f.readline()

                while (t and f.tell() != flags["$EndNodes"][index]):
                    t = t.strip("\t\n").split()
                    
                    if(len(t) <= 1):
                        break
                    
                    for i in range(int(t[-1])):
                        f.readline()

                    for i in range(int(t[-1])):
                        node = f.readline().strip("\t\n").split()
                        nodes.append([float(node[0]), float(node[1]), float(node[2])])
                    
                    t = f.readline()

        if version == '2':
            for index in range(len(flags["$Nodes"])):
                f.seek(flags["$Nodes"][index])
                f.readline()    # line ignored (nb of nodes)

                t = f.readline()

                while (t and f.tell() != flags["$EndNodes"][index]):
                    t = t.strip("\t\n").split()
                    
                    if(len(t) <= 1):
                        break

                    nodes.append([float(t[1]), float(t[2]), float(t[3])])
                    
                    t = f.readline()

    nodes = np.array(nodes)
    
    dim = 3
    if(len(connect_3d) == 0):
        if(np.all(nodes[:, 0] == nodes[0, 0])):
            dim = 2
            nodes = nodes[:, 1:]
        elif(np.all(nodes[:, 1] == nodes[0, 1])):
            dim = 2
            nodes = nodes[:, -1:1]
        elif(np.all(nodes[:, 2] == nodes[0, 2])):
            dim = 2
            nodes = nodes[:, :2]
        else:
            dim = 2.5

    # Apparently Cimlib prefers normals looking down in 2D
    if(dim == 2):
        print("   - Checking normals")  # Actually only checking the first normal
        normal = np.cross(nodes[connect_2d[0][1]] - nodes[connect_2d[0][0]], nodes[connect_2d[0][2]] - nodes[connect_2d[0][0]])
        if(normal > 0):
            connect_2d = connect_2d[:, [0, 2, 1]]

    print("   - Detecting edges")

    if dim == 3:
        del connect_2d
        
        tris1 = connect_3d[:, [0, 2, 1]]  # Order is very important !
        tris2 = connect_3d[:, [0, 1, 3]]
        tris3 = connect_3d[:, [0, 3, 2]]
        tris4 = connect_3d[:, [1, 2, 3]]
        
        tris = np.concatenate((tris1, tris2, tris3, tris4), axis=0)
        tris_sorted = np.sort(tris, axis=1)  # creates a copy, may be source of memory error
        tris_sorted, uniq_idx, uniq_cnt = np.unique(tris_sorted, axis=0, return_index=True, return_counts=True)
        connect_2d = tris[uniq_idx][uniq_cnt == 1]

    if dim == 2:
        lin1 = connect_2d[:, [0, 1]]  # Once again, order is very important !
        lin2 = connect_2d[:, [2, 0]]
        lin3 = connect_2d[:, [1, 2]]
        
        lin = np.concatenate((lin1, lin2, lin3), axis=0)
        lin_sorted = np.sort(lin, axis=1)   # creates a copy, may be source of memory error
        lin_sorted, uniq_idx, uniq_cnt = np.unique(lin_sorted, axis=0, return_index=True, return_counts=True)
        connect_1d = lin[uniq_idx][uniq_cnt == 1]
        
    print("   - Detecting unused nodes")

    to_delete = np.arange(1, len(nodes) + 1)   # Every node index
    used_elems = np.unique(np.concatenate((connect_3d.flat, connect_2d.flat)))   # Every index of USED nodes

    bools_keep = np.in1d(to_delete, used_elems)
    to_delete = to_delete[~bools_keep]
    del used_elems

    print("   - Deleting unused nodes\n")

    nodes = nodes[bools_keep]
    del bools_keep

    if dim == 3 or dim == 2.5:
        connect_3d.flat -= np.searchsorted(to_delete, connect_3d.flat, side='left')
        connect_2d.flat -= np.searchsorted(to_delete, connect_2d.flat, side='left')

    if dim == 2:
        connect_2d.flat -= np.searchsorted(to_delete, connect_2d.flat, side='left')
        connect_1d.flat -= np.searchsorted(to_delete, connect_1d.flat, side='left')

    ##########

    nb_elems = len(connect_2d) + len(connect_3d)
    if dim == 2:
        nb_elems += len(connect_1d)
        print("Nb elements 1d : " + str(len(connect_1d)))

    print("Nb elements 2d : " + str(len(connect_2d)))
    print("Nb elements 3d : " + str(len(connect_3d)))
    print("Dimension : " + str(dim) + "\n")

    ###########

    print("Writing .t file...")

    with open(output, "w") as fo:
        lig = str(len(nodes)) + " " + str(dim) + " " + str(nb_elems) + " " + str(dim + 1) + "\n"
        if(dim == 2.5):
            lig = str(len(nodes)) + " 3 " + str(nb_elems) + " 4\n"
        fo.write(lig)

        for node in nodes:
            fo.write("{0:.16f} {1:.16f}".format(node[0], node[1]))
            if(dim == 3 or dim == 2.5):
                fo.write(" {0:.16f}".format(node[2]))
            fo.write(" \n")

        for e in connect_3d:
            fo.write(str(e[0]) + " " + str(e[1]) + " " + str(e[2]) + " " + str(e[3]) + " \n")

        for e in connect_2d:
            if(dim == 3 or dim == 2.5):
                fo.write(str(e[0]) + " " + str(e[1]) + " " + str(e[2]) + " 0 \n")
            else:
                fo.write(str(e[0]) + " " + str(e[1]) + " " + str(e[2]) + " \n")

        if dim == 2:
            for e in connect_1d:
                fo.write(str(e[0]) + " " + str(e[1]) + " 0 \n")

    print("Done.")
    return


def process_all_meshes():
    """Process all .msh files from meshes_folder and save .t files to t_files folder"""
    
    # Create t_files folder if it doesn't exist
    os.makedirs("t_files", exist_ok=True)
    
    # Check if meshes_folder exists
    if not os.path.exists("meshes"):
        print("Error: meshes folder not found!")
        return
    
    msh_files = [f for f in os.listdir("meshes") if f.endswith('.msh')]
    
    if not msh_files:
        print("No .msh files found in meshes")
        return
    
    print(f"Found {len(msh_files)} .msh files to convert\n")
    
    # Process each .msh file
    for msh_file in msh_files:
        input_path = os.path.join("meshes", msh_file)
        output_filename = msh_file.replace('.msh', '.t')
        output_path = os.path.join("t_files", output_filename)
        
        print(f"\nProcessing: {msh_file}")
        gmsh4mtc_single_step(input_path, output_path)
        print(f"Saved: {output_path}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single file mode (original behavior)
        fichier_entree = sys.argv[1]
        
        _, ext = os.path.splitext(fichier_entree)

        if ext == '.stl':
            try:
                import gmsh
                gmsh.initialize()
                gmsh.open(fichier_entree)
                gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
                gmsh.write(fichier_entree.rsplit('.stl', 1)[0] + '.msh')
                gmsh.finalize()
                fichier_entree = fichier_entree.rsplit('.stl', 1)[0] + '.msh'
            except:
                print("An error occured with your gmsh installation.")
                exit

        fichier_sortie = fichier_entree.rsplit('.msh', 1)[0] + '.t'
        gmsh4mtc_single_step(fichier_entree, fichier_sortie)
    else:
        # Batch mode - process all meshes from meshes
        process_all_meshes()

# Convert to geo, to msh and to t
f = convert_txt_to_geo("vanilla_profile/naca_0010.txt", 0.01, type="bezier")

for alpha in range(-40, -9, 2):
    rotate_and_mesh(f, alpha)
for alpha in range(10, 41, 2):
    rotate_and_mesh(f, alpha)
for alpha in range(-9, 10, 1):
    rotate_and_mesh(f, alpha)

process_all_meshes()
    