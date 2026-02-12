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

import sys, os
import numpy as np
import time

###########

def gmsh4mtc_single_step(input,output):

    print("#########################################################")
    print("# GMSH VERSION 2 to 4 MTC CONVERTER - CFL - MINES PARIS #")
    print("#########################################################")
    print("No mtc.exe version")
    print("")

    print("Initialisation...\n")

    with open(input) as f:
        f.readline()
        version = f.readline().split()[0]
        if len(version) > 1 :
            version = version.split('.')[0]
        if version != '4' and version != '2' :
            print("This version of gmsh isn't supported")
            input("Press enter to close...")
            sys.exit()

        flags = { "$Nodes" : [],
                "$EndNodes" : [],
                "$Elements" : [],
                "$EndElements" : [] }

        connect_3d = []
        connect_2d = []
        connect_1d = []

        ###########

        print("Getting position flags...\n")

        t = f.readline()

        while t:
            t = t.strip("\t\n")
            if t.startswith("$") :
                for i in range(len(list(flags.keys()))):
                    if(t == list(flags.keys())[i]):
                        flags[t].append(f.tell())
                        break
            t = f.readline()

        ###########

        print("Treating connectivities...\n")

        if version == '4' :
            for index in range(len(flags["$Elements"])) :
                f.seek(flags["$Elements"][index])

                t = f.readline()    # line ignored (nb of elements)
                t = f.readline()

                while (t and f.tell()!=flags["$EndElements"][index]):
                    t = t.strip("\t\n").split()

                    if(len(t) <= 1):
                        break
                    
                    if(t[2]!='2' and t[2]!='4'):
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

        if version == '2' :
            for index in range(len(flags["$Elements"])) :
                f.seek(flags["$Elements"][index])

                t = f.readline()    # line ignored (nb of elements)
                t = f.readline()
                
                while (t and f.tell()!=flags["$EndElements"][index]):
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

        if version == '4' :
            for index in range(len(flags["$Nodes"])) :
                f.seek(flags["$Nodes"][index])
                f.readline()    # line ignored (nb of nodes)

                t = f.readline()

                while (t and f.tell()!=flags["$EndNodes"][index]):
                    t = t.strip("\t\n").split()
                    
                    if(len(t) <= 1):
                        break
                    
                    for i in range(int(t[-1])):
                        f.readline()

                    for i in range(int(t[-1])):
                        node = f.readline().strip("\t\n").split()
                        nodes.append([float(node[0]), float(node[1]), float(node[2])])
                    
                    t = f.readline()

        if version == '2' :
            for index in range(len(flags["$Nodes"])) :
                f.seek(flags["$Nodes"][index])
                f.readline()    # line ignored (nb of nodes)

                t = f.readline()

                while (t and f.tell()!=flags["$EndNodes"][index]):
                    t = t.strip("\t\n").split()
                    
                    if(len(t) <= 1):
                        break

                    nodes.append([float(t[1]), float(t[2]), float(t[3])])
                    
                    t = f.readline()

    nodes = np.array(nodes)
    
    dim = 3
    if(len(connect_3d) == 0):
        if(np.all(nodes[:,0]==nodes[0,0])):
            dim = 2
            nodes = nodes[:,1:]
        elif(np.all(nodes[:,1]==nodes[0,1])):
            dim = 2
            nodes = nodes[:,-1:1]
        elif(np.all(nodes[:,2]==nodes[0,2])):
            dim = 2
            nodes = nodes[:,:2]
        else:
            dim = 2.5

    # Apparently Cimlib prefers normals looking down in 2D
    if(dim == 2):
        print("   - Checking normals") # Actually only checking the first normal
        normal = np.cross(nodes[connect_2d[0][1]]-nodes[connect_2d[0][0]],nodes[connect_2d[0][2]]-nodes[connect_2d[0][0]])
        if(normal>0):
            connect_2d = connect_2d[:,[0,2,1]]

    print("   - Detecting edges")

    if dim == 3 :
        del connect_2d
        
        tris1 = connect_3d[:,[0,2,1]]  # Order is very important !
        tris2 = connect_3d[:,[0,1,3]]
        tris3 = connect_3d[:,[0,3,2]]
        tris4 = connect_3d[:,[1,2,3]]
        
        tris = np.concatenate((tris1,tris2,tris3,tris4), axis=0)
        tris_sorted = np.sort(tris, axis=1) # creates a copy, may be source of memory error
        tris_sorted, uniq_idx, uniq_cnt = np.unique(tris_sorted, axis=0, return_index=True, return_counts=True)
        connect_2d = tris[uniq_idx][uniq_cnt==1]

    if dim == 2 :
        lin1 = connect_2d[:,[0,1]]  # Once again, order is very important !
        lin2 = connect_2d[:,[2,0]]
        lin3 = connect_2d[:,[1,2]]
        
        lin = np.concatenate((lin1,lin2,lin3), axis=0)
        lin_sorted = np.sort(lin, axis=1)   # creates a copy, may be source of memory error
        lin_sorted, uniq_idx, uniq_cnt = np.unique(lin_sorted, axis=0, return_index=True, return_counts=True)
        connect_1d = lin[uniq_idx][uniq_cnt==1]
        
    print("   - Detecting unused nodes")

    to_delete = np.arange(1,len(nodes)+1)   # Every node index
    used_elems = np.unique(np.concatenate((connect_3d.flat,connect_2d.flat)))   # Every index of USED nodes

    bools_keep = np.in1d(to_delete, used_elems)
    to_delete = to_delete[~bools_keep]
    del used_elems

    print("   - Deleting unused nodes\n")

    nodes = nodes[bools_keep]
    del bools_keep

    if dim == 3 or dim == 2.5 :
        connect_3d.flat -= np.searchsorted(to_delete, connect_3d.flat, side='left')
        connect_2d.flat -= np.searchsorted(to_delete, connect_2d.flat, side='left')

    if dim == 2 :
        connect_2d.flat -= np.searchsorted(to_delete, connect_2d.flat, side='left')
        connect_1d.flat -= np.searchsorted(to_delete, connect_1d.flat, side='left')

    ##########

    nb_elems = len(connect_2d)+len(connect_3d)
    if dim == 2 :
        nb_elems += len(connect_1d)
        print("Nb elements 1d : "+str(len(connect_1d)))

    print("Nb elements 2d : "+str(len(connect_2d)))
    print("Nb elements 3d : "+str(len(connect_3d)))
    print("Dimension : "+str(dim)+"\n")

    ###########

    print("Writing .t file...")

    with open(output,"w") as fo:
        lig = str(len(nodes))+" "+str(dim)+" "+str(nb_elems)+" "+str(dim+1)+"\n"
        if(dim==2.5):
            lig = str(len(nodes))+" 3 "+str(nb_elems)+" 4\n"
        fo.write(lig)

        for node in nodes :
            fo.write("{0:.16f} {1:.16f}".format(node[0],node[1]))
            if(dim==3 or dim==2.5):
                fo.write(" {0:.16f}".format(node[2]))
            fo.write(" \n")

        for e in connect_3d:
            fo.write(str(e[0]) + " " + str(e[1]) + " " + str(e[2]) + " " + str(e[3]) + " \n")

        for e in connect_2d:
            if(dim==3 or dim==2.5):
                fo.write(str(e[0]) + " " + str(e[1]) + " " + str(e[2]) + " 0 \n")
            else:
                fo.write(str(e[0]) + " " + str(e[1]) + " " + str(e[2]) + " \n")

        if dim==2 :
            for e in connect_1d:
                fo.write(str(e[0]) + " " + str(e[1]) + " 0 \n")

    print("Done.")
    return

if __name__ == "__main__":
    #fichier_entree = sys.argv[1]
    
    fichier_entree = input("Enter the path to the .msh or .stl file to convert: ").strip('"')
    _, ext = os.path.splitext(fichier_entree)

    if ext == '.stl':
        try :
            import gmsh
            gmsh.initialize()
            gmsh.open(fichier_entree)
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.write(fichier_entree.rsplit('.stl', 1)[0] + '.msh')
            gmsh.finalize()
            fichier_entree = fichier_entree.rsplit('.stl', 1)[0] + '.msh'
        except :
            print("An error occured with your gmsh installation.")
            exit

    fichier_sortie = fichier_entree.rsplit('.msh', 1)[0] + '.t'
    gmsh4mtc_single_step(fichier_entree, fichier_sortie)