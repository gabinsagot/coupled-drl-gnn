import json
import os
import pickle
import re
import shutil
from typing import Any, Dict, List, Tuple

import meshio
import numpy as np
import pandas as pd
from tqdm import tqdm


def apply_Center2(Center2: list, Center2_path: str) -> None:
    """Applies the Center2 values to the Center2_path file.
    Center2 : list of  x y (z) coordinates
    Center2_path : file path"""
    with open(Center2_path, "w") as file:
        file.write(" ".join(map(str, Center2)))


def load_json_to_dict(json_file_path: str) -> dict:
    """
    Loads the content of a JSON file into a dictionary.

    :param json_file_path: Path to the JSON file
    :returns: Dictionary containing the data from the JSON file
    """
    try:
        with open(json_file_path, "r") as json_file:
            dic = json.load(json_file)
        return dic
    except FileNotFoundError:
        print(f"Error: file {json_file_path} not found.")
    except json.JSONDecodeError:
        print("Error: the json file is misformatted.")
    except Exception as e:
        print(f"Unexpected error: {e}")


def remove_directory(directory: str):
    try:
        shutil.rmtree(directory)
    except Exception as e:
        print(f"Error removing directory {directory}: {e}")
    return ()


def move_meshes(
    output_directory="meshes",
    extensions: list[str] = [".msh", ".geo_unrolled", ".vtk", ".t"],
    source_directory: str = "./",
):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for file in os.listdir(os.path.abspath(source_directory)):
        file = os.path.join(source_directory, file)
        if any(file.endswith(ext) for ext in extensions):
            destination_file = os.path.join(output_directory, os.path.basename(file))
            if os.path.abspath(file) != os.path.abspath(destination_file):
                shutil.copy2(file, output_directory)
                os.remove(file)


def xdmf_to_meshes(
    xdmf_file_path: str, verbose: bool = False
) -> Tuple[List[meshio.Mesh], List[float]]:
    """
    Returns meshio mesh objects for every timestep in an XDMF archive file. Also returns timestep list
    """
    meshes = []
    times = []
    with meshio.xdmf.TimeSeriesReader(xdmf_file_path) as reader:
        points, cells = reader.read_points_cells()
        for i in tqdm(
            range(reader.num_steps),
            desc="Extracting meshes from XDMF file",
            disable=not verbose,
        ):
            try:
                time, point_data, cell_data, _ = reader.read_data(i)
            except ValueError:
                try:
                    time, point_data, cell_data = reader.read_data(i)
                except Exception as e:
                    print(f"Error reading time/point/cell(/user) data: {e}")
                    raise
            mesh = meshio.Mesh(
                points, cells, point_data=point_data, cell_data=cell_data
            )
            meshes.append(mesh)
            times.append(time)
    return meshes, times


def meshes_to_xdmf(
    filename: str,
    meshes: List[meshio.Mesh],
    timestep: float | List[float] = 1.0,
    verbose: bool = False,
    drop_firststep=False,
) -> None:
    """
    Writes a time series of meshes (same points and cells) into XDMF/HDF5 format.
    The function will write two files: 'filename.xdmf' and 'filename.h5'.

    filename: chosen name for the archive files.
    meshes: List of meshes to compress, they need to share their cells and points.
    timestep: Timestep between two frames.
    """
    points = meshes[0].points
    cells = meshes[0].cells

    filename = os.path.splitext(filename)[0]
    h5_filename = f"{filename}.h5"
    xdmf_filename = f"{filename}.xdmf"

    # Open the TimeSeriesWriter for HDF5
    with meshio.xdmf.TimeSeriesWriter(xdmf_filename) as writer:
        # Write the mesh (points and cells) once
        writer.write_points_cells(points, cells)

        # Loop through time steps and write data
        if isinstance(timestep, (int, float)):
            timestep = [i * float(timestep) for i in range(len(meshes))]
        elif len(timestep) != len(meshes):
            raise ValueError("Length of timestep list must match the number of meshes.")
        for mesh, t in tqdm(
            zip(meshes, timestep),
            desc="Compressing mesh into XDMF files",
            disable=not verbose,
        ):
            point_data = mesh.point_data
            cell_data = mesh.cell_data
            if not (drop_firststep and t == 0.0):
                writer.write_data(t, point_data=point_data, cell_data=cell_data)

    # The H5 archive is systematically created in cwd with the original meshio library
    if os.path.exists(os.path.join(os.getcwd(), os.path.split(h5_filename)[1])):
        shutil.move(
            src=os.path.join(os.getcwd(), os.path.split(h5_filename)[1]),
            dst=h5_filename,
        )
    if verbose:
        print(f"Time series written to {xdmf_filename} and {h5_filename}")


def list_of_strings(arg):
    return arg.split(",")


def read_vtu_fields(filename):
    mesh = meshio.read(filename)
    return mesh.cell_data, mesh.point_data, mesh.user_data


def read_vtu_mesh(filename):
    mesh = meshio.read(filename)
    return mesh.points.astype(np.float32), mesh.cells


def get_unique_path(path):
    if not os.path.exists(path):
        return path
    base_path = path
    counter = 1
    while os.path.exists(f"{base_path}_v{counter}"):
        counter += 1
    return f"{base_path}_v{counter}"


def pretty_print_dict(d: dict) -> None:
    """Pretty print a dictionary that contains subdicts."""

    def recursive_print(d: dict, prefix: str = "") -> None:
        for key, value in d.items():
            if isinstance(value, dict):
                print(f"{'-' * 10} {prefix}{key} {'-' * (38 - len(prefix) - len(key))}")
                recursive_print(value, prefix + "    ")
            else:
                print(f"{prefix}{key:<30}: {value}")

    recursive_print(d)


def print_section(title: str, content: Dict[str, Any]) -> None:
    """Helper function to print a section with a title and key-value pairs."""
    print("-" * 50)
    print(f"\t\t\t{title}")
    print("-" * 50)
    pretty_print_dict(d=content)
    print("-" * 50, "\n")
    return


class Mtc:
    """Class to handle MTC file formatting and operations."""

    def __init__(self, path: str):
        self.path = path
        self.content = self.load()

    def load(self) -> str:
        """Load the MTC file content as a string."""
        try:
            with open(self.path, "r") as file:
                mtc_content = file.read()
            return mtc_content
        except FileNotFoundError:
            return ""

    def write(self, content: str, overwrite: bool = False) -> None:
        """Write the MTC content to the file. If overwrite is True, it will overwrite the file content,
        otherwise it will append the content."""
        if not overwrite:
            content = self.content + "\n" + content
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w") as file:
                file.write(content)
        except Exception as e:
            print(f"Error writing to MTC file {self.path}: {e}")

    @staticmethod
    def format_ModeleDeModeles(
        model_name: str = "Model",
        submodels_list: List[str] = ["Model1", "Model2"],
        mesh_model: str = "MaillagePrincipal",
    ) -> str:
        """Format a ModeleDeModeles mtc modele in the required format."""
        submodels_list = [f"\n\t\t{{ Modele= {model} }}" for model in submodels_list]
        modele_de_modeles_start = f"""{{ {model_name}= \n\t{{ Type= ModeleDeModeles }} \
        \n\t{{ Dependance=\n\t\t{{ Maillage= {mesh_model} }}"""
        modele_de_modeles_end = "\n\t}\n}"
        modele_de_modeles = (
            modele_de_modeles_start + "".join(submodels_list) + modele_de_modeles_end
        )
        return modele_de_modeles

    @staticmethod
    def format_mtc_declaration(list_mtc_paths: List[str]) -> str:
        """Format the mtc file declaration in the required format.

        :param list_mtc_paths: List of relative paths (from simu setup root)
        to the mtc files to be declared."""
        mtc_declaration = "\n".join(
            [f"{{ Fichier: {path} }}" for path in list_mtc_paths]
        )
        return mtc_declaration

    @staticmethod
    def format_definition_model(
        model_name: str = "DefinitionModel",
        mesh_model: str = "MaillagePrincipal",
        dim: int = 2,
    ) -> str:
        """Format the definition model in the required format.

        :param model_name: Name of the model
        :param mesh_model: Name of the mesh model used
        :param dim: Dimension of the mesh and geometre (2 or 3)

        :return: Formatted definition model string"""
        if dim == 2:
            origin = "0 0"
            axes = "1 0 0 1"
        elif dim == 3:
            origin = "0 0 0"
            axes = "1 0 0 0 1 0 0 0 1"
        else:
            raise ValueError("dim must be 2 or 3")

        definition_model = (
            f"{{ Definition{model_name}= \n\t{{ Type= ModeleDeMouvements }}\n\t{{ Data=\n\t\t"
            f"{{ Repere=\n\t\t\t{{ Dimension= {dim} }} \n\t\t\t{{ Nom= {model_name} }}\n\t\t\t"
            f"{{ Origine= {origin} }} \n\t\t\t{{ Axes = {axes} }} \n\t\t}}\n\t}}\n \t{{ Dependance= \n\t\t"
            f"{{ Maillage= {mesh_model} }}\n\t}}\n}}"
        )
        return definition_model

    @staticmethod
    def format_geo_model(
        model_name: str = "GeoModel",
        mesh_file: str = "meshes/mesh.t",
        mesh_model: str = "MaillagePrincipal",
    ) -> str:
        """Format the geometry model in the required format."""
        geo_model = (
            f"{{ Geo{model_name}= \n\t{{ Type= ModeleDeGeometres }}\n\t{{ Data=\n\t\t{{ Geometre= \n\t\t\t"
            f"{{ Nom= {model_name} }}\n\t\t\t{{ Type= GeometreAnalytique }}\n\t\t\t"
            f"{{ Data=\n\t\t\t\t{{ Forme=\n\t\t\t\t\t"
            f"{{ Type= FormeNewMaillageBis }}\n\t\t\t\t\t{{ Data=\n\t\t\t\t\t\t{{ M: {mesh_file} }}\n\t\t\t\t\t\t"
            f"{{ Localisation=\n\t\t\t\t\t\t\t{{ Brique= Boite }}\n\t\t\t\t\t\t\t"
            f"{{ Methode= Lineaire }}\n\t\t\t\t\t\t\t"
            f"{{ TailleMax= 1024 }}\n\t\t\t\t\t\t}}\n\t\t\t\t\t}}\n\t\t\t\t}}\n\t\t\t\t"
            f"{{ Repere= {model_name} }}\n\t\t\t}}\n\t\t}}\n\t}}\n\t{{ Dependance=\n \t\t"
            f"{{ Maillage= {mesh_model} }}\n \t\t{{ Modele= Definition{model_name} }}\n \t}}\n}}"
        )
        return geo_model

    @staticmethod
    def format_distance_model(
        model_name: str = "DistanceModel", mesh_model: str = "MaillagePrincipal"
    ) -> str:
        """Format the distance model in the required format."""
        distance_model = (
            f"{{ Distance{model_name}= \n\t{{ Type= ModeleParticulaire }}\n\t{{ Data=\n\t\t"
            f"{{ Champ= {{ Type= P1_Scalaire_Par }}\t{{ Nom= LevelSet{model_name} }}\t"
            f"{{ Data= ValeurItem 1 0.0 }}\t}}\n\t\t"
            f"{{ Champ= {{ Type= P1_Scalaire_Par }}\t{{ Nom= Appartient{model_name} }}\t"
            f"{{ Data= ValeurItem 1 0.0 }}\t}}\n\t\t"
            f"{{ ItemSolveur=\n\t\t\t{{ Type= ISGeometre }}\n\t\t\t{{ NbChampSolution= 2 }}\n\t\t\t"
            f"{{ ChampSolution= Appartient{model_name} LevelSet{model_name} }}\n\t\t\t{{ NbChampParametre= 2 }}\n\t\t\t"
            f"{{ ChampParametre= Coordonnees PrecisionFrontieres }}\n\t\t\t{{ Geometre= {model_name} }}\n\t\t\t"
            f"{{ Distance= 1 }}\n\t\t\t{{ Appartient= 1 }}\n\t}}\n\t}}\n\t{{ Dependance=\n\t\t"
            f"{{ Maillage= {mesh_model} }}\n\t\t{{ Champ= PrecisionFrontieres }}\n\t\t"
            f"{{ Champ= Coordonnees }}\n\t}}\n}}"
        )
        return distance_model

    @staticmethod
    def format_draglift_model(
        object_name: str = "0", dim: int = 2, mesh_model: str = "MaillagePrincipal"
    ) -> str:
        """Format the draglift model in the required format."""
        if dim != 2:
            raise ValueError(
                "Automatic DragLift model formatting only available in 2D for now!"
            )
        object_number = re.search(r"\d+$", object_name)
        object_number = object_number.group() if object_number else object_name
        draglift_model = (
            f"{{ DragLift{object_name}= \n\t{{ Type= ModeleDeModeles }}\n\t{{ Data= \n\t\t"
            f"{{ Champ= {{ Type= P0C_Vecteur_Par }}\t{{ Nom= VectorXi{object_number} }}\t"
            f"{{ Data= ValeurItem 2 1 0 }} }}\n\t\t"
            f"{{ Champ= {{ Type= P0C_Vecteur_Par }}\t{{ Nom= VectorYi{object_number} }}\t"
            f"{{ Data= ValeurItem 2 0 1 }} }}\n\t\t"
            f"{{ Champ= {{ Type= P1_Vecteur_Par }}\t{{ Nom= VectorX{object_number} }}\t"
            f"{{ Data= ValeurItem 2 1 0 }} }}\n\t\t"
            f"{{ Champ= {{ Type= P1_Vecteur_Par }}\t{{ Nom= VectorY{object_number} }}\t"
            f"{{ Data= ValeurItem 2 0 1 }} }}\n\t}}\n\t"
            f"{{ Dependance=  \n\t\t{{ Maillage= {mesh_model} }}\n\t\t"
            f"{{ Modele= StartCdCl{object_number} }}\t\n\t\t{{ Modele= CalculCx{object_number} }}\n\t\t"
            f"{{ Modele= CalculCy{object_number} }}\n\t\t"
            f"{{ Modele= CalculCxS{object_number} }}\n\t\t{{ Modele= CalculCyS{object_number} }}\n\t\t"
            f"{{ Modele= UpdateCxyS{object_number} }}\n\t\t"
            f"{{ Modele= Capteurs{object_number} }}\n\t}}\n}}\n\n"
            f"{{ StartCdCl{object_number}=\n\t{{ Type= ModeleArithmetique }}\n\t{{ Dependance= \n\t\t"
            f"{{ Maillage= {mesh_model} }}\n\t\t{{ Champ= Appartient{object_name} }}\n\t\t"
            f"{{ Champ= VectorXi{object_number} }}\n\t\t{{ Champ= VectorYi{object_number} }}\n\t\t"
            f"{{ Champ= Zero }}\n\t}}\n\t"
            f"{{ DependanceModifiable= \n\t\t{{ Champ= VectorX{object_number} }}\n\t\t"
            f"{{ Champ= VectorY{object_number} }}\n\t}}\n\t"
            f"{{ Operation= VectorX{object_number} = VectorXi{object_number} }}\n\t"
            f"{{ Operation= VectorY{object_number} = VectorYi{object_number} }}\n\t"
            f"{{ Operation= VectorX{object_number} *= Appartient{object_name} }}\n\t"
            f"{{ Operation= VectorY{object_number} *= Appartient{object_name} }}\n}}\n\n"
            f"{{ CalculCx{object_number}=\n\t{{ Type= ModeleFonctionnel }}\n\t{{ Data=\n\t\t"
            f"{{ Champ= {{ Type= P0_Scalaire_Par }}\t{{ Nom= Cx{object_number} }}\t"
            f"{{ Data= ValeurItem 1 0.0 }} }}\n\t\t"
            f"{{ SimplexSolveurFonctionnel=\n\t\t\t{{ Type= SsfDragLift }}\n\t\t\t"
            f"{{ NbChampSolution= 1 }}\n\t\t\t{{ ChampSolution= Cx{object_number} }}\n\t\t\t"
            f"{{ NbChampParametre= 7 }}\n\t\t\t"
            f"{{ ChampParametre= Pression Eta1 Vitesse VitesseMoins VectorX{object_number} PasDeTemps Un }}"
            f"\n\t\t}}\n\t\t"
            f"{{ Boucle= Volume }} \n\t}}\n\t{{ Dependance=\n\t\t"
            f"{{ Maillage= {mesh_model} }}\n\t\t{{ Champ= Vitesse }}\n\t\t{{ Champ= VitesseMoins }}\n\t\t"
            f"{{ Champ= PasDeTemps }}\n\t\t{{ Champ= Pression }}\n\t\t{{ Champ= Zero }}\n\t\t"
            f"{{ Champ= Un }}\n\t\t{{ Champ= VectorX{object_number} }}\n\t\t{{ Champ= Eta1 }}\n\t}}\n\t"
            f"{{ DependanceModifiable=\n\t\t{{ Champ= Cx{object_number} }}\n\t}}\n}}\n\n"
            f"{{ CalculCy{object_number}=\n\t{{ Type= ModeleFonctionnel }}\n\t{{ Data=\n\t\t"
            f"{{ Champ= {{ Type= P0_Scalaire_Par }}\t{{ Nom= Cy{object_number} }}\t"
            f"{{ Data= ValeurItem 1 0.0 }} }}\n\t\t"
            f"{{ SimplexSolveurFonctionnel=\n\t\t\t{{ Type= SsfDragLift }}\n\t\t\t"
            f"{{ NbChampSolution= 1 }}\n\t\t\t{{ ChampSolution= Cy{object_number} }}\n\t\t\t"
            f"{{ NbChampParametre= 7 }}\n\t\t\t"
            f"{{ ChampParametre= Pression Eta1 Vitesse VitesseMoins VectorY{object_number} PasDeTemps Un }}"
            f"\n\t\t}}\n\t\t"
            f"{{ Boucle= Volume }} \n\t}}\n\t{{ Dependance=\n\t\t"
            f"{{ Maillage= {mesh_model} }}\n\t\t{{ Champ= Vitesse }}\n\t\t{{ Champ= VitesseMoins }}\n\t\t"
            f"{{ Champ= PasDeTemps }}\n\t\t{{ Champ= Pression }}\n\t\t{{ Champ= Zero }}\n\t\t"
            f"{{ Champ= Un }}\n\t\t{{ Champ= VectorY{object_number} }}\n\t\t{{ Champ= Eta1 }}\n\t}}\n\t"
            f"{{ DependanceModifiable=\n\t\t{{ Champ= Cy{object_number} }}\n\t}}\n}}\n\n"
            f"{{ CalculCxS{object_number}=\n\t{{ Type= ModeleParticulaire }}\n\t{{ Data=\n\t\t"
            f"{{ Champ= {{ Type= P0C_Scalaire_Par }}\t{{ Nom= CxS{object_number} }}\t"
            f"{{ Data= ValeurItem 1 0.0 }} }}\n\t\t"
            f"{{ Boucle= Volume }}\n\t\t{{ ItemSolveur=\n\t\t\t{{ Type= ISSomme }}\n\t\t\t"
            f"{{ Operations=\n\t\t\t\t{{ Initialise= Zero }}\n\t\t\t}}\n\t\t\t"
            f"{{ NbChampSolution= 1 }}\n\t\t\t{{ ChampSolution= CxS{object_number} }}\n\t\t\t"
            f"{{ NbChampParametre= 1 }}\n\t\t\t{{ ChampParametre= Cx{object_number} }}\n\t\t}}\n\t}}\n\t"
            f"{{ Dependance=\n\t\t{{ Maillage= {mesh_model} }}\n\t\t"
            f"{{ Champ= Cx{object_number} }}\n\t\t{{ Champ= Zero }}\n\t}}\n\t"
            f"{{ DependanceModifiable=\n\t\t{{ Champ= CxS{object_number} }}\n\t}}\n}}\n\n"
            f"{{ CalculCyS{object_number}=\n\t{{ Type= ModeleParticulaire }}\n\t{{ Data=\n\t\t"
            f"{{ Champ= {{ Type= P0C_Scalaire_Par }}\t{{ Nom= CyS{object_number} }}\t"
            f"{{ Data= ValeurItem 1 0.0 }} }}\n\t\t"
            f"{{ Boucle= Volume }}\n\t\t{{ ItemSolveur=\n\t\t\t{{ Type= ISSomme }}\n\t\t\t"
            f"{{ Operations=\n\t\t\t\t{{ Initialise= Zero }}\n\t\t\t}}\n\t\t\t"
            f"{{ NbChampSolution= 1 }}\n\t\t\t{{ ChampSolution= CyS{object_number} }}\n\t\t\t"
            f"{{ NbChampParametre= 1 }}\n\t\t\t{{ ChampParametre= Cy{object_number} }}\n\t\t}}\n\t}}\n\t"
            f"{{ Dependance=\n\t\t{{ Maillage= {mesh_model} }}\n\t\t"
            f"{{ Champ= Cy{object_number} }}\n\t\t{{ Champ= Zero }}\n\t}}\n\t"
            f"{{ DependanceModifiable=\n\t\t{{ Champ= CyS{object_number} }}\n\t}}\n}}\n\n"
            f"{{ UpdateCxyS{object_number}=\n\t{{ Type= ModeleArithmetique }}\n\t{{ Dependance=\n\t\t"
            f"{{ Maillage= {mesh_model} }}\n\t\t{{ Champ= mVs }}\n\t}}\n\t"
            f"{{ DependanceModifiable=\n\t\t{{ Champ= CxS{object_number} }}\n\t\t"
            f"{{ Champ= CyS{object_number} }}\n\t}}\n\t"
            f"{{ Operation= CxS{object_number} *= mVs }}\n\t{{ Operation= CyS{object_number} *= mVs }}\n}}\n\n"
            f"{{ Capteurs{object_number}=\n\t{{ Affichage= 1 }}\n\t{{ Type= ModeleCapteur }}\n\t{{ Data= \n\t\t"
            f"{{ NomFichier= Resultats/Efforts{object_number} }}\n\t\t"
            f"{{ NbCapteurs= 0 }}\n\t\t{{ HAdaptation= 1 }}\t\n\t}}\n\t"
            f"{{ Dependance= \n\t\t{{ Maillage= {mesh_model} }}\n\t\t{{ Champ= CompteurTemps }}\n\t\t"
            f"{{ Champ= Temps }}\n\t\t{{ Champ= CxS{object_number} }}\n\t\t"
            f"{{ Champ= CyS{object_number} }}\n\t}}\n}}\n"
        )
        return draglift_model

    @staticmethod
    def modif_champ(mtc_content: str, raise_not_found: bool = False, **kwargs) -> str:
        """Modify the value of multiple champ fields in an mtc file, assuming each champ is declared on a single line.
        Usage: modif_champ(mtc_content, Champ1=value1, Champ2=value2, ...)
        """
        for champ_name, champ_value in kwargs.items():
            found = False
            for line in mtc_content.splitlines():
                if f"{{ Nom= {champ_name} }}" in line:
                    found = True
                    new_line = re.sub(
                        r"Data= ValeurItem [\d ]+ [\d\.\-eE]+",
                        (
                            f"Data= ValeurItem {len(champ_value) if isinstance(champ_value, list) else 1} "
                            f"{' '.join(map(str, champ_value)) if isinstance(champ_value, list) else champ_value}"
                        ),
                        line,
                    )
                    mtc_content = mtc_content.replace(line, new_line)
                    break
            if not found and raise_not_found:
                raise KeyError(f"Champ '{champ_name}' not found in MTC file content.")
        return mtc_content

    @staticmethod
    def modif_target(mtc_content: str, raise_not_found: bool = False, **kwargs) -> str:
        """Modify the value of multiple Target fields in an mtc file, assuming each Target is declared on a single line.
        Usage: modif_target(mtc_content, Target1=value1, Target2=value2, ...)
        """
        for target_name, target_value in kwargs.items():
            found = False
            for line in mtc_content.splitlines():
                if f"{{ Target= {target_name} " in line:
                    found = True
                    new_line = re.sub(
                        r"\{ Target= " + re.escape(target_name) + r" [^\}]+\}",
                        f"{{ Target= {target_name} {target_value} }}",
                        line,
                    )
                    mtc_content = mtc_content.replace(line, new_line)
                    break
            if not found and raise_not_found:
                raise KeyError(f"Target '{target_name}' not found in MTC file content.")
        return mtc_content


def get_simu_name(simu_dir: str) -> str:
    """Get the simulation name from the directory path.
    import argparse

    Args:
        simu_dir (str): path to the simulation directory.

    Returns:
        str: name of the simulation repo.
    """
    simu_dir = simu_dir.rstrip("/")
    return os.path.basename(simu_dir)


def load_dataframe(df_path: str) -> pd.DataFrame:
    """Load a DataFrame from a pickle file."""
    with open(df_path, "rb") as f:
        try:
            df = pickle.load(f)
        except pickle.UnpicklingError:
            raise ValueError(
                f"Failed to load DataFrame from {df_path}. The file may be corrupted or not a valid DataFrame."
            )
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Loaded object from {df_path} is not a pandas DataFrame.")
    return df


def save_dataframe(df: pd.DataFrame, out_path: str, stringify: bool = True) -> None:
    """
    Save a DataFrame to a pickle file and optionally as a string.

    Args:
        df (pd.DataFrame): DataFrame to save.
        out_path (str): Path/name under which to save the DataFrame (no extension).
        stringify (bool): If True, also save a string representation of the DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    with open(f"{out_path}.pkl", "wb") as f:
        pickle.dump(df, f)
    if stringify:
        with open(f"{out_path}.txt", "w") as f:
            f.write(df.to_string(index=False))


def get_predefined_airfoil(airfoil_name: str, n_points: int = 10) -> np.ndarray:
    """Get predefined airfoil points using actual NACA formulas.

    Args:
        airfoil_name (str): Name of the airfoil (e.g., "NACA0010").

    Returns:
        np.ndarray: Array of shape (n_points, 2) with (x, y) coordinates.
    """

    def _naca_4digit(naca_code: str, n_points: int = 10) -> np.ndarray:
        """Generate NACA 4-digit airfoil coordinates using the actual formula.

        Args:
            naca_code (str): 4-digit NACA code (e.g., '0012')
            n_points (int): Number of points on upper surface, including trailing/leading edge points.

        Returns:
            np.ndarray: Array of (x, y) coordinates forming a closed loop.
                    Points go: trailing edge -> upper surface -> leading edge ->
                    lower surface -> back to trailing edge (closed)
        """
        # Parse NACA 4-digit code
        t = int(naca_code[2:4]) / 100.0  # Maximum thickness as fraction of chord

        # Cosine spacing for better leading edge resolution
        # enforce minimum to avoid degenerate spacing
        if n_points < 4:
            print(
                f"Warning: n_points too low ({n_points}) for creating NACA airfoil, setting to minimum of 4."
            )
        n_points = max(4, n_points)
        beta = np.linspace(0, np.pi, n_points)
        x = 0.5 * (1.0 - np.cos(beta))

        # NACA 4-digit thickness distribution formula
        yt = (
            5
            * t
            * (
                0.2969 * np.sqrt(x)
                - 0.1260 * x
                - 0.3516 * x**2
                + 0.2843 * x**3
                - 0.1015 * x**4
            )
        )

        # Upper surface: from trailing edge (x=1) to leading edge (x=0)
        x_upper = x[::-1]
        y_upper = yt[::-1]

        # Lower surface: from leading edge (x=0) to trailing edge (x=1)
        x_lower = x[1:]  # skip first point (in upper)
        y_lower = -yt[1:]

        # Combine into closed loop
        x_combined = np.concatenate([x_upper, x_lower])
        y_combined = np.concatenate([y_upper, y_lower])

        points = np.column_stack([x_combined, y_combined])
        return points

    if airfoil_name.upper().startswith("NACA"):
        naca_code = airfoil_name.upper().replace("NACA", "")
        if len(naca_code) == 4 and naca_code.isdigit():
            return _naca_4digit(naca_code, n_points=n_points)
        else:
            raise ValueError(f"Invalid NACA code: {naca_code}. Must be 4 digits.")
    else:
        raise ValueError(
            f"Unknown airfoil type: {airfoil_name}. "
            f"Use NACA#### format (e.g., NACA0012)"
        )


def morphed_airfoil(
    camber_parameters: List[float],
    thickness_parameters: List[float],
    plot: bool = False,
) -> np.ndarray:
    """Create a morphed airfoil based on NACA0010 using camber and thickness parameters.
    Uses a simple point adjustment method, where camber and thickness control points are adjusted.
    The number of control points is inferred from the length of camber_parameters with
    control points excluding leading/trailing edge points. Thus, total airfoil points = 2 * num_control_points + 3
    because the trailing edge is 2 points.
    Convention is that point 0 and -1 is trailing edge, and indexing goes counterclockwise.

    Args:
        camber_parameters (List[float]): Parameters to adjust camber of airfoil start points.\
            Camber parameters are the new y-coordinates for the camber line points except for leading/trailing edges.
        thickness_parameters (List[float]): Parameters to adjust the thickness distribution, applied symmetrically.\
            about the camber line, not applied to leading/trailing and surrounding points (2 less than camber).
        plot (bool): If True, plot the morphed airfoil points (blue) vs original NACA0010 (black).
    Returns:
        np.ndarray: Morphed airfoil points of shape (n_points, 2).
    """
    # get start airfoil points
    num_control_points = len(camber_parameters)
    num_total_points = num_control_points * 2 + 3
    airfoil_points = get_predefined_airfoil(
        "NACA0010", n_points=1 + num_total_points // 2
    )

    # sanity check
    if len(thickness_parameters) != num_control_points - 2:
        raise ValueError(
            "Number of thickness parameters does not match number of control points - 2 "
            "(airfoil points without leading/trailing edge and surrounding points)."
            f"Should get {num_control_points - 2} thickness parameters, got {len(thickness_parameters)}."
        )
    if any(tp < 0 for tp in thickness_parameters):
        raise ValueError("Thickness parameters must be non-negative.")

    # morphing: fix trail and leading edge points, adjust camber and thickness
    morphed_airfoil = airfoil_points.copy()
    # edge_points = [0, num_control_points+1,-1]  # trailing and leading edge indices
    adj_to_edge_points = [
        1,
        num_control_points,
        num_control_points + 2,
        num_total_points - 2,
    ]
    belly_points_upper = list(range(2, num_control_points, 1))
    belly_points_lower = list(range(num_total_points - 3, num_control_points + 2, -1))
    # apply camber to edge-adjacent points
    morphed_airfoil[adj_to_edge_points[0], 1] += camber_parameters[0]
    morphed_airfoil[adj_to_edge_points[1], 1] += camber_parameters[-1]
    morphed_airfoil[adj_to_edge_points[2], 1] += camber_parameters[-1]
    morphed_airfoil[adj_to_edge_points[3], 1] += camber_parameters[0]
    # apply thickness and camber to belly points, symmetrically across camber line
    for i, pt_idx in enumerate(belly_points_upper):
        morphed_airfoil[pt_idx, 1] = (
            camber_parameters[i + 1] + thickness_parameters[i] / 2
        )
    for i, pt_idx in enumerate(belly_points_lower):
        morphed_airfoil[pt_idx, 1] = (
            camber_parameters[i + 1] - thickness_parameters[i] / 2
        )
    if plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.plot(
            airfoil_points[:, 0],
            airfoil_points[:, 1],
            color="black",
            label="NACA0010",
            linewidth=2,
            marker="o",
        )
        plt.plot(
            morphed_airfoil[:, 0],
            morphed_airfoil[:, 1],
            color="blue",
            label="Morphed Airfoil",
            linewidth=2,
            marker="o",
        )
        plt.axis("equal")
        plt.title("Morphed Airfoil vs NACA0010")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.show()
    return morphed_airfoil


def compute_airfoil_surface(airfoil_points: np.ndarray) -> float:
    """Compute the surface area of an airfoil defined by its 2D points using the shoelace formula.

    Args:
        airfoil_points (np.ndarray): Array of shape (n_points, 2) with (x, y) coordinates.

    Returns:
        float: Surface area of the airfoil.
    """
    x = airfoil_points[:, 0]
    y = airfoil_points[:, 1]
    area = 0.5 * np.abs(
        np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
    )  # Shoelace formula
    return area
