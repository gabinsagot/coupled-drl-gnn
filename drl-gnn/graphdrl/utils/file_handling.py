import os
import platform
import re
import shutil
from typing import Dict, List

import json


def parse_parameters_file(file_path: str) -> Dict[str, str]:
    params = json.load(open(file_path))
    return params


def adjust_driver_to_architecture(params):
    """Adjust the driver path in params based on the system architecture."""
    system = platform.system()
    arch = os.uname().machine

    if system == "Linux":
        # Check if it's Ubuntu or CentOS
        try:
            with open("/etc/os-release", "r") as f:
                os_info = f.read()
            if "ubuntu" in os_info.lower() and arch == "x86_64":
                driver_path = "environment_config/driver/ubuntu64_cfd_driver"
            elif "centos" in os_info.lower():
                driver_path = "environment_config/driver/laffitte_cfd_driver"
            else:
                raise ValueError(
                    f"Unsupported Linux distribution or architecture for "
                    f"available cimlib test driver binaries: {os_info}"
                )
        except FileNotFoundError:
            raise ValueError("Could not determine Linux distribution")
    else:
        raise ValueError(f"Unsupported operating system: {system}")

    params["traj_parameters"]["driver"] = driver_path
    return params


def gather_cases(case_folder: str, case_base_name: str = "") -> Dict[str, str]:
    """
    Gather all cases from a folder (case_folder) of xdmfs that have a specific base name (case_base_name).

    Returns a dictionary of keys case ID and values xdmf file path.
    """
    cases = {
        os.path.splitext(f)[0].split("_")[-1]: os.path.join(case_folder, f)
        for f in os.listdir(case_folder)
        if f.startswith(case_base_name) and f.endswith(".xdmf")
    }
    cases = dict(sorted(cases.items()))
    return cases


def gather_cases_for_reward_recursive(
    case_folder: str, case_base_name: str = "", get_data_files_only: bool = False
) -> List[str]:
    """
    Gather all xdmf or csv files from a case folder (case_folder) that have a specific base name (case_base_name).
    This function searches recursively in subdirectories.

    Returns a dictionary of keys case ID and values xdmf file path.
    """
    # Return empty list if folder does not exist
    if not os.path.exists(case_folder):
        return []

    cases = []
    seen = set()
    for root, dirs, files in os.walk(case_folder):
        for f in files:
            # base name filter
            if case_base_name and not f.startswith(case_base_name):
                continue
            # suffix check
            if not f.lower().endswith((".xdmf", ".csv")):
                continue
            rel_path = os.path.relpath(os.path.join(root, f))
            # nvoid duplicates and keep rel paths
            if rel_path not in seen:
                seen.add(rel_path)
                if get_data_files_only and not f.lower().endswith(".csv"):
                    continue
                if not get_data_files_only and not f.lower().endswith(".xdmf"):
                    continue
                cases.append(rel_path)
    cases.sort()
    return cases


def remove_directory(directory: str):
    try:
        shutil.rmtree(directory)
    except Exception as e:
        print(f"Error removing directory {directory}: {e}")
    return ()


def _robust_copyfileobj(
    src: str, dst: str, attempts: int = 3, chunk: int = 16 * 1024, backoff: float = 0.1
) -> bool:
    """
    Attempt to copy a file by streaming (copyfileobj) with retries on transient errors.

    Args:
        src (str): Source file path.
        dst (str): Destination file path.
        attempts (int): Number of attempts to try copying.
        chunk (int): Chunk size for copying.
        backoff (float): Backoff time in seconds between attempts.

    Returns True on success, False if all attempts failed (without raising).
    """
    import errno
    import time

    for attempt in range(attempts):
        try:
            with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
                shutil.copyfileobj(fsrc, fdst, length=chunk)
            try:
                shutil.copystat(src, dst)
            except Exception:
                # metadata copy is best-effort
                pass
            return True
        except (BlockingIOError, OSError) as e:
            err_no = getattr(e, "errno", None)
            if err_no in (errno.EAGAIN, errno.EWOULDBLOCK, None):
                time.sleep(backoff * (attempt + 1))
                continue
            # non-transient error -> re-raise
            raise
    return False


def _copy_with_fallback(src: str, dst: str) -> None:
    """
    Copy a file from src to dst using a fast path, with a robust fallback for EAGAIN/BlockingIOError.

    Args:
        src (str): Source file path.
        dst (str): Destination file path.

    Will raise on non-transient errors or if the final attempt fails.
    """
    try:
        shutil.copy2(src, dst)
        return
    except (BlockingIOError, OSError) as e:
        import errno

        err_no = getattr(e, "errno", None)
        # If this is not an EAGAIN/EWOULDBLOCK-style transient error, re-raise
        if isinstance(e, OSError) and err_no not in (errno.EAGAIN, errno.EWOULDBLOCK):
            raise

        # Try robust streaming copy with retries
        if _robust_copyfileobj(src, dst):
            return

        # Last attempt: let copy2 raise the final exception if it fails
        shutil.copy2(src, dst)


def _safe_remove(path: str) -> None:
    """
    Remove a file path in a best-effort manner, ignore any error.
    """
    try:
        os.remove(path)
    except Exception:
        pass


def move_meshes(
    output_directory: str = "meshes",
    extensions: list[str] = [".msh", ".geo_unrolled", ".vtk", ".t"],
    source_directory: str = "./",
):
    """
    Move matching mesh files from source_directory into output_directory.

    Args:
        output_directory (str): Directory to move files into.
        extensions (list[str]): List of file extensions to consider.
        source_directory (str): Directory to search for source files.

    Uses a resilient copy routine to handle transient BlockingIOError/EAGAIN errors.
    """
    os.makedirs(output_directory, exist_ok=True)

    src_root = os.path.abspath(source_directory)
    for entry in os.listdir(src_root):
        src = os.path.join(src_root, entry)
        # only consider regular files and the requested extensions
        if not os.path.isfile(src):
            continue
        if not any(entry.endswith(ext) for ext in extensions):
            continue

        destination_file = os.path.join(output_directory, os.path.basename(entry))
        # skip if source and destination are identical
        if os.path.abspath(src) == os.path.abspath(destination_file):
            continue

        # perform copy with fallback and then remove the source if copy succeeded
        _copy_with_fallback(src=src, dst=destination_file)
        _safe_remove(path=src)


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
            raise RuntimeError(f"Error writing to MTC file {self.path}: {e}") from e

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
