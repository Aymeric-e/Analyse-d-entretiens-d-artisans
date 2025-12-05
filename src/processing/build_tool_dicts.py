"""
build_tool_dicts.py

Ce script lit deux fichiers CSV :
1. Le fichier d'entretien contenant les colonnes "Nom Fichier", "Matériau", "Artisanat".
2. Le fichier des outils contenant les colonnes "filename", "text", "tools_found_unique".

Pour chaque catégorie (matériau et artisanat), le script :
- Compte le nombre d'occurrences de chaque outil dans les fichiers correspondants.
- Génère deux fichiers CSV de sortie dans un dossier spécifié :
    - dict_outils_materiau.csv
    - dict_outils_artisanat.csv

Usage :
    python build_tool_dicts.py <csv_entretien> <csv_tools> <dossier_sortie>

Si aucun argument n'est fourni, le script utilise des chemins par défaut définis dans le bloc __main__.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

from utils.logger_config import setup_logger

logger = setup_logger(__name__, level="INFO")


def load_entretien_file(path):
    """
    Charge le fichier d'entretien et retourne deux dictionnaires.

    Le fichier CSV doit contenir au moins les colonnes :
        "Nom Fichier", "Matériau", "Artisanat"

    Args:
        path (str | Path): Chemin vers le fichier CSV d'entretien.

    Returns:
        tuple[dict, dict]:
            - file_to_materiau: dictionnaire fichier -> matériau
            - file_to_artisanat: dictionnaire fichier -> artisanat
    """
    file_to_materiau = {}
    file_to_artisanat = {}

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("Nom Fichier", "").strip()
            materiau = row.get("Matériau", "").strip()
            artisanat = row.get("Artisanat", "").strip()

            if filename:
                file_to_materiau[filename] = materiau
                file_to_artisanat[filename] = artisanat

    return file_to_materiau, file_to_artisanat


def load_tools_file(path):
    """
    Charge le fichier tools et retourne un dictionnaire fichier -> liste d'outils.

    Le fichier CSV doit contenir au moins les colonnes :
        "filename", "tools_found_unique"

    Args:
        path (str | Path): Chemin vers le fichier CSV des outils.

    Returns:
        dict[str, list[str]]: Mapping du nom de fichier vers la liste des outils uniques trouvés.
    """
    file_to_tools = {}

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("filename", "").strip().replace(" ", "_").replace("_traitement_A", "")
            tools_raw = row.get("tools_found_unique", "")
            tools = [t.strip() for t in tools_raw.split(",") if t.strip()]

            if filename:
                file_to_tools[filename] = tools

    return file_to_tools


def count_tools_by_category(mapping_file_to_category, mapping_file_to_tools):
    """
    Compte les occurrences des outils par catégorie.

    Args:
        mapping_file_to_category (dict[str, str]): Mapping fichier -> catégorie (matériau ou artisanat)
        mapping_file_to_tools (dict[str, list[str]]): Mapping fichier -> liste d'outils

    Returns:
        dict[str, dict[str, int]]: Dictionnaire catégorie -> dictionnaire outil -> count
    """
    result = defaultdict(lambda: defaultdict(int))

    for filename, category in mapping_file_to_category.items():
        if not category:
            continue
        tools = mapping_file_to_tools.get(filename, [])
        for tool in tools:
            result[category][tool] += 1

    return result


def write_output_csv(path, category_name, data_dict):
    """
    Écrit un dictionnaire d'outils par catégorie dans un fichier CSV.

    Args:
        path (str | Path): Chemin du fichier de sortie CSV.
        category_name (str): Nom de la colonne pour la catégorie (ex: "materiau", "artisanat").
        data_dict (dict[str, dict[str, int]]): Dictionnaire catégorie -> outil -> count
    """
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([category_name, "outil", "nombre_apparition"])

        for category, tool_dict in data_dict.items():
            for tool, count in tool_dict.items():
                writer.writerow([category, tool, count])


def main():
    """
    Fonction principale pour générer les CSV de comptage d'outils par catégorie.

    Utilisation :
        python build_tool_dicts.py <csv_entretien> <csv_tools> <dossier_sortie>

    Si aucun argument n'est fourni, utilise les chemins par défaut définis à la fin du script.
    """
    if len(sys.argv) != 4:
        logger.error("Usage : python build_tool_dicts.py <csv_entretien> <csv_tools> <dossier_sortie>")
        sys.exit(1)

    entretien_csv = Path(sys.argv[1])
    tools_csv = Path(sys.argv[2])
    output_dir = Path(sys.argv[3])

    if not entretien_csv.exists():
        logger.error("Fichier introuvable: %s", entretien_csv)
        sys.exit(1)
    if not tools_csv.exists():
        logger.error("Fichier introuvable: %s", tools_csv)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    file_to_materiau, file_to_artisanat = load_entretien_file(entretien_csv)
    file_to_tools = load_tools_file(tools_csv)

    dict_materiau = count_tools_by_category(file_to_materiau, file_to_tools)
    dict_artisanat = count_tools_by_category(file_to_artisanat, file_to_tools)

    out_materiau = output_dir / "dict_outils_materiau.csv"
    out_artisanat = output_dir / "dict_outils_artisanat.csv"

    write_output_csv(out_materiau, "materiau", dict_materiau)
    write_output_csv(out_artisanat, "artisanat", dict_artisanat)

    logger.info("Fichier créé : %s", out_materiau)
    logger.info("Fichier créé : %s", out_artisanat)


if __name__ == "__main__":
    # Appel automatique des chemins voulus
    ENTRETIEN_PATH = "data/recap_entretien.csv"
    TOOLS_PATH = "data/processed_tool_comparaison_strict/cleaned_full_with_tools.csv"
    OUTPUT_PATH = "results/tool_comparaison"

    # Si arguments fournis, on les utilise, sinon on prend les chemins par défaut
    if len(sys.argv) == 1:
        sys.argv = ["", ENTRETIEN_PATH, TOOLS_PATH, OUTPUT_PATH]

    main()
