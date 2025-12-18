"""
Module pour détecter les outils dans des textes à partir d'une liste CSV d'outils.

Ce programme :
- charge une liste d'outils depuis un CSV (une colonne, pas d'en-tête),
- parcourt plusieurs fichiers CSV de textes nettoyés (par paragraphe, phrase ou entier),
- détecte les occurrences des outils dans chaque texte,
- écrit des CSV annotés avec les outils trouvés et leurs occurrences uniques,
- génère un dictionnaire global d'outils trié par fréquence.
"""

import csv
import re
from collections import Counter
from pathlib import Path

from utils.logger_config import setup_logger

logger = setup_logger(__name__, level="INFO")


def load_tool_list(tool_csv_path):
    """
    Charge la liste des outils depuis un CSV.

    Parameters:
    tool_csv_path (Path): chemin vers le fichier CSV contenant les outils.

    Returns:
    list: liste des outils en minuscules.
    """
    tools = []
    with open(tool_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                tools.append(row[0].strip().lower())
    return tools


def detect_tools_in_text(text, tools):
    """
    Détecte les outils présents dans un texte donné.

    Parameters:
    text (str): texte à analyser.
    tools (list): liste des outils à rechercher.

    Returns:
    list: outils trouvés, répétitions incluses.
    """
    found = []
    for tool in tools:
        pattern = r"\b" + re.escape(tool) + r"\b"
        matches = re.findall(pattern, text)
        found.extend([tool] * len(matches))
    return found


def process_single_csv(tool_csv, input_csv, output_dir):
    """
    Traite un CSV unique : détecte les outils et écrit le CSV annoté + dictionnaire global.

    Parameters:
    tool_csv (Path): chemin vers le CSV des outils.
    input_csv (Path): chemin vers le CSV à traiter.
    output_dir (Path): dossier de sortie pour les fichiers générés.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    tools = load_tool_list(tool_csv)

    output_csv_path = output_dir / (input_csv.stem + "_with_tools.csv")
    dict_csv_path = output_dir / (input_csv.stem + "_tool_dict.csv")

    global_counter = Counter()

    with (
        open(input_csv, newline="", encoding="utf-8") as fin,
        open(output_csv_path, "w", newline="", encoding="utf-8") as fout,
    ):
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames + ["tools_found", "tools_found_unique"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            # colonne texte peut être "texte" ou "text"
            text = row.get("texte", row.get("text", ""))

            found = detect_tools_in_text(text, tools)
            global_counter.update(found)

            row["tools_found"] = ", ".join(found)
            row["tools_found_unique"] = ", ".join(sorted(set(found)))

            writer.writerow(row)

    # Écrire le dictionnaire global et le trier par nombre d'occurence dans l'ordre décroissant
    global_counter = dict(sorted(global_counter.items(), key=lambda item: item[1], reverse=True))

    with open(dict_csv_path, "w", newline="", encoding="utf-8") as fdict:
        writer = csv.writer(fdict)
        writer.writerow(["tool", "count"])
        for tool, count in global_counter.items():
            writer.writerow([tool, count])

    logger.info("Fichier traité : %s", input_csv.name)
    logger.info("  Résultat : %s", output_csv_path)
    logger.info("  Dictionnaire : %s", dict_csv_path)


def process_all_csvs(  # pylint: disable=dangerous-default-value
    input_csvs=None,
    output_dir=Path("data/tool_detection"),
    tool_csv=None,
):
    """
    Traite tous les CSV de textes nettoyés : paragraphe, phrase et full.

    Par défaut, les fichiers lus sont dans `data/processed` et les résultats
    sont écrits dans `data/tool_detection`.
    """

    # Defaults set here to avoid mutable/default-eval issues
    if input_csvs is None:
        input_csvs = [
            Path("data/processed") / "cleaned_paragraph.csv",
            Path("data/processed") / "cleaned_sentence.csv",
            Path("data/processed") / "cleaned_full.csv",
        ]

    if tool_csv is None:
        tool_csv = Path(output_dir) / "list_tool_wiki.csv"

    for csv_file in input_csvs:
        logger.info("Processing input CSV: %s", csv_file)
        process_single_csv(tool_csv, csv_file, output_dir)


if __name__ == "__main__":
    process_all_csvs()
