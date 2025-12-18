"""
extract_tool_wiki.py

Ce module permet d'extraire automatiquement la liste des outils depuis la page Wikipedia
"Liste d'outils" et de générer un fichier CSV contenant ces outils.

Usage :
    python src/preprocessing/extract_tool_wiki.py
"""

import csv

import requests
from bs4 import BeautifulSoup

from utils.logger_config import setup_logger

# Liste des outils à exclure
EXCEPTION = ["verre", "chaîne", "tour", "coin", "bol", "plane", "niveau"]

logger = setup_logger(__name__, level="INFO")


def extract_tools(output_dir):
    """
    Extrait la liste des outils depuis la page Wikipedia "Liste d'outils" et génère un CSV.
    """
    url = "https://fr.wikipedia.org/wiki/Liste_d'outils"
    headers = {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/117.0"}

    logger.info("Début de l'extraction des outils depuis Wikipedia.")

    # Récupération du contenu HTML
    try:
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        logger.info("Page Wikipedia récupérée avec succès.")
    except requests.RequestException:
        logger.error("Échec de la récupération de la page Wikipedia.", exc_info=True)
        raise

    soup = BeautifulSoup(response.text, "html.parser")

    # Extraction des liens <a[title]>
    tool_links = soup.select("li > a[title]")
    logger.debug("Nombre d éléments <li> avec attribut title trouvés : %d", len(tool_links))

    tools = []
    for a in tool_links:
        text = a.get("title", "").strip()
        if text:
            # nettoyage des noms
            if "(" in text:
                text = text[: text.index("(")].strip()

            # filtrage des exceptions
            if text.lower() not in EXCEPTION:
                tools.append(text)

    logger.info("Nombre d' outils récupérés avant filtrage sur Aérographe → Xylographe : %d", len(tools))

    # Filtrage par plage alphabétique
    start_index = None
    end_index = None
    for i, tool in enumerate(tools):
        if tool == "Aérographe":
            start_index = i
        elif tool == "Xylographe":
            end_index = i + 1
            break

    if start_index is None or end_index is None:
        logger.warning("Impossible de trouver la plage Aérographe → Xylographe. Aucun outil sauvegardé.")
        return

    tools = tools[start_index:end_index]
    logger.info(" Nombre d' outils conservés après filtrage alphabétique : %d", len(tools))

    # Emplacement du dossier data/
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "list_tool_wiki.csv"

    # Écriture du fichier CSV
    try:
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for t in tools:
                writer.writerow([t])
        logger.info("CSV créé avec succès : %s", output_file)
    except Exception:
        logger.error("Erreur lors de l'écriture du fichier CSV.", exc_info=True)
        raise


if __name__ == "__main__":
    extract_tools("data")
