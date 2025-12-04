"""
csv_extract_tools.py

Ce module permet d'extraire automatiquement la liste des outils depuis la page Wikipedia
"Liste d'outils" et de générer un fichier CSV contenant ces outils.

Fonctionnalités :
- Récupération de la page Wikipedia.
- Extraction des noms d'outils à partir des éléments <li> contenant des liens.
- Nettoyage des noms (suppression des textes entre parenthèses).
- Filtrage des exceptions et limitation aux outils entre "Aérographe" et "Xylographe".
- Sauvegarde dans un CSV dans le dossier `data/` à la racine du projet.

Usage :
    python csv_extract_tools.py
"""

import csv
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# Liste des outils à exclure
EXCEPTION = ["verre", "chaîne", "tour", "coin", "bol", "plane", "niveau"]
# tas ? Outil mais aussi expression pour "beacoup"


def extract_tools():
    """
    Extrait la liste des outils depuis la page Wikipedia "Liste d'outils" et génère un CSV.

    Étapes :
    1. Récupère la page Wikipedia avec requests.
    2. Parse le HTML avec BeautifulSoup.
    3. Sélectionne les éléments <li> contenant des liens d'outils (<a title="...">).
    4. Nettoie les noms (supprime le texte entre parenthèses).
    5. Exclut les exceptions définies dans EXCEPTION.
    6. Ne conserve que les outils entre "Aérographe" et "Xylographe".
    7. Crée un fichier CSV "list_tool.csv" dans le dossier `data/` à la racine du projet.

    Aucun paramètre n'est requis. Le CSV généré contient une colonne avec le nom de chaque outil.
    """
    # URL de la page Wikipedia
    url = "https://fr.wikipedia.org/wiki/Liste_d'outils"

    # Récupération du contenu HTML
    headers = {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/117.0"}

    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Récupérer uniquement les <li> contenant un lien d'outil
    # Ne sélectionner que les liens <a> avec un attribut title (nom d’outil)
    tool_links = soup.select("li > a[title]")

    tools = []
    for a in tool_links:
        # Le vrai nom est dans l'attribut title
        text = a.get("title", "").strip()
        # On ne garde que les vrais noms (non vides)
        if text:
            # on veut supprimer les textes entre parenthèses
            if "(" in text:
                text = text[: text.index("(")].strip()
            # On exclut les exceptions
            if text.lower() not in EXCEPTION:
                tools.append(text)

    # On veut garder les mots entre Aérographe et Xylographe car le reste ne sont pas des outils
    start_index = None
    end_index = None
    for i, tool in enumerate(tools):
        if tool == "Aérographe":
            start_index = i
        elif tool == "Xylographe":
            end_index = i + 1  # Inclure Xylographe
            break
    if start_index is not None and end_index is not None:
        tools = tools[start_index:end_index]

    # Dossier de sortie : data/ à côté de src/
    output_dir = Path(__file__).resolve().parents[2] / "data"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "list_tool.csv"

    # Écriture CSV
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for t in tools:
            writer.writerow([t])

    print(f"Extraction terminée. CSV créé : {output_file}")


if __name__ == "__main__":
    extract_tools()
