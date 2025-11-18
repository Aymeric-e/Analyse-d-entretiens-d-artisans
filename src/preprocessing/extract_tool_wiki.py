import requests
from bs4 import BeautifulSoup
import csv
from pathlib import Path

#Liste des outils à exclure
EXCEPTION = ["verre", "chaîne", "tour", "coin","bol", "plane", "niveau"]
#tas ? Outil mais aussi expression pour "beacoup"

def extract_tools():
    # URL de la page Wikipedia
    url = "https://fr.wikipedia.org/wiki/Liste_d'outils"

    # Récupération du contenu HTML  
    headers = {
    "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/117.0"
    }   

    response = requests.get(url,headers=headers)
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
            #on veut supprimer les textes entre parenthèses
            if "(" in text:
                text = text[:text.index("(")].strip()
            #On exclut les exceptions
            if text.lower() not in EXCEPTION:
                tools.append(text)

    #On veut garder les mots entre Aérographe et Xylographe car le reste ne sont pas des outils
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
