import csv
import sys
from collections import defaultdict
from pathlib import Path

def load_entretien_file(path):
    """
    Charge le fichier entretien (Index,Nom Fichier,...,Matériau,Artisanat,...)
    Retourne deux dictionnaires :
        fichier -> matériau
        fichier -> artisanat
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
    Charge le fichier tools (filename,text,word_count,tools_found,tools_found_unique)
    Retourne un dict : fichier -> liste outils
    """
    file_to_tools = {}

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("filename", "").strip().replace(" ", "_").replace("_traitement_A","")
            tools_raw = row.get("tools_found_unique", "")
            tools = [t.strip() for t in tools_raw.split(",") if t.strip()]

            if filename:
                file_to_tools[filename] = tools

    return file_to_tools


def count_tools_by_category(mapping_file_to_category, mapping_file_to_tools):
    """
    mapping_file_to_category : dict fichier -> materiau ou artisanat
    mapping_file_to_tools : dict fichier -> liste d'outils
    Retourne un dict category -> dict outil -> count
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
    Écrit un CSV :
    category,outil,count
    """
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([category_name, "outil", "nombre_apparition"])

        for category, tool_dict in data_dict.items():
            for tool, count in tool_dict.items():
                writer.writerow([category, tool, count])


def main():
    if len(sys.argv) != 4:
        print("Usage : python build_tool_dicts.py <csv_entretien> <csv_tools> <dossier_sortie>")
        sys.exit(1)

    entretien_csv = Path(sys.argv[1])
    tools_csv = Path(sys.argv[2])
    output_dir = Path(sys.argv[3])

    if not entretien_csv.exists():
        print(f"Erreur : fichier introuvable : {entretien_csv}")
        sys.exit(1)
    if not tools_csv.exists():
        print(f"Erreur : fichier introuvable : {tools_csv}")
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

    print(f"Fichier créé : {out_materiau}")
    print(f"Fichier créé : {out_artisanat}")


if __name__ == "__main__":
    # Appel automatique des chemins voulus
    entretien_path = "data/recap_entretien.csv"
    tools_path = "data/processed_tool_comparaison_strict/cleaned_full_with_tools.csv"
    output_path = "results/tool_comparaison"

    # Si arguments fournis, on les utilise, sinon on prend les chemins par défaut
    if len(sys.argv) == 1:
        sys.argv = ["", entretien_path, tools_path, output_path]

    main()
