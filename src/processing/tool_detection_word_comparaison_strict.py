import csv
from pathlib import Path
from collections import Counter
import re


def load_tool_list(tool_csv_path):
    """Charge la liste des outils dans un CSV (une colonne, pas d'entête)."""
    tools = []
    with open(tool_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                tools.append(row[0].strip().lower())
    return tools


def detect_tools_in_text(text, tools):
    found = []
    for tool in tools:
        pattern = r"\b" + re.escape(tool) + r"\b"
        matches = re.findall(pattern, text)
        found.extend([tool] * len(matches))
    return found


def process_single_csv(tool_csv, input_csv, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    tools = load_tool_list(tool_csv)

    output_csv_path = output_dir / (input_csv.stem + "_with_tools.csv")
    dict_csv_path = output_dir / (input_csv.stem + "_tool_dict.csv")

    global_counter = Counter()

    with open(input_csv, newline="", encoding="utf-8") as fin, \
         open(output_csv_path, "w", newline="", encoding="utf-8") as fout:

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

    # Écrire le dictionnaire global et le trie par nombre d'occurence dans l'ordre décroissant
    global_counter = dict(sorted(global_counter.items(), key=lambda item: item[1], reverse=True))

    with open(dict_csv_path, "w", newline="", encoding="utf-8") as fdict:
        writer = csv.writer(fdict)
        writer.writerow(["tool", "count"])
        for tool, count in global_counter.items():
            writer.writerow([tool, count])        

    print(f" Fichier traité : {input_csv.name}")
    print(f"   Résultat : {output_csv_path}")
    print(f"   Dictionnaire : {dict_csv_path}")


def process_all_csvs():
    tool_csv = Path("data/list_tool.csv")

    base_dir = Path("data/processed")
    input_csvs = [
        base_dir / "cleaned_paragraph.csv",
        base_dir / "cleaned_sentence.csv",
        base_dir / "cleaned_full.csv",
    ]

    output_dir = Path("data/processed_tool_comparaison_strict")

    for csv_file in input_csvs:
        process_single_csv(tool_csv, csv_file, output_dir)


if __name__ == "__main__":
    process_all_csvs()
