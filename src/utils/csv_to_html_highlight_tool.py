#!/usr/bin/env python3
"""
csv_to_html_highlight_tool.py

Lit un fichier CSV avec :
    filename,text,word_count,tools_found,tools_found_unique

Produit un fichier HTML où les outils de tools_found_unique
sont surlignés dans le texte.

Usage :
    python csv_to_html_highlight_tool.py <fichier.csv> <dossier_output>
"""

import csv
import html
import re
import sys
from pathlib import Path

from utils.logger_config import setup_logger

logger = setup_logger(__name__, level="INFO")

HIGHLIGHT_TEMPLATE = '<mark style="background:#fff59d;padding:0 2px;border-radius:3px;"><strong>{}</strong></mark>'

HTML_PAGE_TEMPLATE = """<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8"/>
<title>Export highlight</title>
<style>
body {{ font-family: Arial, Helvetica, sans-serif; background:#f7f7fb; color:#222; padding:20px; }}
.container {{ max-width:1200px; margin:auto; }}
table {{ border-collapse: collapse; width:100%; background:#fff; box-shadow:0 2px 6px rgba(0,0,0,0.06); }}
th, td {{ padding:10px 12px; border-bottom:1px solid #eee; text-align:left; vertical-align:top; }}
th {{ background:#263238; color:#fff; font-weight:600; position:sticky; top:0; }}
tr:nth-child(even) td {{ background:#fbfcff; }}
.code {{ font-family:"Courier New", monospace; font-size:0.95em; color:#333; }}
.small {{ font-size:0.9em; color:#555; }}
</style>
</head>
<body>
<div class="container">
<h2>Export highlight – {title}</h2>
<p class="small">Lignes : {rows_count}. Les outils présents dans <code>tools_found_unique</code> sont surlignés dans le texte.</p>
<table>
<thead>
<tr>
{thead}
</tr>
</thead>
<tbody>
{tbody}
</tbody>
</table>
</div>
</body>
</html>
"""


def build_pattern_for_tools(tools):
    """
    Construit un motif regex pour détecter tous les outils dans une liste.

    Args:
        tools (list[str]): Liste d'outils à rechercher dans le texte.

    Returns:
        re.Pattern | None: Objet regex compilé pouvant être utilisé pour
        rechercher les outils, ou None si la liste est vide.
    """
    cleaned = [t.strip() for t in tools if t.strip()]
    if not cleaned:
        return None
    cleaned.sort(key=len, reverse=True)
    escaped = [re.escape(t) for t in cleaned]
    pattern = r"\b(?:" + "|".join(escaped) + r")\b"
    return re.compile(pattern)


def highlight_text_with_tools(text, tools):
    """
    Surligne les occurrences des outils dans un texte donné.

    Args:
        text (str): Texte dans lequel surligner les outils.
        tools (list[str]): Liste d'outils à surligner.

    Returns:
        str: Texte HTML avec les outils surlignés.
    """
    if not text:
        return ""
    if not tools:
        return html.escape(text)

    pattern = build_pattern_for_tools(tools)
    if pattern is None:
        return html.escape(text)

    parts = []
    last_pos = 0

    for match in pattern.finditer(text):
        start, end = match.start(), match.end()
        parts.append(html.escape(text[last_pos:start]))
        parts.append(HIGHLIGHT_TEMPLATE.format(html.escape(text[start:end])))
        last_pos = end

    parts.append(html.escape(text[last_pos:]))
    return "".join(parts)


def process_csv_to_html(input_csv_path: Path, output_folder: Path):
    """
    Lit un fichier CSV et génère un fichier HTML avec les outils surlignés.

    Args:
        input_csv_path (Path): Chemin vers le fichier CSV d'entrée.
        output_folder (Path): Dossier où enregistrer le fichier HTML.

    Returns:
        None
    """
    rows = []
    with input_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        logger.warning("Aucun contenu dans le CSV: %s", input_csv_path)
        return

    expected_cols = ["filename", "text", "word_count", "tools_found", "tools_found_unique"]
    columns = [c for c in expected_cols if c in rows[0]]

    tbody = []
    for row in rows:
        tools_list = [t.strip() for t in (row.get("tools_found_unique") or "").split(",") if t.strip()]
        highlighted = highlight_text_with_tools(row.get("text", ""), tools_list)

        tds = []
        for col in columns:
            if col == "text":
                tds.append(f"<td>{highlighted}</td>")
            else:
                tds.append(f"<td class='code'>{html.escape(row.get(col, ''))}</td>")

        tbody.append("<tr>" + "".join(tds) + "</tr>")

    thead_html = "".join(f"<th>{html.escape(col)}</th>" for col in columns)

    html_page = HTML_PAGE_TEMPLATE.format(
        title=input_csv_path.name,
        rows_count=len(rows),
        thead=thead_html,
        tbody="\n".join(tbody),
    )

    output_folder.mkdir(parents=True, exist_ok=True)

    output_html_path = output_folder / (input_csv_path.stem + "_highlight.html")

    with output_html_path.open("w", encoding="utf-8") as f:
        f.write(html_page)

    logger.info("Fichier HTML généré : %s", output_html_path)


def main():
    """
    Point d'entrée du script. Vérifie les arguments et lance le traitement CSV → HTML.

    Args:
        None

    Returns:
        None
    """
    import argparse

    parser = argparse.ArgumentParser(description="Générer un HTML avec surlignage des outils à partir d'un CSV")
    parser.add_argument("--input", type=Path, required=True, help="Fichier CSV d'entrée")
    parser.add_argument("--output-dir", type=Path, required=True, help="Dossier de sortie pour le HTML")

    args = parser.parse_args()

    if not args.input.exists():
        logger.error("Erreur : fichier introuvable : %s", args.input)
        sys.exit(1)

    process_csv_to_html(args.input, args.output_dir)


if __name__ == "__main__":
    main()
