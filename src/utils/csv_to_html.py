"""
csv_to_html_converter.py

Convertit un fichier CSV en une page HTML stylée.

Fonctionnalités principales :
- Lit un CSV avec pandas (en tenant compte du séparateur choisi)
- Génère un tableau HTML reprenant toutes les colonnes
- Applique un style CSS simple pour rendre le tableau lisible et agréable
- Crée automatiquement le dossier de sortie si nécessaire
- Gère les erreurs de lecture du CSV ou de chemin invalide

Usage :
    python csv_to_html_converter.py --input <fichier.csv> --separator <sep> [--output-dir <dossier_html>]

Arguments :
    --input : Chemin du fichier CSV à convertir (obligatoire)
    --separator : Séparateur utilisé dans le CSV (obligatoire, ex: ',' ou ';')
    --output-dir : Dossier où générer le HTML (optionnel, par défaut le dossier du CSV d'entrée)

Exemple :
    python csv_to_html_converter.py --input data/my_data.csv --separator , --output-dir results/html
"""

import argparse
import os
import sys

import pandas as pd

from utils.logger_config import setup_logger

logger = setup_logger(__name__, level="INFO")


def csv_to_html(csv_path, output_dir, separator):
    """
    Convertit un fichier CSV en une page HTML avec style.

    Chaque colonne du CSV devient une colonne du tableau HTML.
    Le style reprend celui utilisé dans csv_to_html_highlight_tool.py.

    Args:
        csv_path (str): Chemin vers le fichier CSV à convertir.
        output_dir (str): Dossier où sera généré le fichier HTML.
        separator (str): Séparateur utilisé dans le CSV (ex: ',' ou ';').

    Raises:
        SystemExit: Si le fichier CSV est introuvable ou ne peut pas être lu.

    Outputs:
        Fichier HTML dans `output_dir` portant le même nom que le CSV.
    """
    # Vérification CSV
    if not os.path.isfile(csv_path):
        logger.error("Erreur : le fichier '%s' est introuvable.", csv_path)
        sys.exit(1)

    # Récupère juste le nom sans extension
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_path = os.path.join(output_dir, base_name + ".html")

    # Lit le CSV avec pandas
    try:
        df = pd.read_csv(csv_path, sep=separator)
    except Exception:  # pylint: disable=broad-except
        logger.exception("Impossible de lire le CSV: %s", csv_path)
        sys.exit(1)

    # Nouveau style HTML (reprend le style du premier script)
    css_style = """
    <style>
    body { font-family: Arial, Helvetica, sans-serif; background:#f7f7fb; color:#222; padding:20px; }
    .container { max-width:1200px; margin:auto; }
    table { border-collapse: collapse; width:100%; background:#fff; box-shadow:0 2px 6px rgba(0,0,0,0.06); }
    th, td { padding:10px 12px; border-bottom:1px solid #eee; text-align:left; vertical-align:top; }
    th { background:#263238; color:#fff; font-weight:600; position:sticky; top:0; }
    tr:nth-child(even) td { background:#fbfcff; }
    .code { font-family:"Courier New", monospace; font-size:0.95em; color:#333; }
    .small { font-size:0.9em; color:#555; }
    </style>
    """

    html_content = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>{base_name}</title>
        {css_style}
    </head>
    <body>
        <div class="container">
            <h2>Table : {base_name}</h2>
            {df.to_html(index=False, escape=False)}
        </div>
    </body>
    </html>
    """

    # Crée dossier sortie si inexistant
    os.makedirs(output_dir, exist_ok=True)

    # Écriture fichier HTML
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info("Fichier HTML généré : %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convertir un fichier CSV en une page HTML jolie")
    parser.add_argument("--input", required=True, help="Chemin du fichier CSV à convertir")
    parser.add_argument(
        "--separator",
        required=True,
        help="Séparateur utilisé dans le CSV (ex: ; ou ,), par défaut ,",
        default=",",
    )
    parser.add_argument(
        "--output-dir",
        help="Dossier contenant le fichier html de sorti, par défault le même dossier que le CSV d'entrée",
        default=None,
    )

    args = parser.parse_args()

    logger.info("CONVERSION CSV -> HTML")
    logger.info("Fichier d'entrée : %s", args.input)

    output_folder = args.output_dir if args.output_dir else args.input.rsplit("/", 1)[0]

    csv_to_html(args.input, output_folder, args.separator)
