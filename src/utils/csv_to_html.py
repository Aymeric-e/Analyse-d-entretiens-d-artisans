import os
import sys
import pandas as pd

def csv_to_html(csv_path, output_dir):
    # Vérification CSV
    if not os.path.isfile(csv_path):
        print(f"Erreur : le fichier '{csv_path}' est introuvable.")
        sys.exit(1)

    # Récupère juste le nom sans extension
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_path = os.path.join(output_dir, base_name + ".html")

    # Lit le CSV avec pandas
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Impossible de lire le CSV : {e}")
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

    print(f"Fichier HTML généré : {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage : python csv_to_html.py <fichier.csv> <chemin_output>")
        sys.exit(1)

    csv_file = sys.argv[1]
    output_dir = sys.argv[2]

    csv_to_html(csv_file, output_dir)
