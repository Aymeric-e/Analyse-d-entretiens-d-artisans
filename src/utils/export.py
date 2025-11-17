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

    # CSS pour un rendu propre
    css_style = """
    <style>
        body { font-family: Arial, sans-serif; background: #f5f5f5; padding: 20px; }
        table { border-collapse: collapse; width: 100%; background: white; }
        th { background: #4CAF50; color: white; padding: 10px; }
        td { padding: 8px; border-bottom: 1px solid #ddd; }
        tr:nth-child(even) { background: #f2f2f2; }
        tr:hover { background: #ddd; }
        .container { max-width: 1100px; margin: auto; }
        h2 { text-align: center; }
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
