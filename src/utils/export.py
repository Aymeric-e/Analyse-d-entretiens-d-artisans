import pandas as pd
from pathlib import Path

def export_html(csv_file: Path, output_html: Path):
    """Exporter en HTML lisible"""
    df = pd.read_csv(csv_file)

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Entretiens d'artisans nettoyés</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-bottom: 30px;
        }}
        .stat-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-box h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            opacity: 0.9;
        }}
        .stat-box .number {{
            font-size: 28px;
            font-weight: bold;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        thead {{
            background-color: #34495e;
            color: white;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
        }}
        tbody tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        tbody tr:hover {{
            background-color: #f0f7ff;
        }}
        .file-name {{
            font-weight: 600;
            color: #2980b9;
        }}
        .word-count {{
            text-align: center;
            color: #7f8c8d;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Entretiens d'artisans - Données nettoyées</h1>
        <div class="stats">
            <div class="stat-box"><h3>Total de lignes</h3><div class="number">{len(df)}</div></div>
            <div class="stat-box"><h3>Fichiers</h3><div class="number">{df['filename'].nunique()}</div></div>
            <div class="stat-box"><h3>Mots/ligne (moy)</h3><div class="number">{df['word_count'].mean():.0f}</div></div>
        </div>
        <table>
            <thead><tr><th>Fichier</th><th>Texte</th><th>Mots</th></tr></thead>
            <tbody>
"""
    for _, row in df.iterrows():
        html += f"<tr><td class='file-name'>{row['filename']}</td><td>{row['text']}</td><td class='word-count'>{row['word_count']}</td></tr>\n"

    html += """</tbody></table></div></body></html>"""
    output_html.write_text(html, encoding="utf-8")
    print(f"HTML créé : {output_html}")


def export_txt(csv_file: Path, output_txt: Path):
    """Exporter en TXT lisible"""
    df = pd.read_csv(csv_file)

    txt = "=" * 80 + "\n"
    txt += "ENTRETIENS D'ARTISANS - DONNÉES NETTOYÉES\n"
    txt += "=" * 80 + "\n\n"
    txt += f"Total de lignes: {len(df)}\n"
    txt += f"Fichiers: {df['filename'].nunique()}\n"
    txt += f"Mots/ligne (moy): {df['word_count'].mean():.1f}\n\n"

    for filename in sorted(df['filename'].unique()):
        df_file = df[df['filename'] == filename]
        txt += "\n" + "─" * 80 + "\n"
        txt += f"FICHIER: {filename}\n"
        txt += f"Lignes: {len(df_file)}\n"
        txt += "─" * 80 + "\n\n"
        for line_num, (_, row) in enumerate(df_file.iterrows(), 1):
            txt += f"{line_num}. {row['text']}\n"
            txt += f"   ({row['word_count']} mots)\n\n"

    output_txt.write_text(txt, encoding="utf-8")
    print(f"TXT créé : {output_txt}")
