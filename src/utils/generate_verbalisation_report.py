"""
Generate HTML report with verbalization difficulty color-coded visualization.
Merges annotated phrases with interview transcriptions and creates interactive HTML report.

"""

import argparse
import colorsys
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd


class DifficultyReportGenerator:
    """Generate HTML report with color-coded verbalization difficulty"""

    def __init__(self):
        self.phrases_df = None
        self.interviews_df = None
        self.merged_data = None
        self.difficulty_col = None

    def load_data(self, phrases_csv: Path, interviews_csv: Path, difficulty_col: str) -> None:
        """Load both CSV files"""
        print(f"Chargement des phrases annotées depuis {phrases_csv}...")
        self.phrases_df = pd.read_csv(phrases_csv, sep=";")

        print(f"Chargement des transcriptions d'entretiens depuis {interviews_csv}...")
        self.interviews_df = pd.read_csv(interviews_csv, sep=",")

        # Validate columns
        if "filename" not in self.phrases_df.columns or "text" not in self.phrases_df.columns:
            raise ValueError("Le CSV des phrases doit contenir les colonnes 'filename' et 'text'")

        if difficulty_col not in self.phrases_df.columns:
            raise ValueError(f"Colonne '{difficulty_col}' non trouvée dans le CSV des phrases")

        if "filename" not in self.interviews_df.columns:
            raise ValueError("Le CSV des entretiens doit contenir la colonne 'filename'")

        self.difficulty_col = difficulty_col

        print(f"Phrases chargées: {len(self.phrases_df)}")
        print(f"Entretiens chargés: {len(self.interviews_df)}")

    def difficulty_to_rgb(self, difficulty: float) -> Tuple[int, int, int]:
        """Convert difficulty score (0-10) to RGB color (green to red)"""
        # Normalize to 0-1
        normalized = max(0, min(1, difficulty / 10.0))

        # Green (0) to Red (1): HSL color space
        # Start at green (120°), end at red (0°)
        hue = (1 - normalized) * 120 / 360  # Convert degrees to 0-1
        saturation = 0.6
        lightness = 0.5

        # Convert HSL to RGB
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)

        return int(r * 255), int(g * 255), int(b * 255)

    def rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB tuple to hex color"""
        r, g, b = rgb
        return f"#{r:02x}{g:02x}{b:02x}"

    def merge_data(self) -> None:
        """Merge phrases with interview metadata"""
        print("\nMerge des données...")

        # Créer un dictionnaire pour chaque filename avec ses phrases
        self.merged_data = {}

        for _, row in self.interviews_df.iterrows():
            filename = row["filename"]
            self.merged_data[filename] = {"filename": filename, "phrases": []}

            # Ajouter colonnes supplémentaires de l'interview si présentes
            for col in self.interviews_df.columns:
                if col != "filename":
                    self.merged_data[filename][col] = row[col]

        # Ajouter les phrases avec leurs difficultés
        for _, row in self.phrases_df.iterrows():
            filename = row["filename"]

            if filename in self.merged_data:
                self.merged_data[filename]["phrases"].append({"text": row["text"], "difficulty": float(row[self.difficulty_col])})

        print(f"Données mergées: {len(self.merged_data)} entretiens")

    def colorize_text(self, text: str, difficulty: float) -> str:
        """Wrap text with color span based on difficulty"""
        rgb = self.difficulty_to_rgb(difficulty)
        hex_color = self.rgb_to_hex(rgb)

        return f'<span style="background-color: {hex_color}; padding: 2px 4px; border-radius: 3px;">{text}</span>'

    def generate_html(self) -> str:
        """Generate complete HTML report"""
        css_style = """
    <style>
    body { font-family: Arial, Helvetica, sans-serif; background:#f7f7fb; color:#222; padding:20px; }
    .container { max-width:1200px; margin:auto; }
    .header { margin-bottom:30px; }
    .header h1 { color:#263238; margin:0 0 10px 0; }
    .header p { color:#555; margin:0; }
    .interview-section { background:#fff; margin-bottom:20px; padding:20px; border-radius:6px; box-shadow:0 2px 6px rgba(0,0,0,0.06); }
    .interview-title { background:#263238; color:#fff; padding:12px 16px; margin:-20px -20px 15px -20px; border-radius:6px 6px 0 0; font-size:1.1em; font-weight:600; }
    .interview-content { line-height:1.8; color:#333; }
    .phrase { margin-bottom:12px; padding:8px; background:#fafbff; border-left:3px solid #e0e0e0; border-radius:3px; }
    .difficulty-badge { display:inline-block; background:#263238; color:#fff; padding:3px 8px; border-radius:3px; font-size:0.85em; margin-left:8px; }
    .scale-info { background:#e3f2fd; border-left:4px solid #2196F3; padding:12px 16px; margin-bottom:20px; border-radius:3px; }
    .scale-info h3 { margin:0 0 8px 0; color:#1976D2; }
    .scale-info p { margin:0; color:#555; font-size:0.95em; }
    table { border-collapse: collapse; width:100%; background:#fff; margin-top:20px; box-shadow:0 2px 6px rgba(0,0,0,0.06); border-radius:6px; overflow:hidden; }
    th, td { padding:10px 12px; border-bottom:1px solid #eee; text-align:left; vertical-align:top; }
    th { background:#263238; color:#fff; font-weight:600; position:sticky; top:0; }
    tr:nth-child(even) td { background:#fbfcff; }
    .code { font-family:"Courier New", monospace; font-size:0.95em; color:#333; }
    .small { font-size:0.9em; color:#555; }
    .footer { margin-top:40px; padding-top:20px; border-top:1px solid #eee; color:#888; font-size:0.9em; }
    </style>
    """

        html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport Difficulté de Verbalisation</title>
    {css_style}
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Rapport d'Analyse: Difficulté de Verbalisation</h1>
            <p>Visualisation des difficultés de verbalisation par entretien et phrase</p>
        </div>
        
        <div class="scale-info">
            <h3>Échelle de couleurs</h3>
            <p>🟢 <strong>Vert (0/10)</strong>: Aucune difficulté | 🟡 <strong>Jaune (5/10)</strong>: Difficulté modérée | 🔴 <strong>Rouge (10/10)</strong>: Difficulté importante</p>
        </div>
"""

        # Add each interview section
        for filename in sorted(self.merged_data.keys()):
            interview = self.merged_data[filename]
            phrases = interview.get("phrases", [])

            html += f"""
        <div class="interview-section">
            <div class="interview-title">{filename}</div>
            <div class="interview-content">
"""

            if phrases:
                for phrase_data in phrases:
                    text = phrase_data["text"]
                    difficulty = phrase_data["difficulty"]
                    colored_text = self.colorize_text(text, difficulty)

                    html += f"""
                <div class="phrase">
                    {colored_text}
                    <span class="difficulty-badge">{difficulty:.2f}/10</span>
                </div>
"""
            else:
                html += """
                <div class="phrase" style="color:#999;">
                    Aucune phrase annotée pour cet entretien
                </div>
"""

            html += """
            </div>
        </div>
"""

        # Add statistics table
        html += self._generate_statistics_table()

        html += """
        <div class="footer">
            <p>Rapport généré automatiquement</p>
        </div>
    </div>
</body>
</html>
"""

        return html

    def _generate_statistics_table(self) -> str:
        """Generate statistics table by interview"""
        html = "<h2 style='margin-top:40px; color:#263238;'>Statistiques par entretien</h2>"
        html += """
        <table>
            <thead>
                <tr>
                    <th>Entretien</th>
                    <th>Nb phrases</th>
                    <th>Difficulté moyenne</th>
                    <th>Min</th>
                    <th>Max</th>
                </tr>
            </thead>
            <tbody>
"""

        for filename in sorted(self.merged_data.keys()):
            interview = self.merged_data[filename]
            phrases = interview.get("phrases", [])

            if phrases:
                difficulties = [p["difficulty"] for p in phrases]
                avg_diff = sum(difficulties) / len(difficulties)
                min_diff = min(difficulties)
                max_diff = max(difficulties)

                html += f"""
                <tr>
                    <td><strong>{filename}</strong></td>
                    <td>{len(phrases)}</td>
                    <td>{avg_diff:.2f}/10</td>
                    <td>{min_diff:.2f}/10</td>
                    <td>{max_diff:.2f}/10</td>
                </tr>
"""

        html += """
            </tbody>
        </table>
"""

        return html

    def save_html(self, output_path: Path) -> None:
        """Save HTML report to file"""
        print("\nGénération du rapport HTML...")

        html_content = self.generate_html()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"Rapport sauvegardé: {output_path}")

    def run(
        self,
        phrases_csv: Path,
        interviews_csv: Path,
        output_html: Path,
        difficulty_col: str = "difficulté_verbalisation",
    ) -> None:
        """Complete pipeline"""

        print("GÉNÉRATION: Rapport HTML Difficulté de Verbalisation")

        self.load_data(phrases_csv, interviews_csv, difficulty_col)
        self.merge_data()
        self.save_html(output_html)

        print("Rapport généré avec succès!")


def main():
    """Main function to parse arguments and run report generation"""
    parser = argparse.ArgumentParser(description="Générer un rapport HTML avec visualisation des difficultés de verbalisation")
    parser.add_argument(
        "--phrases",
        type=Path,
        required=True,
        help="Chemin du CSV avec phrases annotées (colonnes requises: filename, text, [difficulty_col])",
    )
    parser.add_argument(
        "--interviews",
        type=Path,
        required=True,
        help="Chemin du CSV avec métadonnées des entretiens (colonne requise: filename)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Chemin du fichier HTML de sortie")
    parser.add_argument(
        "--difficulty-col",
        type=str,
        default="difficulté_verbalisation",
        help="Nom de la colonne contenant les notes de difficulté (défaut: difficulté_verbalisation)",
    )

    args = parser.parse_args()

    # Validate input files
    if not args.phrases.exists():
        print(f"ERREUR: Fichier introuvable: {args.phrases}")
        sys.exit(1)

    if not args.interviews.exists():
        print(f"ERREUR: Fichier introuvable: {args.interviews}")
        sys.exit(1)

    # Run generation
    generator = DifficultyReportGenerator()
    generator.run(args.phrases, args.interviews, args.output, args.difficulty_col)


if __name__ == "__main__":
    main()
