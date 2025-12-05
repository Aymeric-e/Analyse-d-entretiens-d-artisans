"""
Generate HTML report with verbalization difficulty color-coded visualization.
Merges annotated phrases with interview transcriptions and creates interactive HTML report with table view.

"""

import argparse
import colorsys
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

from utils.logger_config import setup_logger

logger = setup_logger(__name__, level="INFO")


class DifficultyReportGenerator:
    """Generate HTML report with color-coded verbalization difficulty"""

    def __init__(self):
        self.phrases_df = None
        self.interviews_df = None
        self.merged_data = None
        self.difficulty_col = None

    def load_data(self, phrases_csv: Path, interviews_csv: Path, difficulty_col: str) -> None:
        """Load both CSV files"""
        logger.info("Chargement des phrases annotées depuis %s...", phrases_csv)
        self.phrases_df = pd.read_csv(phrases_csv, sep=";")
        logger.info("Chargement des transcriptions d'entretiens depuis %s...", interviews_csv)
        self.interviews_df = pd.read_csv(interviews_csv, sep=",")

        # Validate columns
        if "filename" not in self.phrases_df.columns or "text" not in self.phrases_df.columns:
            raise ValueError("Le CSV des phrases doit contenir les colonnes 'filename' et 'text'")

        if difficulty_col not in self.phrases_df.columns:
            raise ValueError(f"Colonne '{difficulty_col}' non trouvée dans le CSV des phrases")

        if "filename" not in self.interviews_df.columns:
            raise ValueError("Le CSV des entretiens doit contenir la colonne 'filename'")

        self.difficulty_col = difficulty_col

        logger.info("Phrases chargées: %d", len(self.phrases_df))
        logger.info("Entretiens chargés: %d", len(self.interviews_df))

    def difficulty_to_rgb(self, difficulty: float) -> Tuple[int, int, int]:
        """Convert difficulty score (0-10) to RGB color (green to red)"""
        normalized = max(0, min(1, difficulty / 10.0))
        hue = (1 - normalized) * 120 / 360
        saturation = 0.5
        lightness = 0.5
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        return int(r * 255), int(g * 255), int(b * 255)

    def rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB tuple to hex color"""
        r, g, b = rgb
        return f"#{r:02x}{g:02x}{b:02x}"

    def merge_data(self) -> None:
        """Merge phrases with interview metadata"""
        logger.info("Merge des données...")

        self.merged_data = {}

        for _, row in self.interviews_df.iterrows():
            filename = row["filename"]
            self.merged_data[filename] = {"filename": filename, "phrases": []}

            for col in self.interviews_df.columns:
                if col != "filename":
                    self.merged_data[filename][col] = row[col]

        for _, row in self.phrases_df.iterrows():
            filename = row["filename"]

            if filename in self.merged_data:
                self.merged_data[filename]["phrases"].append({"text": row["text"], "difficulty": float(row[self.difficulty_col])})

        logger.info("Données mergées: %d entretiens", len(self.merged_data))

    def colorize_text_with_tooltip(self, text: str, difficulty: float) -> str:
        """Wrap text with color span and tooltip on hover"""
        rgb = self.difficulty_to_rgb(difficulty)
        hex_color = self.rgb_to_hex(rgb)
        tooltip = f"Difficulté: {difficulty:.2f}/10"

        return f'<span style="background-color: {hex_color}; padding: 2px 4px; border-radius: 3px; cursor: help;" \
                       title="{tooltip}">{text}</span>'

    def generate_html(self) -> str:
        """Generate complete HTML report"""
        css_style = """
    <style>
    body { 
        font-family: Arial, Helvetica, sans-serif; 
        background: #f7f7fb; 
        color: #222; 
        padding: 20px; 
    }
    .container { 
        max-width: 1400px; 
        margin: auto; 
    }
    .header { 
        margin-bottom: 30px; 
    }
    .header h1 { 
        color: #263238; 
        margin: 0 0 10px 0; 
    }
    .header p { 
        color: #555; 
        margin: 0; 
    }
    .scale-info { 
        background: #e3f2fd; 
        border-left: 4px solid #2196F3; 
        padding: 12px 16px; 
        margin-bottom: 20px; 
        border-radius: 3px; 
    }
    .scale-info h3 { 
        margin: 0 0 8px 0; 
        color: #1976D2; 
    }
    .scale-info p { 
        margin: 0; 
        color: #555; 
        font-size: 0.95em; 
    }
    .table-container {
        background: #fff;
        border-radius: 6px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        overflow: hidden;
    }
    table { 
        border-collapse: collapse; 
        width: 100%; 
        background: #fff; 
    }
    th, td { 
        padding: 12px 16px; 
        border-bottom: 1px solid #eee; 
        text-align: left; 
        vertical-align: top;
    }
    th { 
        background: #263238; 
        color: #fff; 
        font-weight: 600; 
        position: sticky; 
        top: 0;
        z-index: 10;
    }
    tr:nth-child(even) td { 
        background: #fbfcff; 
    }
    tr:hover td {
        background: #f0f4ff;
    }
    .filename-col {
        font-weight: 600;
        color: #263238;
        min-width: 200px;
        max-width: 250px;
        word-break: break-word;
    }
    .text-col {
        line-height: 1.6;
        color: #333;
    }
    span[title] {
        cursor: help;
        text-decoration: underline dotted rgba(0,0,0,0.2);
    }
    .footer { 
        margin-top: 40px; 
        padding-top: 20px; 
        border-top: 1px solid #eee; 
        color: #888; 
        font-size: 0.9em; 
    }
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
            <p><em>Survolez les phrases colorées pour voir la note de difficulté exacte</em></p>
        </div>
        
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Entretien</th>
                        <th>Phrases</th>
                    </tr>
                </thead>
                <tbody>
"""

        # Add each interview as a row
        for filename in sorted(self.merged_data.keys()):
            interview = self.merged_data[filename]
            phrases = interview.get("phrases", [])

            # Combine all phrases with colored spans
            if phrases:
                colored_phrases = []
                for phrase_data in phrases:
                    text = phrase_data["text"]
                    difficulty = phrase_data["difficulty"]
                    colored_text = self.colorize_text_with_tooltip(text, difficulty)
                    colored_phrases.append(colored_text)

                combined_text = " ".join(colored_phrases)
            else:
                combined_text = '<span style="color: #999; font-style: italic;">Aucune phrase annotée pour cet entretien</span>'

            html += f"""
                <tr>
                    <td class="filename-col">{filename}</td>
                    <td class="text-col">{combined_text}</td>
                </tr>
"""

        html += """
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>Rapport généré automatiquement</p>
        </div>
    </div>
</body>
</html>
"""

        return html

    def save_html(self, output_path: Path) -> None:
        """Save HTML report to file"""
        logger.info("Génération du rapport HTML...")

        html_content = self.generate_html()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info("Rapport sauvegardé: %s", output_path)

    def run(
        self,
        phrases_csv: Path,
        interviews_csv: Path,
        output_html: Path,
        difficulty_col: str = "difficulté_verbalisation",
    ) -> None:
        """Complete pipeline"""
        logger.info("Génération: Rapport HTML Difficulté de Verbalisation")

        self.load_data(phrases_csv, interviews_csv, difficulty_col)
        self.merge_data()
        self.save_html(output_html)

        logger.info("Rapport généré avec succès")


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

    if not args.phrases.exists():
        logger.error("ERREUR: Fichier introuvable: %s", args.phrases)
        sys.exit(1)

    if not args.interviews.exists():
        logger.error("ERREUR: Fichier introuvable: %s", args.interviews)
        sys.exit(1)

    generator = DifficultyReportGenerator()
    generator.run(args.phrases, args.interviews, args.output, args.difficulty_col)


if __name__ == "__main__":
    main()
