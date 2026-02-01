"""
InterviewCleaner : Nettoyage et traitement des entretiens au format .docx

Ce programme permet de :
- Extraire le texte des fichiers .docx d'entretiens.
- Nettoyer le texte des mentions de [Interviewe], [Interviewé], [Chercheur]
  et des indications de temps ou symboles inutiles.
- Produire des versions nettoyées du texte selon trois niveaux de segmentation :
    - paragraph : chaque paragraphe est une ligne (par défaut)
    - sentence  : chaque phrase est une ligne
    - full      : tout le texte de l’entretien est concaténé en un seul bloc
- Exporter le résultat dans un fichier CSV avec les colonnes :
    - filename : nom du fichier source
    - text     : texte nettoyé
    - word_count : nombre de mots dans le texte

Exemple d'utilisation :
    cleaner = InterviewCleaner()
    df_full = cleaner.batch_process(Path("data/raw"), Path("data/processed/cleaned_full.csv"), mode="full")
    df_paragraph = cleaner.batch_process(Path("data/raw"), Path("data/processed/cleaned_paragraph.csv"), mode="paragraph")
    df_sentence = cleaner.batch_process(Path("data/raw"), Path("data/processed/cleaned_sentence.csv"), mode="sentence")
"""

import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
from docx import Document
from tqdm import tqdm

from utils.logger_config import setup_logger

logger = setup_logger(__name__, level="INFO")


class InterviewCleaner:
    """Nettoie les entretiens (3 niveaux de segmentation possibles)"""

    def clean_artisan_text(self, text: str) -> str:
        """Nettoyer une ligne de parole d'artisan"""
        text = re.sub(r"\[(Interviewe|Interviewé)\]", "", text)
        text = re.sub(r"\(\s*\d+:\d+\s*(?:–|-|‐)\s*\d+:\d+\s*\)", "", text)
        text = re.sub(r"\(\s*\d+:\d+\s*\)", "", text)
        text = re.sub(r"^\s*(?:–|-|‐)\s*", "", text)
        text = re.sub(r"^\s*\)\s*$", "", text)
        text = re.sub(r"^\s*\(\s*$", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def process_file(self, file_path: Path) -> List[Dict]:
        """
        Traiter un fichier .docx et extraire les textes nettoyés par paragraphe (niveau de base).
        """
        doc = Document(file_path)
        cleaned_lines = []
        filename = file_path.stem
        current_interviewee = False

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            if "[Chercheur]" in text:
                current_interviewee = False
                continue

            if "[Interviewe]" in text or "[Interviewé]" in text:
                current_interviewee = True
                cleaned = self.clean_artisan_text(text)
            elif current_interviewee:
                cleaned = self.clean_artisan_text(text)
            else:
                continue

            if cleaned and len(cleaned.split()) >= 2:
                cleaned_lines.append({"filename": filename, "text": cleaned, "word_count": len(cleaned.split())})

        return cleaned_lines

    def batch_process(self, input_dir: Path, output_csv: Path, mode: str = "paragraph"):
        """
        Traiter tous les fichiers du dossier selon le mode choisi :
        - mode='paragraph'  : par paragraphe (par défaut)
        - mode='sentence'   : segmenté par phrases
        - mode='full'       : tout l’entretien en un seul bloc
        """
        assert mode in {
            "paragraph",
            "sentence",
            "full",
        }, "mode doit être 'paragraph', 'sentence' ou 'full'"

        all_rows = []
        docx_files = sorted(list(input_dir.glob("*.docx")))

        logger.info("%s", "=" * 70)
        logger.info("NETTOYAGE DE %d FICHIERS (mode: %s)", len(docx_files), mode)
        logger.info("%s\n", "=" * 70)

        for file_path in tqdm(docx_files, desc="Nettoyage"):
            try:
                utterances = self.process_file(file_path)
                if not utterances:
                    continue

                if mode == "sentence":
                    # Découper chaque paragraphe en phrases
                    for item in utterances:
                        parts = re.split(r"(?<=[\.\?\!])\s+", item["text"])
                        for part in parts:
                            part = part.strip()
                            if part:
                                all_rows.append(
                                    {
                                        "filename": item["filename"],
                                        "text": part,
                                        "word_count": len(part.split()),
                                    }
                                )

                elif mode == "full":
                    # Tout concaténer dans un seul texte
                    full_text = " ".join(item["text"] for item in utterances)
                    full_text = re.sub(r"\s+", " ", full_text).strip()
                    all_rows.append(
                        {
                            "filename": file_path.stem,
                            "text": full_text,
                            "word_count": len(full_text.split()),
                        }
                    )

                else:  # paragraph
                    all_rows.extend(utterances)

            except Exception:  # pylint: disable=broad-except
                logger.exception("Erreur avec %s", file_path.name)

        df = pd.DataFrame(all_rows)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False, encoding="utf-8")

        logger.info("Nettoyage terminé")
        logger.info("   - Mode: %s", mode)
        logger.info("   - Lignes nettoyées: %d", len(df))
        logger.info("   - Fichiers traités: %d", df["filename"].nunique() if not df.empty else 0)
        if not df.empty:
            logger.info("   - Mots/ligne (moy): %.1f", df["word_count"].mean())
        else:
            logger.info("   - Mots/ligne (moy): 0")
        logger.info("   - Fichier de sortie: %s\n", output_csv)

        return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Nettoyer les fichiers .docx d'entretiens")
    parser.add_argument(
        "--input", type=Path, default=Path("data/raw"), help="Dossier contenant les fichiers .docx (défaut: data/raw)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help="Dossier de sortie pour les CSV nettoyés (défaut: data/processed)",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "paragraph", "sentence", "all"],
        default="all",
        help="Mode de segmentation: 'full' (document entier), 'paragraph' (paragraphe), 'sentence' (phrase), 'all' (tous les modes)",
    )

    args = parser.parse_args()

    cleaner = InterviewCleaner()
    input_dir_path = args.input
    out_dir_path = args.output

    modes_to_process = ["full", "paragraph", "sentence"] if args.mode == "all" else [args.mode]

    for mode in modes_to_process:
        output_file = out_dir_path / f"cleaned_{mode}.csv"
        logger.info("NETTOYAGE : MODE '%s'", mode.upper())
        cleaner.batch_process(input_dir_path, output_file, mode=mode)
