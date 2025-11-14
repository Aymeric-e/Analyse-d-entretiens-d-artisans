#!/usr/bin/env python3
"""Lancer le nettoyage des entretiens (3 versions)"""

from pathlib import Path
from preprocessing.text_cleaning import InterviewCleaner

if __name__ == "__main__":
    cleaner = InterviewCleaner()
    input_dir = Path("data/raw")
    out_dir = Path("data/processed")

    print("\n=== NETTOYAGE : VERSION NON SEGMENTÉE ===")
    cleaner.batch_process(input_dir, out_dir / "cleaned_full.csv", mode="full")

    print("\n=== NETTOYAGE : VERSION PARAGRAPHE ===")
    cleaner.batch_process(input_dir, out_dir / "cleaned_paragraph.csv", mode="paragraph")

    print("\n=== NETTOYAGE : VERSION PAR PHRASE ===")
    cleaner.batch_process(input_dir, out_dir / "cleaned_sentence.csv", mode="sentence")
