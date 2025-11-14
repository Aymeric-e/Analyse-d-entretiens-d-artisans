import re
from pathlib import Path
from typing import List, Dict
from docx import Document
import pandas as pd
from tqdm import tqdm


class InterviewCleaner:
    """Nettoie les entretiens (3 niveaux de segmentation possibles)"""

    def clean_artisan_text(self, text: str) -> str:
        """Nettoyer une ligne de parole d'artisan"""
        text = re.sub(r'\[(Interviewe|Interviewé)\]', '', text)
        text = re.sub(r'\(\s*\d+:\d+\s*(?:–|-|‐)\s*\d+:\d+\s*\)', '', text)
        text = re.sub(r'\(\s*\d+:\d+\s*\)', '', text)
        text = re.sub(r'^\s*(?:–|-|‐)\s*', '', text)
        text = re.sub(r'^\s*\)\s*$', '', text)
        text = re.sub(r'^\s*\(\s*$', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
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
                cleaned_lines.append({
                    "filename": filename,
                    "text": cleaned,
                    "word_count": len(cleaned.split())
                })

        return cleaned_lines

    def batch_process(self, input_dir: Path, output_csv: Path, mode: str = "paragraph"):
        """
        Traiter tous les fichiers du dossier selon le mode choisi :
        - mode='paragraph'  : par paragraphe (par défaut)
        - mode='sentence'   : segmenté par phrases
        - mode='full'       : tout l’entretien en un seul bloc
        """
        assert mode in {"paragraph", "sentence", "full"}, "mode doit être 'paragraph', 'sentence' ou 'full'"

        all_rows = []
        docx_files = sorted(list(input_dir.glob("*.docx")))

        print(f"\n{'='*70}")
        print(f"NETTOYAGE DE {len(docx_files)} FICHIERS (mode: {mode})")
        print(f"{'='*70}\n")

        for file_path in tqdm(docx_files, desc="Nettoyage"):
            try:
                utterances = self.process_file(file_path)
                if not utterances:
                    continue

                if mode == "sentence":
                    # Découper chaque paragraphe en phrases
                    for item in utterances:
                        parts = re.split(r'(?<=[\.\?\!])\s+', item["text"])
                        for part in parts:
                            part = part.strip()
                            if part:
                                all_rows.append({
                                    "filename": item["filename"],
                                    "text": part,
                                    "word_count": len(part.split())
                                })

                elif mode == "full":
                    # Tout concaténer dans un seul texte
                    full_text = " ".join(item["text"] for item in utterances)
                    full_text = re.sub(r'\s+', ' ', full_text).strip()
                    all_rows.append({
                        "filename": file_path.stem,
                        "text": full_text,
                        "word_count": len(full_text.split())
                    })

                else:  # paragraph
                    all_rows.extend(utterances)

            except Exception as e:
                print(f"Erreur avec {file_path.name}: {e}")

        df = pd.DataFrame(all_rows)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False, encoding='utf-8')

        print(f"\n{'='*70}")
        print("NETTOYAGE TERMINÉ")
        print(f"{'='*70}")
        print(f"   - Mode: {mode}")
        print(f"   - Lignes nettoyées: {len(df)}")
        print(f"   - Fichiers traités: {df['filename'].nunique() if not df.empty else 0}")
        print(f"   - Mots/ligne (moy): {df['word_count'].mean():.1f}" if not df.empty else "   - Mots/ligne (moy): 0")
        print(f"   - Fichier de sortie: {output_csv}\n")

        return df


if __name__ == "__main__":
    cleaner = InterviewCleaner()
    cleaner.batch_process(Path("data/raw"), Path("data/processed/cleaned_full.csv"), mode="full")
    cleaner.batch_process(Path("data/raw"), Path("data/processed/cleaned.csv"), mode="paragraph")
    cleaner.batch_process(Path("data/raw"), Path("data/processed/cleaned_segmente.csv"), mode="sentence")
