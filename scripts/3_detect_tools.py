#!/usr/bin/env python3
"""
Script de detection d'outils sur les 3 versions de CSV nettoyes
Lance: poetry run python scripts/3_detect_tools.py
"""

from pathlib import Path
import sys

# Ajouter le chemin src au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.tool_detection import ToolDetector


def main():
    """Point d'entree principal"""
    
    print("\n" + "=" * 70)
    print("ETAPE 3: DETECTION D'OUTILS AVEC NER")
    print("=" * 70 + "\n")
    
    # Initialiser le detecteur
    detector = ToolDetector()
    
    # Definir les chemins
    # ADAPTER SELON LES NOMS EXACTS DE VOS FICHIERS CSV
    base_dir = Path("data/processed")
    
    # Les 3 CSV d'entree (adapter les noms selon vos fichiers)
    input_csvs = [
        base_dir / "cleaned_paragraph.csv",  # Version paragraphe
        base_dir / "cleaned_sentence.csv",   # Version phrase
        base_dir / "cleaned_full.csv",       # Version document entier
    ]
    
    # Verifier quels fichiers existent
    existing_csvs = [csv for csv in input_csvs if csv.exists()]
    
    if not existing_csvs:
        print("ERREUR: Aucun fichier CSV trouve dans data/processed/")
        print("\nFichiers attendus:")
        for csv in input_csvs:
            print(f"  - {csv}")
        print("\nVerifiez les noms de fichiers et relancez le script.")
        return
    
    print(f"Fichiers CSV trouves: {len(existing_csvs)}")
    for csv in existing_csvs:
        print(f"  - {csv.name}")
    print()
    
    # Dossier de sortie
    output_dir = Path("data/processed_tool")
    
    # Traiter tous les CSV
    detector.process_all_csvs(
        input_csvs=existing_csvs,
        output_dir=output_dir,
        dict_filename="tool_dictionary.csv"
    )
    
    print("\n" + "=" * 70)
    print("FICHIERS GENERES:")
    print("=" * 70)
    
    # Lister les fichiers generes
    if output_dir.exists():
        for file in sorted(output_dir.glob("*.csv")):
            size_kb = file.stat().st_size / 1024
            print(f"  - {file.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
