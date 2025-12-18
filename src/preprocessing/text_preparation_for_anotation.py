"""
Préparer un CSV pour annotation.

En entrée, le CSV doit contenir les colonnes : filename, text (et optionnellement word_count).
En sortie, le CSV contiendra : id, filename, text, augmentation_type, label_1, label_2, ...

Fonctionnalités :
- Crée une colonne 'id' unique pour chaque ligne
- Ajoute une colonne 'augmentation_type' initialisée à 'original'
- Permet de filtrer par noms de fichiers (colonne 'filename') via --filenames
- Ajoute des colonnes de labels vides pour annotation manuelle
- Supprime la colonne 'word_count' si présente

Usage :
    python text_preparation_for_anotation.py --input data/processed/cleaned_paragraph.csv \\
        --output data/annotation/sentences_prepared.csv \\
        --labels intimité fluidité émotions \\
        --filenames entretien1 entretien2

"""

import argparse
import os
import sys

import pandas as pd

from utils.logger_config import setup_logger

logger = setup_logger(__name__, level="INFO")


def prepare_text_for_annotation(input_csv, output_csv, label_columns, filenames_filter=None):
    """
    Prépare le CSV pour l'annotation en ajoutant les colonnes nécessaires.

    Charge le CSV, filtre optionnellement par noms de fichiers, ajoute des colonnes
    (id, augmentation_type, labels) et sauvegarde le résultat.

    Parameters:
    -----------
    input_csv : str
        Chemin vers le fichier CSV d'entrée.
    output_csv : str
        Chemin vers le fichier CSV de sortie (peut être un fichier ou un dossier).
    label_columns : list[str]
        Liste des noms de colonnes de labels à ajouter (initialisées vides).
    filenames_filter : list[str] or None
        Liste des noms de fichiers (colonne 'filename') à conserver.
        Si None, tous les fichiers sont conservés. Défaut : None

    Returns:
    --------
    None. Sauvegarde le fichier préparé à output_csv.
    """
    logger.info("Chargement du fichier d'entrée : %s", input_csv)
    df = pd.read_csv(input_csv, sep=",")

    # Filtrer par noms de fichiers si fourni
    if filenames_filter:
        logger.info("Filtrage par noms de fichiers : %s", filenames_filter)
        if "filename" not in df.columns:
            logger.error("Colonne 'filename' introuvable dans le CSV. Arrêt.")
            raise ValueError("Le CSV d'entrée doit contenir une colonne 'filename'.")
        df = df[df["filename"].isin(filenames_filter)]
        logger.info("Après filtrage : %d lignes conservées", len(df))
    else:
        logger.info("Aucun filtrage par nom de fichier ; conservation de toutes les lignes (%d)", len(df))

    # Créer une colonne 'id' unique
    df.insert(0, "id", range(1, len(df) + 1))
    logger.info("Colonne 'id' créée (de 1 à %d)", len(df))

    # Ajouter la colonne 'augmentation_type' initialisée à 'original'
    df["augmentation_type"] = "original"
    logger.info("Colonne 'augmentation_type' créée et initialisée à 'original'")

    # Supprimer la colonne word_count si elle existe
    if "word_count" in df.columns:
        df = df.drop(columns=["word_count"])
        logger.info("Colonne 'word_count' supprimée")

    # Ajouter les colonnes de labels initialisées vides
    for label in label_columns:
        df[label] = ""
    if label_columns:
        logger.info("Colonnes de labels créées : %s", label_columns)

    # Créer le dossier parent si nécessaire
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        logger.info("Création du dossier de sortie : %s", output_dir)
        os.makedirs(output_dir, exist_ok=True)

    # Sauvegarder le CSV
    df.to_csv(output_csv, index=False, sep=";")
    logger.info("Fichier préparé sauvegardé : %s", output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prépare un CSV pour l'annotation en ajoutant les colonnes nécessaires et en filtrant les entretiens.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  # Préparer tous les entretiens avec 3 labels
  python text_preparation_for_anotation.py \\
    --input data/processed/cleaned_paragraph.csv \\
    --output data/annotation/sentences_prepared.csv \\
    --labels intimité fluidité émotions

  # Préparer uniquement les entretiens 'entretien1' et 'entretien2'
  python text_preparation_for_anotation.py \\
    --input data/processed/cleaned_paragraph.csv \\
    --output data/annotation/sentences_prepared.csv \\
    --labels intimité fluidité \\
    --filenames entretien1 entretien2
        """,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Chemin vers le fichier CSV d'entrée (doit contenir colonnes 'filename' et 'text').",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Chemin complet du fichier CSV de sortie (ex: data/annotation/sentences_prepared.csv). "
        "Les dossiers parent seront créés si nécessaire.",
    )

    parser.add_argument(
        "--labels",
        type=str,
        nargs="*",
        default=[],
        help="Liste des noms de colonnes de labels à ajouter (séparés par des espaces). Défaut : aucun label.",
    )

    parser.add_argument(
        "--filenames",
        type=str,
        nargs="*",
        default=None,
        help="Liste des noms de fichiers (valeurs colonne 'filename') à conserver. "
        "Si non fourni, tous les entretiens sont conservés.",
    )

    args = parser.parse_args()

    # Vérifier que le fichier d'entrée existe
    if not os.path.isfile(args.input):
        logger.error("Fichier d'entrée introuvable : %s", args.input)
        sys.exit(1)

    logger.info("Démarrage de la préparation du CSV pour annotation")
    logger.info("  Entrée : %s", args.input)
    logger.info("  Sortie : %s", args.output)
    logger.info("  Labels : %s", args.labels if args.labels else "aucun")
    logger.info("  Filtrage par filenames : %s", args.filenames if args.filenames else "aucun (tous conservés)")

    try:
        prepare_text_for_annotation(args.input, args.output, args.labels, filenames_filter=args.filenames)
        logger.info("Préparation terminée avec succès")
    except Exception:  # pylint: disable=broad-except
        logger.exception("Erreur lors de la préparation")
        sys.exit(1)
