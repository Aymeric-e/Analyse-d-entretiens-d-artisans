"""
csv_text_augmentation.py

Ce module permet d'augmenter automatiquement le texte contenu dans un CSV pour des 
besoins de data augmentation en NLP. 

Fonctionnalités principales :
- Lecture d'un fichier CSV contenant une colonne de texte.
- Augmentation du texte via différents types d'augmenter :
    - "contextual" : substitutions contextuelles avec CamemBERT
    - "translation" : back-translation (FR->DE->FR)
    - "swap" : permutation aléatoire de mots
- Génération de plusieurs versions augmentées par type d'augmentation.
- Création d'un CSV final contenant toutes les lignes originales et augmentées.
- Support de traitement de plusieurs CSV dans un dossier.
- Gestion automatique de la création des dossiers de sortie si nécessaire.

Usage :
    python csv_text_augmentation.py --input <chemin_csv> --output <dossier_sortie> 
                                    [--text_column <nom_colonne_texte>]
                                    [--augmenter_types contextual translation swap]
                                    [--num_aug 1]

Arguments :
- --input : chemin vers le fichier CSV d'entrée ou dossier contenant des CSV.
- --output : chemin vers le fichier CSV de sortie ou dossier pour sauvegarder les CSV augmentés.
- --text_column : nom de la colonne contenant le texte à augmenter (par défaut "text").
- --augmenter_types : liste des types d'augmentation à appliquer (par défaut ["contextual", "translation", "swap"]).
- --num_aug : nombre d'augmentations à générer par type (par défaut 1).

Exemple :
    python csv_text_augmentation.py --input data/my_data.csv --output data/augmented
"""

import argparse
import os

import nlpaug.augmenter.word as naw
import pandas as pd
from tqdm import tqdm

from utils.logger_config import setup_logger

tqdm.pandas()

# Désactiver le parallélisme des tokenizers pour éviter les avertissements
os.environ["TOKENIZERS_PARALLELISM"] = "false"


logger = setup_logger(__name__, level="INFO")


def augment_text(text, augmenter_type, num_aug):
    """
    Genère du texte similaire pour data augmentation en utilisant le type spécifié

    Parameters:
    text (str): text d entrée à être augmenté
    augmenter_type (str): type d augmentation ( 'translation', etc.).
    num_aug (int): nombre d augmentation par type à faire.

    Returns:
    list: liste des versions de texte augmenté.
    """
    match augmenter_type:
        case "contextual":
            augmenter = naw.ContextualWordEmbsAug(model_path="camembert-base", action="substitute")
        case "translation":
            augmenter = naw.BackTranslationAug(
                from_model_name="Helsinki-NLP/opus-mt-fr-de",
                to_model_name="Helsinki-NLP/opus-mt-de-fr",
            )
        case "swap":
            augmenter = naw.RandomWordAug(action="swap")
        case _:
            raise ValueError(f"Augmenter non supporté: {augmenter_type}")

    augmented_texts = []
    for _ in range(num_aug):
        text_augmented = augmenter.augment(text)[0].replace("[", "").replace("]", "").replace('"', "")
        augmented_texts.append(text_augmented)
    return augmented_texts


def augment_dataframe(df, text_column, augmenter_type, num_aug):
    """
    Augmente un DataFrame en générant des versions augmentées du texte dans la colonne spécifiée.

    Parameters:
    df (pd.DataFrame): df d entrée à être augmenté, format attendu : id,text,augmentation_type, label_1,label_2...
    text_column (str): Nom de la colonne où se trouve le texte.
    augmenter_type (str): Type d augmenter.
    num_aug (int): Nombre d augmentation à faire par type.

    Returns:
    pd.DataFrame: Df avec les versions augmentées.
    """

    # Fonction appliquée à chaque ligne
    def process_row(row):
        original_text = row[text_column]
        augmented_texts = augment_text(original_text, augmenter_type, num_aug)
        return augmented_texts

    # Appliquer l’augmentation + barre de progression
    df["augmented_texts"] = df.progress_apply(process_row, axis=1)

    # Dupliquer les lignes pour chaque augmentation
    df_exploded = df.explode("augmented_texts").copy()

    # Renommer la colonne
    df_exploded[text_column] = df_exploded["augmented_texts"]
    df_exploded["augmentation_type"] = augmenter_type

    # Nettoyage
    df_exploded.drop(columns=["augmented_texts"], inplace=True)

    return df_exploded


def process_csv(input_csv, output_csv, text_column, augmenter_types, num_aug):
    """
    Lit un CSV, augmente le texte dans la colonne spécifiée et sauvegarde le résultat dans un nouveau CSV.

    Parameters:
    input_csv (str): Chemin vers le fichier CSV d entrée.
    output_csv (str): Chemin vers le fichier CSV de sortie.
    text_column (str): Nom de la colonne où se trouve le texte.
    augmenter_type (str): Type d augmenter.
    num_aug (int): Nombre d augmentation à faire par type.
    """
    df = pd.read_csv(input_csv, sep=";")
    augmented_df_complete = df.copy()
    augmented_df_complete["id"] = augmented_df_complete["id"].astype(str)
    for augmenter_type in augmenter_types:
        logger.info("Applying augmenter type: %s", augmenter_type)
        augmented_df = augment_dataframe(df, text_column, augmenter_type, num_aug)
        augmented_df_complete = pd.concat([augmented_df_complete, augmented_df], ignore_index=True)

    # trie par id alphabétique et nouvelle colonne id
    old_id_col = augmented_df_complete.columns[0]  # colonne 0 du df
    augmented_df_complete["id"] = augmented_df_complete[old_id_col].astype(str) + "_" + augmented_df_complete["augmentation_type"]
    augmented_df_complete = augmented_df_complete.sort_values(by="id")

    # Si le dossier n'existe pas, le créer
    if not os.path.exists(output_csv):
        logger.info("Creating output directory: %s", output_csv)
        os.makedirs(output_csv, exist_ok=True)

    # Si le output_csv est un dossier, créer le chemin complet
    if os.path.isdir(output_csv):
        filename = os.path.basename(input_csv).replace(".csv", "_augmented.csv")
        output_csv = os.path.join(output_csv, filename)

    logger.info("Saving augmented data to: %s", output_csv)
    augmented_df_complete.to_csv(output_csv, index=False, sep=";")


if __name__ == "__main__":
    # Configuration des paramètres si spécifiés en argument
    # Soit on demande pour un csv particulier ou tous les csv d un dossier
    parser = argparse.ArgumentParser(description="Augmente le texte dans un CSV pour data augmentation.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Chemin vers le fichier CSV d entrée ou dossier contenant des CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Chemin vers le fichier CSV de sortie ou dossier pour sauvegarder les CSV augmentés.",
    )
    parser.add_argument("--text_column", type=str, default="text", help="Nom de la colonne où se trouve le texte.")
    parser.add_argument(
        "--augmenter_types",
        type=str,
        nargs="*",
        default=["contextual", "translation", "swap"],
        help="Liste des types d augmenter ( 'contextual', 'translation', 'swap').",
    )
    parser.add_argument("--num_aug", type=int, default=1, help="Nombre d augmentation à faire par type.")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        for file_name in os.listdir(args.input):
            if file_name.endswith(".csv"):
                input_path = os.path.join(args.input, file_name)
                logger.info("Processing file: %s", input_path)
                process_csv(input_path, args.output, args.text_column, args.augmenter_types, args.num_aug)
    else:
        process_csv(args.input, args.output, args.text_column, args.augmenter_types, args.num_aug)
