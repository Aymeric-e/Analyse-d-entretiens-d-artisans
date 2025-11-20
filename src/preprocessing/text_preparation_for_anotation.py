"""
Pour préparer notre csv pour anotation
En entrée csv on a les colonnes : filename, text,word_count
En sortie csv on a les colonnes : id, filename, text, augmentation_type, label_1, label_2,...
On crée un id unique pour chaque ligne, on ajoute une colonne augmentation_type initialisée à "original" 
et on peut ajouter autant de colonne label qu'on veut, toutes initialisées vides.
"""

import pandas as pd
import argparse
import os

def prepare_text_for_annotation(input_csv, output_csv, label_columns):
    """
    Prépare le CSV pour l'annotation en ajoutant les colonnes nécessaires.

    Parameters:
    input_csv (str): Chemin vers le fichier CSV d'entrée.
    output_csv (str): Chemin vers le fichier CSV de sortie.
    label_columns (list): Liste des noms de colonnes de labels à ajouter.
    """
    df = pd.read_csv(input_csv)

    # Créer une colonne 'id' unique
    df.insert(0, 'id', range(1, len(df) + 1))

    # Ajouter la colonne 'augmentation_type' initialisée à 'original'
    df['augmentation_type'] = 'original'

    #Supprimer la colonne word_count si elle existe
    if 'word_count' in df.columns:
        df = df.drop(columns=['word_count'])

    # Ajouter les colonnes de labels initialisées vides
    for label in label_columns:
        df[label] = ''

    #Si le dossier n'existe pas, le créer
    if not os.path.exists(output_csv):
        print("Creating output directory:", output_csv)
        os.makedirs(output_csv, exist_ok=True)

    # Si le output_csv est un dossier, créer le chemin complet
    if os.path.isdir(output_csv):

        filename = os.path.basename(input_csv).replace(".csv", "_prepared.csv")
        output_csv = os.path.join(output_csv, filename)
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":

        #Configuration des paramètres si spécifiés en argument
    #Soit on demande pour un csv particulier ou tous les csv d un dossier
    parser = argparse.ArgumentParser(description="Prépare un CSV pour l'annotation.")
    parser.add_argument("--input", type=str, required=True, help="Chemin vers le fichier CSV d'entrée ou dossier contenant des CSV.")
    parser.add_argument("--output", type=str, required=True, help="Chemin vers le fichier CSV de sortie ou dossier pour sauvegarder les CSV préparés.")
    parser.add_argument("--labels", type=str, nargs='*', default=[], help="Liste des noms de colonnes de labels à ajouter.")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        # Traiter tous les fichiers CSV dans le dossier
        for filename in os.listdir(args.input):
            if filename.endswith(".csv"):
                input_path = os.path.join(args.input, filename)
                print("Processing file:", input_path)
                prepare_text_for_annotation(input_path, args.output, args.labels)
    else:
        # Traiter un seul fichier CSV
        prepare_text_for_annotation(args.input, args.output, args.labels)
    