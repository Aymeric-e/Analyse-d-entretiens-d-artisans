"""
multiple_bert_models.py

Run tune -> train -> predict for multiple target score columns and aggregate predictions

Usage:
  python scripts/multiple_bert_models.py \
    --annotated-csv data/interviews/annotation/sentences_annoted_verb_augmented.csv \
    --predict-csv data/interviews/processed/cleaned_sentence.csv \
    --output-csv results/verbalisation/new/all_scores_predictions.csv \
    --n-scores 1 \
    --tun-dir data/verbalisation/new
    --model-dir models
    

If --columns is not provided the script will infer score columns by taking N columns 
starting at index 4 where N is --n-scores (default 7).

For each column:
 - run tuning and save tuning results to data/verbalisation/<col>_bert_tuning_results.csv
 - train and save model to models/<col>/bert_final
 - save hyperparams CSV to data/hyperparamètres/<col>_bert_hyperparam.csv (created by trainer)
 - predict on the provided predict CSV and add a column `note_bert_<col>` to the aggregated output

"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from processing.predict_bert import BertPredictor
from processing.train_bert import BertFinalTrainer
from processing.tune_bert import BertHyperparameterTuner

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def infer_columns(annotated_csv: Path, start_index: int = 4, n_scores: int = 7):
    """Infer score columns from annotated CSV by taking n_scores columns starting at start_index"""
    df = pd.read_csv(annotated_csv, sep=";")
    cols = list(df.columns)
    return cols[start_index : start_index + n_scores]


def main():
    """Main function to run the pipeline for multiple columns"""
    parser = argparse.ArgumentParser(description="Run BERT pipeline for multiple score columns")
    parser.add_argument("--annotated-csv", type=Path, required=True, help="CSV annoté avec colonnes de scores (séparateur ';')")
    parser.add_argument(
        "--predict-csv", type=Path, required=True, help="CSV contenant les textes à prédire (doit contenir 'text' et 'filename')"
    )
    parser.add_argument(
        "--output-csv", type=Path, required=True, help="CSV de sortie agrégé avec toutes les colonnes de prédiction"
    )
    parser.add_argument("--columns", type=str, nargs="*", default=None, help="Liste de colonnes à traiter (si absent on infère)")
    parser.add_argument("--n-scores", type=int, default=7, help="Nombre de colonnes scores si non spécifié (défaut:7)")
    parser.add_argument(
        "--start-index", type=int, default=4, help="Index de départ (0-based) pour inférer les colonnes (défaut:4)"
    )
    parser.add_argument("--tun-dir", type=Path, default=Path("data/hyperparamètres"), help="Dossier pour stocker tuning/preds")
    parser.add_argument(
        "--model-dir", type=Path, default=Path("models"), help="Dossier racine pour stocker les modèles par colonne"
    )
    parser.add_argument(
        "--bert-model-name", type=str, default="distilbert-base-multilingual-cased", help="Nom du modèle HuggingFace"
    )
    parser.add_argument("--score-scale", type=float, default=10.0, help="Échelle maximale du score annoté")

    args = parser.parse_args()

    if not args.annotated_csv.exists():
        logger.error("Fichier annoté introuvable: %s", args.annotated_csv)
        sys.exit(1)
    if not args.predict_csv.exists():
        logger.error("Fichier de textes à prédire introuvable: %s", args.predict_csv)
        sys.exit(1)

    if args.columns:
        columns = args.columns
    else:
        columns = infer_columns(args.annotated_csv, start_index=args.start_index, n_scores=args.n_scores)

    logger.info("Colonnes à traiter: %s", columns)

    tun_dir = Path(args.tun_dir)
    tun_dir.mkdir(parents=True, exist_ok=True)

    # Load predict CSV (we will append columns to it)
    df_pred = pd.read_csv(args.predict_csv)
    if "text" not in df_pred.columns:
        logger.error("Le CSV de prédiction doit contenir une colonne 'text'")
        sys.exit(1)

    # For each column, run tune -> train -> predict and append predictions
    for col in columns:
        logger.info("===== Traitement de la colonne: %s =====", col)

        try:
            # 1) Tuning
            tuning_out = tun_dir / f"{col}_bert_tuning_results.csv"
            tuner = BertHyperparameterTuner(model_name=args.bert_model_name, score_col=col, score_scale=args.score_scale)
            logger.info("Lancement du tuning pour %s -> %s", col, tuning_out)
            tuner.run(args.annotated_csv, tuning_out)

            # 2) Train
            trainer = BertFinalTrainer(
                model_name=args.bert_model_name, model_dir=args.model_dir, score_col=col, score_scale=args.score_scale
            )
            logger.info("Lancement de l'entraînement pour %s (tuning file: %s)", col, tuning_out)
            trainer.run(args.annotated_csv, tuning_results_path=tuning_out)

            # 3) Predict
            predictor = BertPredictor(model_dir=args.model_dir, score_col=col, score_scale=args.score_scale)
            logger.info("Prédiction pour %s sur %s", col, args.predict_csv)
            preds = predictor.predict(df_pred["text"], max_length=128)

            col_name = f"note_bert_{col}"
            df_pred[col_name] = preds

            logger.info("Terminé pour %s. Colonne ajoutée: %s", col, col_name)

        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Erreur lors du pipeline pour la colonne %s", col)

    # Save final aggregated CSV
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_pred.to_csv(args.output_csv, index=False, sep=";")

    logger.info("Tout terminé. Résultat agrégé sauvegardé: %s", args.output_csv)


if __name__ == "__main__":
    main()
