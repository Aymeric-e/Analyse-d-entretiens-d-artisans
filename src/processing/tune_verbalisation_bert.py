"""
Tune BERT model hyperparameters for verbalization difficulty regression.
Includes epochs in the hyperparameter grid.

Usage:
    poetry run python src/processing/tune_verbalisation_bert.py \
      --input data/annotation/sentenced_annoted_v1_augmented.csv \
      --output data/verbalisation/bert_tuning_results.csv
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from utils.logger_config import setup_logger

warnings.filterwarnings("ignore")


logger = setup_logger(__name__, level="INFO")


class BertHyperparameterTuner:
    """Tune BERT model for verbalization difficulty prediction"""

    def __init__(
        self,
        model_name: str = "distilbert-base-multilingual-cased",
        score_col: str = "difficulté_verbalisation",
        score_scale: float = 10.0,
    ):
        self.model_name = model_name
        self.score_col = score_col
        self.score_scale = float(score_scale)
        self.tokenizer = None
        self.model = None
        self.results = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Utilisation du device: %s", self.device)

    def load_data(self, csv_path: Path) -> Tuple[pd.Series, pd.Series]:
        """Load annotated CSV (returns raw scores, normalization happens in tokenize_data)"""
        logger.info("Chargement des données depuis %s...", csv_path)
        df = pd.read_csv(csv_path, sep=";")

        required_cols = ["text", self.score_col]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes: {missing}")

        # Filtrage défensif des lignes avec valeurs manquantes dans 'text' ou la colonne cible
        orig_len = len(df)
        df = df.dropna(subset=["text", self.score_col])
        dropped = orig_len - len(df)
        if dropped > 0:
            logger.warning("Suppression de %d lignes avec valeurs manquantes dans 'text' ou '%s'", dropped, self.score_col)

        if len(df) == 0:
            raise ValueError("Aucune donnée restante après suppression des lignes manquantes")

        # Assurer que la colonne cible est numérique, et supprimer les lignes non convertibles
        df[self.score_col] = pd.to_numeric(df[self.score_col], errors="coerce")
        nan_after_convert = int(df[self.score_col].isna().sum())
        if nan_after_convert > 0:
            logger.warning("La colonne cible contient %d valeurs non convertibles en float; suppression", nan_after_convert)
            df = df.dropna(subset=[self.score_col])

        X = df["text"].astype(str)  # pylint: disable=invalid-name
        y = df[self.score_col].astype(float)

        logger.info("Données chargées: %d phrases (après filtrage)", len(df))
        return X.reset_index(drop=True), y.reset_index(drop=True)

    def prepare_tokenizer(self) -> None:
        """Load tokenizer"""
        logger.info("Chargement du tokenizer: %s...", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def tokenize_data(self, texts: pd.Series, labels: pd.Series, max_length: int = 128) -> Dataset:
        """Tokenize texts and normalize labels by the configured scale"""
        max_length = int(max_length)
        encodings = self.tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors=None,
        )

        labels_normalized = (labels / self.score_scale).tolist()

        dataset_dict = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels_normalized,
        }

        return Dataset.from_dict(dataset_dict)

    def compute_metrics(self, eval_pred) -> dict:
        """Compute evaluation metrics (rescale to original range)"""
        predictions, labels = eval_pred
        predictions = predictions.squeeze()

        predictions = predictions * self.score_scale
        labels = labels * self.score_scale

        mse = mean_squared_error(labels, predictions)
        mae = mean_absolute_error(labels, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(labels, predictions)

        return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}

    def tune_hyperparameters(self, X: pd.Series, y: pd.Series) -> dict:  # pylint: disable=invalid-name
        """Tune BERT hyperparameters with epochs"""
        logger.info("Tuning des hyperparamètres BERT...")

        self.prepare_tokenizer()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # pylint: disable=invalid-name
        logger.info("Train: %d, Test: %d", len(X_train), len(X_test))

        learning_rates = [1e-5, 2e-5, 5e-5]
        batch_sizes = [16, 32]
        max_lengths = [128, 256]
        num_epochs_list = [3, 5, 7]

        best_r2 = -float("inf")
        best_params = None
        results = []

        total_combinations = len(learning_rates) * len(batch_sizes) * len(max_lengths) * len(num_epochs_list)
        combo_idx = 0

        combo_hyperparams = [learning_rates, batch_sizes, max_lengths, num_epochs_list]

        for permutation in np.array(np.meshgrid(*combo_hyperparams)).T.reshape(-1, 4):
            lr, batch_size, max_length, num_epochs = permutation
            batch_size = int(batch_size)
            max_length = int(max_length)
            num_epochs = int(num_epochs)
            combo_idx += 1
            logger.info(
                "[%d/%d] lr=%s, bs=%s, ml=%s, epochs=%s", combo_idx, total_combinations, lr, batch_size, max_length, num_epochs
            )

            try:
                train_dataset = self.tokenize_data(X_train, y_train, max_length)
                test_dataset = self.tokenize_data(X_test, y_test, max_length)

                model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, num_labels=1, problem_type="regression"
                ).to(self.device)

                training_args = TrainingArguments(
                    output_dir=f"./tmp_bert_tuning/lr_{lr}_bs_{batch_size}_ml_{max_length}_ep_{num_epochs}",
                    num_train_epochs=num_epochs,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    learning_rate=lr,
                    logging_steps=10,
                    save_strategy="no",
                    evaluation_strategy="epoch",
                    use_cpu=self.device.type == "cpu",
                    seed=42,
                    fp16=self.device.type == "cuda",
                )

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=test_dataset,
                    compute_metrics=self.compute_metrics,
                )

                trainer.train()
                eval_results = trainer.evaluate()

                r2 = eval_results.get("eval_r2", -float("inf"))
                mae = eval_results.get("eval_mae", float("inf"))

                logger.info("  → R²: %.4f, MAE: %.4f", r2, mae)

                results.append(
                    {
                        "learning_rate": lr,
                        "batch_size": batch_size,
                        "max_length": max_length,
                        "num_epochs": num_epochs,
                        "r2": r2,
                        "mae": mae,
                        "rmse": eval_results.get("eval_rmse", float("inf")),
                        "mse": eval_results.get("eval_mse", float("inf")),
                    }
                )

                if r2 > best_r2:
                    best_r2 = r2
                    best_params = {
                        "learning_rate": lr,
                        "batch_size": batch_size,
                        "max_length": max_length,
                        "num_epochs": num_epochs,
                    }

                del model
                torch.cuda.empty_cache()

            except Exception:  # pylint: disable=broad-except
                logger.exception("Erreur lors du tuning pour combo %d", combo_idx)
                continue

        self.results = results

        logger.info("Meilleures paramètres: %s", best_params)
        logger.info("Meilleur R²: %.4f", best_r2)

        return {"best_params": best_params, "best_r2": best_r2, "all_results": results}

    def save_results(self, output_path: Path) -> None:
        """Save tuning results"""
        logger.info("Sauvegarde des résultats...")

        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values("r2", ascending=False)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False, sep=";")

        logger.info("Résultats sauvegardés: %s", output_path)

    def run(self, csv_path: Path, output_path: Path) -> None:
        """Complete pipeline"""
        logger.info("Tuning: BERT pour Difficulté de Verbalisation")

        X, y = self.load_data(csv_path)  # pylint: disable=invalid-name
        self.tune_hyperparameters(X, y)
        self.save_results(output_path)

        logger.info("Tuning BERT terminé")


def main():
    """
    Docstring for main
    """
    parser = argparse.ArgumentParser(description="Tuner les hyperparamètres BERT pour la régression de difficulté")
    parser.add_argument("--input", type=Path, required=True, help="Chemin du CSV annoté")
    parser.add_argument("--output", type=Path, required=True, help="Chemin du fichier CSV de sortie")
    parser.add_argument(
        "--model-name",
        type=str,
        default="distilbert-base-multilingual-cased",
        help="Nom du modèle Hugging Face",
    )

    parser.add_argument(
        "--target-col",
        type=str,
        default="difficulté_verbalisation",
        help="Nom de la colonne cible contenant le score annoté",
    )

    parser.add_argument(
        "--score-scale",
        type=float,
        default=10.0,
        help="Échelle maximale du score annoté (utilisé pour normalisation)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        logger.error("ERREUR: Fichier introuvable: %s", args.input)
        sys.exit(1)

    tuner = BertHyperparameterTuner(model_name=args.model_name, score_col=args.target_col, score_scale=args.score_scale)
    tuner.run(args.input, args.output)


if __name__ == "__main__":
    main()
