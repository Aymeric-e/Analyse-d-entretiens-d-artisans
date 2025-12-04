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
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("ignore")


class BertHyperparameterTuner:
    """Tune BERT model for verbalization difficulty prediction"""

    def __init__(self, model_name: str = "distilbert-base-multilingual-cased"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.results = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilisation du device: {self.device}")

    def load_data(self, csv_path: Path) -> Tuple[pd.Series, pd.Series]:
        """Load annotated CSV"""
        print(f"Chargement des données depuis {csv_path}...")
        df = pd.read_csv(csv_path, sep=";")

        required_cols = ["text", "difficulté_verbalisation"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes: {missing}")

        X = df["text"].astype(str)
        y = df["difficulté_verbalisation"].astype(float) / 3.0

        print(f"Données chargées: {len(df)} phrases")
        return X, y

    def prepare_tokenizer(self) -> None:
        """Load tokenizer"""
        print(f"Chargement du tokenizer: {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def tokenize_data(self, texts: pd.Series, labels: pd.Series, max_length: int = 128) -> Dataset:
        """Tokenize texts"""
        encodings = self.tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors=None,
        )

        labels_normalized = (labels / 1.0).tolist()

        dataset_dict = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels_normalized,
        }

        return Dataset.from_dict(dataset_dict)

    def compute_metrics(self, eval_pred) -> dict:
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = predictions.squeeze()

        predictions = predictions * 3.0
        labels = labels * 3.0

        mse = mean_squared_error(labels, predictions)
        mae = mean_absolute_error(labels, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(labels, predictions)

        return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}

    def tune_hyperparameters(self, X: pd.Series, y: pd.Series) -> dict:
        """Tune BERT hyperparameters with epochs"""
        print("Tuning des hyperparamètres BERT...")

        self.prepare_tokenizer()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")

        learning_rates = [1e-5, 2e-5, 5e-5]
        batch_sizes = [16, 32]
        max_lengths = [128, 256]
        num_epochs_list = [3, 5, 7]

        best_r2 = -float("inf")
        best_params = None
        results = []

        total_combinations = (
            len(learning_rates) * len(batch_sizes) * len(max_lengths) * len(num_epochs_list)
        )
        combo_idx = 0

        for lr in learning_rates:
            for batch_size in batch_sizes:
                for max_length in max_lengths:
                    for num_epochs in num_epochs_list:
                        combo_idx += 1
                        print(
                            f"\n[{combo_idx}/{total_combinations}] lr={lr}, bs={batch_size}, ml={max_length}, epochs={num_epochs}"
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

                            print(f"  → R²: {r2:.4f}, MAE: {mae:.4f}")

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

                        except Exception as e:
                            print(f"  Erreur: {str(e)}")
                            continue

        self.results = results

        print(f"\nMeilleures paramètres: {best_params}")
        print(f"Meilleur R²: {best_r2:.4f}")

        return {"best_params": best_params, "best_r2": best_r2, "all_results": results}

    def save_results(self, output_path: Path) -> None:
        """Save tuning results"""
        print(f"\nSauvegarde des résultats...")

        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values("r2", ascending=False)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False, sep=";")

        print(f"Résultats sauvegardés: {output_path}")

    def run(self, csv_path: Path, output_path: Path) -> None:
        """Complete pipeline"""
        print("Tuning: BERT pour Difficulté de Verbalisation")

        X, y = self.load_data(csv_path)
        self.tune_hyperparameters(X, y)
        self.save_results(output_path)

        print("Tuning BERT terminé!")


def main():
    parser = argparse.ArgumentParser(
        description="Tuner les hyperparamètres BERT pour la régression de difficulté"
    )
    parser.add_argument("--input", type=Path, required=True, help="Chemin du CSV annoté")
    parser.add_argument(
        "--output", type=Path, required=True, help="Chemin du fichier CSV de sortie"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="distilbert-base-multilingual-cased",
        help="Nom du modèle Hugging Face",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERREUR: Fichier introuvable: {args.input}")
        sys.exit(1)

    tuner = BertHyperparameterTuner(model_name=args.model_name)
    tuner.run(args.input, args.output)


if __name__ == "__main__":
    main()
