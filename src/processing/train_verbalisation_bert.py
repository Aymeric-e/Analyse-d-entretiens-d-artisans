"""
Train final BERT model for verbalization difficulty regression.
Loads best hyperparameters from tuning results, or uses defaults if not found.

Usage:
    poetry run python src/processing/train_verbalisation_bert.py \
      --input data/annotation/sentenced_annoted_v1_augmented.csv \
      --tuning-results data/verbalisation/bert_tuning_results.csv
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

warnings.filterwarnings("ignore")


class BertFinalTrainer:
    """Train BERT model with best hyperparameters"""

    def __init__(
        self,
        model_name: str = "distilbert-base-multilingual-cased",
        model_dir: Path = Path("models"),
    ):
        self.model_name = model_name
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilisation du device: {self.device}")

    def load_best_hyperparams(self, tuning_results_path: Path) -> Dict:
        """Load best hyperparameters from tuning results CSV"""
        print("Chargement des meilleurs hyperparamètres...")

        # Default hyperparameters
        default_params = {
            "learning_rate": 2e-5,
            "batch_size": 32,
            "max_length": 128,
            "num_epochs": 5,
        }

        if not tuning_results_path.exists():
            print(f" Fichier de tuning non trouvé: {tuning_results_path}")
            print(f"Utilisation des hyperparamètres par défaut: {default_params}")
            return default_params

        try:
            # Load CSV and get best row (should be sorted by R2)
            df = pd.read_csv(tuning_results_path, sep=";")

            if df.empty:
                print(" Fichier de tuning vide, utilisation des défauts")
                return default_params

            # Get row with best R2
            best_row = df.loc[df["r2"].idxmax()]

            best_params = {
                "learning_rate": float(best_row["learning_rate"]),
                "batch_size": int(best_row["batch_size"]),
                "max_length": int(best_row["max_length"]),
                "num_epochs": int(best_row["num_epochs"]),
            }

            print("Meilleurs hyperparamètres trouvés:")
            print(f"  - Learning rate: {best_params['learning_rate']}")
            print(f"  - Batch size: {best_params['batch_size']}")
            print(f"  - Max length: {best_params['max_length']}")
            print(f"  - Num epochs: {best_params['num_epochs']}")
            print(f"  - R² score: {best_row['r2']:.4f}")

            return best_params

        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f" Erreur lors du chargement du tuning: {str(e)}")
            print(f"Utilisation des hyperparamètres par défaut: {default_params}")
            return default_params

    def save_hyperparams(self, hyperparams: Dict) -> None:
        """Save hyperparameters to JSON file"""
        print("\nSauvegarde des hyperparamètres...")

        params_path = self.model_dir / "bert_final" / "hyperparams.json"
        params_path.parent.mkdir(parents=True, exist_ok=True)

        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(hyperparams, f, indent=2)

        print(f"Hyperparamètres sauvegardés: {params_path}")

    def load_data(self, csv_path: Path) -> Tuple[pd.Series, pd.Series]:
        """Load annotated CSV"""
        print(f"Chargement des données depuis {csv_path}...")
        df = pd.read_csv(csv_path, sep=";")

        required_cols = ["text", "difficulté_verbalisation"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes: {missing}")

        X = df["text"].astype(str)  # pylint: disable=invalid-name
        y = df["difficulté_verbalisation"].astype(float)

        print(f"Données chargées: {len(df)} phrases")
        return X, y

    def prepare_tokenizer(self) -> None:
        """Load tokenizer"""
        print(f"Chargement du tokenizer: {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def tokenize_data(self, texts: pd.Series, labels: pd.Series, max_length: int = 128) -> Dataset:
        """Tokenize texts"""
        print(f"Tokenisation des textes (max_length={max_length})...")

        encodings = self.tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors=None,
        )

        labels_normalized = (labels / 3.0).tolist()

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

    def train(
        self,
        data: Tuple[pd.Series, pd.Series, pd.Series, pd.Series],
        best_params: Dict,
    ) -> dict:
        """Train BERT model"""
        print("\nEntraînement du modèle BERT...")

        X_train, y_train, X_test, y_test = data  # pylint: disable=invalid-name

        learning_rate = (best_params["learning_rate"],)
        batch_size = (best_params["batch_size"],)
        max_length = (best_params["max_length"],)
        num_epochs = (best_params["num_epochs"],)

        self.prepare_tokenizer()

        train_dataset = self.tokenize_data(X_train, y_train, max_length)
        test_dataset = self.tokenize_data(X_test, y_test, max_length)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=1, problem_type="regression"
        ).to(self.device)

        training_args = TrainingArguments(
            output_dir="./tmp_bert_training",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=20,
            save_strategy="no",
            evaluation_strategy="epoch",
            use_cpu=self.device.type == "cpu",
            seed=42,
            fp16=self.device.type == "cuda",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        eval_results = trainer.evaluate()

        print("\nRésultats d'évaluation:")
        print(f"  - R² score: {eval_results.get('eval_r2', 0):.4f}")
        print(f"  - MAE: {eval_results.get('eval_mae', 0):.4f}")
        print(f"  - RMSE: {eval_results.get('eval_rmse', 0):.4f}")

        return eval_results

    def save_model(self) -> None:
        """Save model and tokenizer"""
        print("\nSauvegarde du modèle...")

        model_output = self.model_dir / "bert_final"
        model_output.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(str(model_output / "model"))
        self.tokenizer.save_pretrained(str(model_output / "tokenizer"))

        print(f"Modèle sauvegardé dans: {model_output}")

    def run(self, csv_path: Path, tuning_results_path: Path = None) -> None:
        """Complete training pipeline"""
        print("Entraînement final: BERT pour Difficulté de Verbalisation")

        # Load best hyperparameters
        if tuning_results_path is None:
            tuning_results_path = Path("results/bert_tuning_results.csv")

        best_params = self.load_best_hyperparams(tuning_results_path)

        # Load data
        X, y = self.load_data(csv_path)  # pylint: disable=invalid-name

        # Split
        print("\nSéparation train/test (80/20)...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # pylint: disable=invalid-name
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")

        # Train with loaded hyperparameters
        Data_X_y = X_train, y_train, X_test, y_test  # pylint: disable=invalid-name
        self.train(Data_X_y, best_params)  # pylint: disable=invalid-name

        # Save model and hyperparameters
        self.save_model()
        self.save_hyperparams(best_params)

        print("Entraînement BERT terminé!")


def main():
    """
    Point d'entrée du script. Vérifie les arguments et lance l'entraînement final BERT.

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Entraîner le modèle BERT final avec meilleurs hyperparamètres")
    parser.add_argument("--input", type=Path, required=True, help="Chemin du CSV annoté")
    parser.add_argument(
        "--tuning-results",
        type=Path,
        default=Path("results/bert_tuning_results.csv"),
        help="Chemin du CSV de résultats de tuning (défaut: results/bert_tuning_results.csv)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="distilbert-base-multilingual-cased",
        help="Nom du modèle Hugging Face",
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="Dossier de sauvegarde")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERREUR: Fichier introuvable: {args.input}")
        sys.exit(1)

    trainer = BertFinalTrainer(model_name=args.model_name, model_dir=args.model_dir)
    trainer.run(args.input, args.tuning_results)


if __name__ == "__main__":
    main()
