"""
Train final BERT model for verbalization difficulty regression.
Loads best hyperparameters from tuning results, or uses defaults if not found.

Usage:
    poetry run python src/processing/train_bert.py \
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

from utils.logger_config import setup_logger

warnings.filterwarnings("ignore")

logger = setup_logger(__name__, level="INFO")


class BertFinalTrainer:
    """Train BERT model with best hyperparameters"""

    def __init__(
        self,
        model_name: str = "distilbert-base-multilingual-cased",
        model_dir: Path = Path("models"),
        score_col: str = "difficulté_verbalisation",
        score_scale: float = 10.0,
    ):
        self.model_name = model_name
        self.model_dir = model_dir
        self.score_col = score_col
        self.score_scale = float(score_scale)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Utilisation du device: %s", self.device)

    def load_best_hyperparams(self, tuning_results_path: Path) -> Dict:
        """Load best hyperparameters from tuning results CSV"""
        logger.info("Chargement des meilleurs hyperparamètres...")

        # Default hyperparameters
        default_params = {
            "learning_rate": 2e-5,
            "batch_size": 32,
            "max_length": 128,
            "num_epochs": 5,
        }

        if not tuning_results_path.exists():
            logger.warning("Fichier de tuning non trouvé: %s", tuning_results_path)
            logger.info("Utilisation des hyperparamètres par défaut: %s", default_params)
            return default_params

        try:
            # Load CSV and get best row (should be sorted by R2)
            df = pd.read_csv(tuning_results_path, sep=";")

            if df.empty:
                logger.warning("Fichier de tuning vide, utilisation des hyperparamètres par défaut")
                return default_params

            # Get row with best R2
            best_row = df.loc[df["r2"].idxmax()]

            best_params = {
                "learning_rate": float(best_row["learning_rate"]),
                "batch_size": int(best_row["batch_size"]),
                "max_length": int(best_row["max_length"]),
                "num_epochs": int(best_row["num_epochs"]),
            }

            logger.info("Meilleurs hyperparamètres trouvés:")
            logger.info("  - Learning rate: %s", best_params["learning_rate"])
            logger.info("  - Batch size: %s", best_params["batch_size"])
            logger.info("  - Max length: %s", best_params["max_length"])
            logger.info("  - Num epochs: %s", best_params["num_epochs"])
            logger.info("  - R² score: %.4f", best_row["r2"])

            return best_params

        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Erreur lors du chargement du tuning")
            logger.info("Utilisation des hyperparamètres par défaut: %s", default_params)
            return default_params

    def save_hyperparams(self, hyperparams: Dict) -> None:
        """Save hyperparameters to JSON file and CSV in data/hyperparamètres"""
        logger.info("Sauvegarde des hyperparamètres...")

        params_path = self.model_dir / self.score_col / "bert_final" / "hyperparams.json"
        params_path.parent.mkdir(parents=True, exist_ok=True)

        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(hyperparams, f, indent=2)

        # Also save a CSV copy in data/hyperparamètres/<score_col>_bert_hyperparam.csv
        hyper_dir = Path("data") / "hyperparamètres"
        hyper_dir.mkdir(parents=True, exist_ok=True)

        try:
            df = pd.DataFrame([hyperparams])
            csv_path = hyper_dir / f"{self.score_col}_bert_hyperparam.csv"
            df.to_csv(csv_path, index=False, sep=";")
            logger.info("Hyperparamètres CSV sauvegardés: %s", csv_path)
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Impossible de sauvegarder le CSV des hyperparamètres")

        logger.info("Hyperparamètres sauvegardés: %s", params_path)

    def load_data(self, csv_path: Path) -> Tuple[pd.Series, pd.Series]:
        """Load annotated CSV (returns raw scores)"""
        logger.info("Chargement des données depuis %s...", csv_path)
        df = pd.read_csv(csv_path, sep=";")

        required_cols = ["text", self.score_col]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes: {missing}")

        # Filtrage des lignes avec valeurs manquantes dans 'text' ou la colonne cible
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
        logger.info("Tokenisation des textes (max_length=%d)...", max_length)

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

    def train(
        self,
        data: Tuple[pd.Series, pd.Series, pd.Series, pd.Series],
        best_params: Dict,
    ) -> dict:
        """Train BERT model"""
        logger.info("Entraînement du modèle BERT...")

        X_train, y_train, X_test, y_test = data  # pylint: disable=invalid-name

        learning_rate = float(best_params["learning_rate"])
        batch_size = int(best_params["batch_size"])
        max_length = int(best_params["max_length"])
        num_epochs = int(best_params["num_epochs"])

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

        # Sauvegarde des métriques d'évaluation pour l'entraînement final
        try:
            loss_dir = Path("results") / "intimite"
            loss_dir.mkdir(parents=True, exist_ok=True)
            loss_file = loss_dir / f"{self.score_col}_loss.txt"
            with open(loss_file, "a", encoding="utf-8") as f:
                f.write(
                    f"{pd.Timestamp.now().isoformat()} | final_train | lr={learning_rate},\
                        nbs={batch_size}, ml={max_length}, epochs={num_epochs} | "
                    f"r2={eval_results.get('eval_r2')}, mae={eval_results.get('eval_mae')},\
                        rmse={eval_results.get('eval_rmse')}, mse={eval_results.get('eval_mse')}\n"
                )
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Impossible de sauvegarder les métriques d'évaluation pour l'entraînement final")

        logger.info("Résultats d'évaluation:")
        logger.info("  - R² score: %.4f", eval_results.get("eval_r2", 0))
        logger.info("  - MAE: %.4f", eval_results.get("eval_mae", 0))
        logger.info("  - RMSE: %.4f", eval_results.get("eval_rmse", 0))

        return eval_results

    def save_model(self) -> None:
        """Save model and tokenizer"""
        logger.info("Sauvegarde du modèle...")

        model_output = self.model_dir / self.score_col / "bert_final"
        model_output.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(str(model_output / "model"))
        self.tokenizer.save_pretrained(str(model_output / "tokenizer"))

        logger.info("Modèle sauvegardé dans: %s", model_output)

    def run(self, csv_path: Path, tuning_results_path: Path = None) -> None:
        """Complete training pipeline"""
        logger.info("Entraînement final: BERT")

        # Load best hyperparameters
        if tuning_results_path is None:
            tuning_results_path = Path("results/bert_tuning_results.csv")

        best_params = self.load_best_hyperparams(tuning_results_path)

        # Load data
        X, y = self.load_data(csv_path)  # pylint: disable=invalid-name

        # Split
        logger.info("Séparation train/test (80/20)...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # pylint: disable=invalid-name
        logger.info("Train: %d, Test: %d", len(X_train), len(X_test))

        # Train with loaded hyperparameters
        Data_X_y = X_train, y_train, X_test, y_test  # pylint: disable=invalid-name
        self.train(Data_X_y, best_params)  # pylint: disable=invalid-name

        # Save model and hyperparameters
        self.save_model()
        self.save_hyperparams(best_params)

        logger.info("Entraînement BERT terminé")


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

    trainer = BertFinalTrainer(
        model_name=args.model_name, model_dir=args.model_dir, score_col=args.target_col, score_scale=args.score_scale
    )
    trainer.run(args.input, args.tuning_results)


if __name__ == "__main__":
    main()
