"""
Generate predictions using trained BERT model.
Loads a saved BERT model and generates predictions on new data.

Usage:
    poetry run python src/processing/predict_verbalisation_bert.py \
      --input data/processed/cleaned_sentence.csv \
      --model-dir models/bert_final \
      --output data/verbalisation/verbalisation_bert.csv
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

warnings.filterwarnings("ignore")


class BertPredictor:
    """Generate predictions using trained BERT model"""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilisation du device: {self.device}")

        self._load_model()

    def _load_model(self) -> None:
        """Load saved model and tokenizer"""
        print(f"Chargement du modèle depuis {self.model_dir}...")

        model_path = self.model_dir / "model"
        tokenizer_path = self.model_dir / "tokenizer"

        if not model_path.exists() or not tokenizer_path.exists():
            raise FileNotFoundError(f"Modèle non trouvé dans {self.model_dir}. " f"Veuillez d'abord entraîner avec train_bert.py")

        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path)).to(self.device)

        print("Modèle et tokenizer chargés avec succès")

    def load_data(self, csv_path: Path) -> tuple:
        """Load data to predict on"""
        print(f"Chargement des données depuis {csv_path}...")
        df = pd.read_csv(csv_path, sep=",")

        if "text" not in df.columns:
            raise ValueError("Le CSV doit contenir une colonne 'text'")

        if "filename" not in df.columns:
            raise ValueError("Le CSV doit contenir une colonne 'filename'")

        X = df["text"].astype(str)  # pylint: disable=invalid-name
        filenames = df["filename"].astype(str)

        print(f"Données chargées: {len(df)} phrases")

        return X, filenames

    def tokenize_data(self, texts: pd.Series, max_length: int = 128) -> Dataset:
        """Tokenize texts for prediction"""
        print(f"Tokenisation des textes (max_length={max_length})...")

        encodings = self.tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors=None,
        )

        dataset_dict = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
        }

        return Dataset.from_dict(dataset_dict)

    def predict(self, texts: pd.Series, max_length: int = 128) -> np.ndarray:
        """Generate predictions"""
        print("Génération des prédictions...")

        dataset = self.tokenize_data(texts, max_length)

        training_args = TrainingArguments(
            output_dir="./tmp_predictions",
            per_device_eval_batch_size=32,
            use_cpu=self.device.type == "cpu",
        )

        trainer = Trainer(model=self.model, args=training_args)

        predictions = trainer.predict(dataset)

        predictions = predictions.predictions.squeeze() * 10.0
        predictions = np.clip(predictions, 0, 10).round(2)

        return predictions

    def save_predictions(self, filenames: pd.Series, texts: pd.Series, predictions: np.ndarray, output_path: Path) -> None:
        """Save predictions to CSV"""
        print("Sauvegarde des prédictions...")

        results_df = pd.DataFrame({"filename": filenames.values, "text": texts.values, "note_bert": predictions})

        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False, sep=";")

        print(f"Prédictions sauvegardées: {output_path}")
        print(f"Nombre de prédictions: {len(results_df)}")

    def run(self, csv_path: Path, output_csv: Path, max_length: int = 128) -> None:
        """Complete prediction pipeline"""
        print("Prédiction: BERT pour Difficulté de Verbalisation")

        X, filenames = self.load_data(csv_path)  # pylint: disable=invalid-name
        predictions = self.predict(X, max_length)
        self.save_predictions(filenames, X, predictions, output_csv)

        print("Prédictions terminées!")


def main():
    """Main function to parse arguments and run predictor"""
    parser = argparse.ArgumentParser(description="Générer des prédictions avec le modèle BERT")
    parser.add_argument("--input", type=Path, required=True, help="Chemin du CSV avec textes à prédire")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/bert_final"),
        help="Dossier contenant le modèle BERT",
    )
    parser.add_argument("--max-length", type=int, default=128, help="Max token length")
    parser.add_argument("--output", type=Path, required=True, help="Chemin du CSV de sortie")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERREUR: Fichier introuvable: {args.input}")
        sys.exit(1)

    if not args.model_dir.exists():
        print(f"ERREUR: Dossier modèle introuvable: {args.model_dir}")
        sys.exit(1)

    predictor = BertPredictor(model_dir=args.model_dir)
    predictor.run(args.input, args.output, args.max_length)


if __name__ == "__main__":
    main()
