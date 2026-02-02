"""
Predict verbalization difficulty on new data using trained model.
Outputs predictions scaled 0-10 from model trained on 0-3 scale.

"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from utils.logger_config import setup_logger

logger = setup_logger(__name__, level="INFO")


class VerbRegPredicter:
    """Load trained model and make predictions on new data"""

    def __init__(self, model_dir: Path = Path("models")):
        self.model_dir = model_dir
        self.model = None
        self.vectorizer = None

    def load_model(self) -> None:
        """Load trained model and vectorizer"""
        logger.info("Chargement du modèle entraîné...")

        model_path = self.model_dir / "verbalisation_regressor.pkl"
        vectorizer_path = self.model_dir / "verbalisation_vectorizer.pkl"

        if not model_path.exists() or not vectorizer_path.exists():
            raise FileNotFoundError(
                f"Modèle non trouvé dans {self.model_dir}.\n"
                "Veuillez d'abord entraîner le modèle avec train_verbalisation_regression.py"
            )

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

        logger.info("Modèle chargé depuis: %s", model_path)
        logger.info("Vectorizer chargé depuis: %s", vectorizer_path)

    def load_data(self, csv_path: Path) -> pd.DataFrame:
        """Load test data with comma separator"""
        logger.info("Chargement des données depuis %s...", csv_path)

        # Try comma first, then semicolon
        try:
            df = pd.read_csv(csv_path, sep=",")
        except Exception:  # pylint: disable=broad-except
            df = pd.read_csv(csv_path, sep=";")

        # Validate required columns
        required_cols = ["text"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes dans CSV: {missing}")

        logger.info("Données chargées: %d phrases", len(df))

        return df

    def predict(self, texts: pd.Series) -> pd.Series:
        """Predict difficulty scores and scale to 0-10"""
        logger.info("Prédiction des difficultés de verbalisation...")

        # Vectorize
        X_vec = self.vectorizer.transform(texts)  # pylint: disable=invalid-name

        # Predict (0-10 scale)
        y_pred_0_10 = self.model.predict(X_vec)

        # Clip to valid range and round to 2 decimals
        y_pred_0_10 = y_pred_0_10.clip(0, 10)
        y_pred_0_10 = pd.Series(y_pred_0_10).round(2).values

        logger.info("Prédictions complétées: %d phrases", len(y_pred_0_10))
        logger.info("Plage des scores: [%.2f, %.2f]", float(y_pred_0_10.min()), float(y_pred_0_10.max()))

        return y_pred_0_10

    def save_predictions(self, df: pd.DataFrame, output_path: Path) -> None:
        """Save dataframe with predictions to CSV"""
        logger.info("Sauvegarde des prédictions...")

        # Use semicolon separator as specified
        df.to_csv(output_path, sep=";", index=False)

        logger.info("Résultats sauvegardés: %s", output_path)
        logger.debug("Aperçu des premières lignes:\n%s", df.head(10).to_string())

    def run(self, input_csv: Path, output_csv: Path) -> None:
        """Complete prediction pipeline"""

        logger.info("PRÉDICTION: Difficulté de verbalisation sur nouvelles données")

        # Load model
        self.load_model()

        # Load data
        df = self.load_data(input_csv)

        # Predict
        predictions = self.predict(df["text"])

        # Add predictions to dataframe
        df["note_regression"] = predictions

        # Keep only required columns in right order
        output_cols = ["filename", "text", "word_count", "note_regression"]
        df_output = df[output_cols]

        # Save
        self.save_predictions(df_output, output_csv)

        logger.info("Prédiction terminée avec succès")


def main():
    """Main function to parse arguments and run prediction"""
    parser = argparse.ArgumentParser(description="Prédire la difficulté de verbalisation sur de nouvelles phrases")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Chemin du fichier CSV à prédire (séparateur: virgule)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Chemin du fichier CSV de sortie avec prédictions (séparateur: point-virgule)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Dossier contenant le modèle entraîné (défaut: models)",
    )

    args = parser.parse_args()

    # Validate input file
    if not args.input.exists():
        logger.error("ERREUR: Fichier introuvable: %s", args.input)
        sys.exit(1)

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Run prediction
    predictioner = VerbRegPredicter(model_dir=args.model_dir)
    predictioner.run(args.input, args.output)


if __name__ == "__main__":
    main()
