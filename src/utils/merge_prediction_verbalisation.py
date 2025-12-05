"""
Merge regression and BERT predictions into a single comparison CSV.
Computes average predictions and comparison metrics.

Usage:
    poetry run python src/utils/merge_prediction_verbalisation.py \
      --regression data/verbalisation/verbalisation_reg.csv \
      --bert data/verbalisation/verbalisation_bert.csv \
      --output results/verbalisation/merged_predictions_comparison.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from utils.logger_config import setup_logger

logger = setup_logger(__name__, level="INFO")


class PredictionsMerger:
    """Merge regression and BERT predictions"""

    def __init__(self):
        self.regression_df = None
        self.bert_df = None
        self.merged_df = None

    def load_predictions(self, regression_csv: Path, bert_csv: Path) -> None:
        """Load both prediction CSV files"""
        logger.info("Chargement des prédictions regression depuis %s...", regression_csv)
        self.regression_df = pd.read_csv(regression_csv, sep=";")
        logger.info("Chargement des prédictions BERT depuis %s...", bert_csv)
        self.bert_df = pd.read_csv(bert_csv, sep=";")

        required_cols_reg = ["filename", "text", "note_regression"]
        required_cols_bert = ["filename", "text", "note_bert"]

        missing_reg = [col for col in required_cols_reg if col not in self.regression_df.columns]
        missing_bert = [col for col in required_cols_bert if col not in self.bert_df.columns]

        if missing_reg:
            raise ValueError(f"Colonnes manquantes dans CSV regression: {missing_reg}")
        if missing_bert:
            raise ValueError(f"Colonnes manquantes dans CSV BERT: {missing_bert}")

        logger.info("Regression: %d prédictions", len(self.regression_df))
        logger.info("BERT: %d prédictions", len(self.bert_df))

    def merge_predictions(self) -> None:
        """Merge predictions from both models"""
        logger.info("Merge des prédictions...")

        self.merged_df = pd.merge(
            self.regression_df[["filename", "text", "note_regression"]],
            self.bert_df[["filename", "text", "note_bert"]],
            on=["filename", "text"],
            how="inner",
        )

        self.merged_df["moyenne"] = (self.merged_df["note_regression"] + self.merged_df["note_bert"]) / 2.0

        self.merged_df["difference_abs"] = abs(self.merged_df["note_regression"] - self.merged_df["note_bert"])

        logger.info("Prédictions mergées: %d phrases", len(self.merged_df))

        self._display_statistics()

    def _display_statistics(self) -> None:
        """Display comparison statistics"""
        logger.info("Statistiques de comparaison:")
        logger.info("  - Prédictions en commun: %d", len(self.merged_df))
        logger.info(
            "  Regression: Moyenne=%.2f, Std=%.2f, Min=%.2f, Max=%.2f",
            self.merged_df["note_regression"].mean(),
            self.merged_df["note_regression"].std(),
            self.merged_df["note_regression"].min(),
            self.merged_df["note_regression"].max(),
        )
        logger.info(
            "  BERT: Moyenne=%.2f, Std=%.2f, Min=%.2f, Max=%.2f",
            self.merged_df["note_bert"].mean(),
            self.merged_df["note_bert"].std(),
            self.merged_df["note_bert"].min(),
            self.merged_df["note_bert"].max(),
        )
        logger.info(
            "  Différences absolues: Moyenne=%.2f, Std=%.2f, Min=%.2f, Max=%.2f",
            self.merged_df["difference_abs"].mean(),
            self.merged_df["difference_abs"].std(),
            self.merged_df["difference_abs"].min(),
            self.merged_df["difference_abs"].max(),
        )

        correlation = self.merged_df["note_regression"].corr(self.merged_df["note_bert"])
        logger.info("  Corrélation Pearson: %.4f", correlation)

    def save_merged(self, output_path: Path) -> None:
        """Save merged predictions"""
        logger.info("Sauvegarde du fichier merged...")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.merged_df = self.merged_df[["filename", "text", "note_regression", "note_bert", "moyenne", "difference_abs"]]

        self.merged_df.to_csv(output_path, index=False, sep=";")

        logger.info("Fichier sauvegardé: %s", output_path)

    def run(self, regression_csv: Path, bert_csv: Path, output_csv: Path) -> None:
        """Complete pipeline"""
        logger.info("Fusion: Prédictions Regression + BERT")

        self.load_predictions(regression_csv, bert_csv)
        self.merge_predictions()
        self.save_merged(output_csv)

        logger.info("Fusion terminée")


def main():
    """Main function to parse arguments and run merger"""
    parser = argparse.ArgumentParser(description="Fusionner les prédictions de regression et BERT")
    parser.add_argument("--regression", type=Path, required=True, help="Chemin du CSV avec prédictions regression")
    parser.add_argument("--bert", type=Path, required=True, help="Chemin du CSV avec prédictions BERT")
    parser.add_argument("--output", type=Path, required=True, help="Chemin du CSV de sortie")

    args = parser.parse_args()

    if not args.regression.exists():
        logger.error("ERREUR: Fichier introuvable: %s", args.regression)
        sys.exit(1)

    if not args.bert.exists():
        logger.error("ERREUR: Fichier introuvable: %s", args.bert)
        sys.exit(1)

    merger = PredictionsMerger()
    merger.run(args.regression, args.bert, args.output)


if __name__ == "__main__":
    main()
