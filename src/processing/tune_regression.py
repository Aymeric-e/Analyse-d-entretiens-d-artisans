"""
Hyperparameter tuning for verbalization difficulty regression model.
Uses GridSearchCV with cross-validation to find optimal TF-IDF and Ridge parameters.

"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from utils.logger_config import setup_logger

logger = setup_logger(__name__, level="INFO")


class VerbRegHyperparameterTuner:
    """Find optimal hyperparameters for verbalization difficulty model"""

    def __init__(self, model_dir: Path = Path("models")):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.best_model = None
        self.best_vectorizer = None
        self.best_params = None
        self.results = []

    def load_data(self, csv_path: Path) -> tuple:
        """Load annotated CSV with semicolon separator"""
        logger.info("Chargement des données depuis %s...", csv_path)
        df = pd.read_csv(csv_path, sep=";")

        # Validate required columns
        required_cols = ["text", "difficulté_verbalisation"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes dans CSV: {missing}")

        X = df["text"]  # pylint: disable=invalid-name
        y = df["difficulté_verbalisation"].astype(float)

        logger.info("Données chargées: %d phrases", len(df))
        logger.debug("Distribution des difficultés:\n%s", y.value_counts().sort_index().to_string())

        return X, y

    def create_pipeline(self) -> Pipeline:
        """Create sklearn pipeline with TfidfVectorizer and Ridge"""
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
                ("ridge", Ridge(random_state=42)),
            ]
        )
        return pipeline

    def define_grid(self) -> dict:
        """Define hyperparameter grid for search"""
        param_grid = {
            "tfidf__max_features": [1000, 2000, 3000, 4000, 5000],
            "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3), (1, 4)],
            "tfidf__min_df": [1, 2, 3],
            "tfidf__max_df": [0.6, 0.7, 0.8, 0.9],
            "ridge__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        }
        return param_grid

    def tune_hyperparameters(self, X: pd.Series, y: pd.Series, cv: int = 5) -> dict:  # pylint: disable=invalid-name
        """Use GridSearchCV to find best hyperparameters"""
        logger.info("TUNING: Recherche des meilleurs hyperparamètres (%d-fold cross-validation)", cv)

        # Create pipeline and grid
        pipeline = self.create_pipeline()
        param_grid = self.define_grid()

        # GridSearchCV with negative MAE as scoring (sklearn minimizes)
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring="neg_mean_absolute_error",  # Negative because sklearn prefers higher is better
            n_jobs=-1,  # Use all processors
            verbose=1,
        )

        logger.info("Entraînement en cours...")

        # Fit
        grid_search.fit(X, y)

        logger.info("Nombre total de combinaisons à tester: %d", len(list(grid_search.cv_results_["params"])))

        # Store results
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.results = pd.DataFrame(grid_search.cv_results_)

        return {
            "best_params": self.best_params,
            "best_score": -grid_search.best_score_,  # Convert back to positive MAE
            "cv_results": self.results,
        }

    def display_results(self) -> None:
        """Display tuning results"""
        logger.info("RÉSULTATS: Meilleurs hyperparamètres")

        logger.info("Meilleurs paramètres trouvés:")
        for param, value in self.best_params.items():
            logger.info("  - %s: %s", param, value)

        logger.info("Meilleur score MAE: %.4f", -self.results["mean_test_score"].min())

        # Top 5 results
        logger.info("Top 5 des meilleures combinaisons:")
        top_5 = self.results.nsmallest(5, "rank_test_score")[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
        for idx, (_, row) in enumerate(top_5.iterrows()):
            logger.info("%d. MAE: %.4f (±%.4f)", idx + 1, -row["mean_test_score"], row["std_test_score"])
            logger.debug("   Params: %s", row["params"])

    def save_results(self, output_path: Path) -> None:
        """Save tuning results to CSV"""
        logger.info("Sauvegarde des résultats du tuning...")

        # Create simplified results dataframe
        results_simplified = self.results[
            [
                "param_tfidf__max_features",
                "param_tfidf__ngram_range",
                "param_tfidf__min_df",
                "param_tfidf__max_df",
                "param_ridge__alpha",
                "mean_test_score",
                "std_test_score",
                "rank_test_score",
            ]
        ].copy()

        results_simplified["mean_test_score"] = -results_simplified["mean_test_score"]  # Convert back to MAE
        results_simplified = results_simplified.sort_values("rank_test_score")

        results_simplified.to_csv(output_path, index=False, sep=";")
        logger.info("Résultats sauvegardés: %s", output_path)

    def save_best_model(self) -> None:
        """Save best model from tuning"""
        logger.info("Sauvegarde du meilleur modèle...")

        params_path = self.model_dir / "verbalisation_best_params.pkl"

        joblib.dump(self.best_params, params_path)

        logger.info("Paramètres sauvegardés: %s", params_path)

    def run(self, csv_path: Path, output_path: Path, cv: int = 5) -> None:
        """Complete hyperparameter tuning pipeline"""
        logger.info("HYPERPARAMETER TUNING: Verbalization Difficulty Regression")

        # Load
        X, y = self.load_data(csv_path)  # pylint: disable=invalid-name

        # Tune
        _ = self.tune_hyperparameters(X, y, cv=cv)

        # Display
        self.display_results()

        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_results(output_path)

        # Save best model
        self.save_best_model()

        logger.info("Tuning terminé avec succès")


def main():
    """Main function to parse arguments and run hyperparameter tuning"""
    parser = argparse.ArgumentParser(
        description="Trouver les meilleurs hyperparamètres pour le modèle de difficulté de verbalisation"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Chemin du fichier CSV annoté (séparateur: point-virgule)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/hyperparams_tuning.csv"),
        help="Chemin du fichier CSV de sortie avec résultats du tuning (défaut: results/hyperparams_tuning.csv)",
    )
    parser.add_argument("--cv", type=int, default=5, help="Nombre de folds pour cross-validation (défaut: 5)")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Dossier de sauvegarde du modèle (défaut: models)",
    )

    args = parser.parse_args()

    # Validate input file
    if not args.input.exists():
        logger.error("ERREUR: Fichier introuvable: %s", args.input)
        sys.exit(1)

    # Run tuning
    tuner = VerbRegHyperparameterTuner(model_dir=args.model_dir)
    tuner.run(args.input, args.output, cv=args.cv)


if __name__ == "__main__":
    main()
