"""
Hyperparameter tuning for verbalization difficulty regression model.
Uses GridSearchCV with cross-validation to find optimal TF-IDF and Ridge parameters.

"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


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
        print(f"Chargement des données depuis {csv_path}...")
        df = pd.read_csv(csv_path, sep=";")
        
        # Validate required columns
        required_cols = ['text', 'difficulté_verbalisation']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes dans CSV: {missing}")
        
        X = df['text']
        y = df['difficulté_verbalisation'].astype(float)
        
        print(f"Données chargées: {len(df)} phrases")
        print(f"Distribution des difficultés:\n{y.value_counts().sort_index()}")
        
        return X, y
    
    def create_pipeline(self) -> Pipeline:
        """Create sklearn pipeline with TfidfVectorizer and Ridge"""
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                lowercase=True,
                stop_words='english'
            )),
            ('ridge', Ridge(random_state=42))
        ])
        return pipeline
    
    def define_grid(self) -> dict:
        """Define hyperparameter grid for search"""
        param_grid = {
            'tfidf__max_features': [1000, 2000, 3000, 4000, 5000],
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
            'tfidf__min_df': [1, 2, 3],
            'tfidf__max_df': [0.6, 0.7, 0.8, 0.9],
            'ridge__alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
        }
        return param_grid
    
    def tune_hyperparameters(self, X: pd.Series, y: pd.Series, cv: int = 5) -> dict:
        """Use GridSearchCV to find best hyperparameters"""
        print(f"\nTUNING: Recherche des meilleurs hyperparamètres ({cv}-fold cross-validation)")
        
        # Create pipeline and grid
        pipeline = self.create_pipeline()
        param_grid = self.define_grid()
        
        # GridSearchCV with negative MAE as scoring (sklearn minimizes)
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='neg_mean_absolute_error',  # Negative because sklearn prefers higher is better
            n_jobs=-1,  # Use all processors
            verbose=1
        )
        
        print("\nEntraînement en cours...")
        
        # Fit
        grid_search.fit(X, y)

        print(f"\nNombre total de combinaisons à tester: {len(list(grid_search.cv_results_['params']))}")

        
        # Store results
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.results = pd.DataFrame(grid_search.cv_results_)
        
        return {
            'best_params': self.best_params,
            'best_score': -grid_search.best_score_,  # Convert back to positive MAE
            'cv_results': self.results
        }
    
    def display_results(self) -> None:
        """Display tuning results"""
        print("\nRÉSULTATS: Meilleurs hyperparamètres")
        
        print("\nMeilleurs paramètres trouvés:")
        for param, value in self.best_params.items():
            print(f"  - {param}: {value}")
        
        print(f"\nMeilleur score MAE (5-fold CV): {-self.results['mean_test_score'].min():.4f}")
        
        # Top 5 results
        print("\nTop 5 des meilleures combinaisons:")
        top_5 = self.results.nsmallest(5, 'rank_test_score')[
            ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
        ]
        for idx, (_, row) in enumerate(top_5.iterrows()):
            print(f"\n  {idx+1}. MAE: {-row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")
            print(f"     Params: {row['params']}")
    
    def save_results(self, output_path: Path) -> None:
        """Save tuning results to CSV"""
        print(f"\nSauvegarde des résultats du tuning...")
        
        # Create simplified results dataframe
        results_simplified = self.results[[
            'param_tfidf__max_features',
            'param_tfidf__ngram_range',
            'param_tfidf__min_df',
            'param_tfidf__max_df',
            'param_ridge__alpha',
            'mean_test_score',
            'std_test_score',
            'rank_test_score'
        ]].copy()
        
        results_simplified['mean_test_score'] = -results_simplified['mean_test_score']  # Convert back to MAE
        results_simplified = results_simplified.sort_values('rank_test_score')
        
        results_simplified.to_csv(output_path, index=False, sep=";")
        print(f"Résultats sauvegardés: {output_path}")
    
    def save_best_model(self) -> None:
        """Save best model from tuning"""
        print(f"\nSauvegarde du meilleur modèle...")
        
        params_path = self.model_dir / "verbalisation_best_params.pkl"
        
        joblib.dump(self.best_params, params_path)
        
        print(f"Paramètres sauvegardés: {params_path}")
    
    def run(self, csv_path: Path, output_path: Path, cv: int = 5) -> None:
        """Complete hyperparameter tuning pipeline"""
        print("HYPERPARAMETER TUNING: Verbalization Difficulty Regression")
        
        # Load
        X, y = self.load_data(csv_path)
        
        # Tune
        results = self.tune_hyperparameters(X, y, cv=cv)
        
        # Display
        self.display_results()
        
        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_results(output_path)
        
        # Save best model
        self.save_best_model()
        
        print("Tuning terminé avec succès!")


def main():
    parser = argparse.ArgumentParser(
        description="Trouver les meilleurs hyperparamètres pour le modèle de difficulté de verbalisation"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Chemin du fichier CSV annoté (séparateur: point-virgule)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/hyperparams_tuning.csv"),
        help="Chemin du fichier CSV de sortie avec résultats du tuning (défaut: results/hyperparams_tuning.csv)"
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Nombre de folds pour cross-validation (défaut: 5)"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Dossier de sauvegarde du modèle (défaut: models)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.input.exists():
        print(f"ERREUR: Fichier introuvable: {args.input}")
        sys.exit(1)
    
    # Run tuning
    tuner = VerbRegHyperparameterTuner(model_dir=args.model_dir)
    tuner.run(args.input, args.output, cv=args.cv)


if __name__ == "__main__":
    main()