"""
Train regression model with best hyperparameters found by tuning.
Uses optimal TF-IDF and Ridge parameters to train on all training data.
Outputs a trained model that can predict on new data with a 0-10 scale.
 
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


class VerbRegTrainer:
    """Train model with best hyperparameters found by tuning"""
    
    def __init__(self, model_dir: Path = Path("models")):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.vectorizer = None
        self.best_params = None
    
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
    
    def load_best_params(self, params_path: Path) -> dict:
        """Load best hyperparameters from tuning"""
        print(f"\nChargement des meilleurs paramètres depuis {params_path}...")
        
        if not params_path.exists():
            raise FileNotFoundError(
                f"Fichier de paramètres non trouvé: {params_path}\n"
                f"Veuillez d'abord lancer le tuning avec tune_verbalisation_hyperparams.py"
            )
        
        best_params = joblib.load(params_path)
        
        print("Meilleurs paramètres chargés:")
        for param, value in best_params.items():
            print(f"  - {param}: {value}")
        
        return best_params
    
    def extract_tfidf_params(self) -> dict:
        """Extract TF-IDF parameters from best_params dict"""
        tfidf_params = {}
        for key, value in self.best_params.items():
            if key.startswith('tfidf__'):
                param_name = key.replace('tfidf__', '')
                tfidf_params[param_name] = value
        
        # Add French stop_words if not in params
        if 'stop_words' not in tfidf_params:
            tfidf_params['stop_words'] = 'english'
        
        return tfidf_params
    
    def extract_ridge_params(self) -> dict:
        """Extract Ridge parameters from best_params dict"""
        ridge_params = {'random_state': 42}
        for key, value in self.best_params.items():
            if key.startswith('ridge__'):
                param_name = key.replace('ridge__', '')
                ridge_params[param_name] = value
        
        return ridge_params
    
    def prepare_features(self, X_train: pd.Series, X_test: pd.Series = None) -> tuple:
        """Vectorize text using TF-IDF with best parameters"""
        print("\nVectorisation TF-IDF avec paramètres optimaux...")
        
        tfidf_params = self.extract_tfidf_params()
        
        self.vectorizer = TfidfVectorizer(**tfidf_params)
        
        X_train_vec = self.vectorizer.fit_transform(X_train)
        
        if X_test is not None:
            X_test_vec = self.vectorizer.transform(X_test)
            print(f"Nombre de features TF-IDF: {X_train_vec.shape[1]}")
            return X_train_vec, X_test_vec
        else:
            print(f"Nombre de features TF-IDF: {X_train_vec.shape[1]}")
            return X_train_vec, None
    
    def train(self, X_train_vec, y_train) -> None:
        """Train Ridge regression model with best parameters"""
        print("\nEntraînement du modèle de régression avec paramètres optimaux...")
        
        ridge_params = self.extract_ridge_params()
        
        self.model = Ridge(**ridge_params)
        self.model.fit(X_train_vec, y_train)
        
        print("Modèle entraîné avec succès.")
    
    def evaluate(self, X_test_vec, y_test) -> dict:
        """Evaluate model performance on test set"""
        print("\nÉvaluation du modèle...")
        
        y_pred = self.model.predict(X_test_vec)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Résultats sur l'ensemble de test:")
        print(f"  - R² score: {r2:.4f}")
        print(f"  - MAE: {mae:.4f}")
        print(f"  - RMSE: {rmse:.4f}")
        
        return {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'mse': mse
        }
    
    def save_model(self) -> None:
        """Save trained model and vectorizer"""
        print("\nSauvegarde du modèle...")
        
        model_path = self.model_dir / f"verbalisation_regressor.pkl"
        vectorizer_path = self.model_dir / f"verbalisation_vectorizer.pkl"
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        
        print(f"Modèle sauvegardé: {model_path}")
        print(f"Vectorizer sauvegardé: {vectorizer_path}")
    
    def run(self, csv_path: Path, best_params_path: Path, with_eval: bool = True) -> None:
        """Complete training pipeline with best hyperparameters"""
        
        print("ENTRAINEMENT : Avec hyperparamètres optimaux")
        
        
        # Load best params
        self.best_params = self.load_best_params(best_params_path)
        
        # Load data
        X, y = self.load_data(csv_path)
        
        if with_eval:
            # Split pour évaluation
            print(f"\nSéparation train/test (80/20) pour évaluation...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42
            )
            print(f"Train: {len(X_train)}, Test: {len(X_test)}")
            
            # Features
            X_train_vec, X_test_vec = self.prepare_features(X_train, X_test)
            
            # Train
            self.train(X_train_vec, y_train)
            
            # Evaluate
            self.evaluate(X_test_vec, y_test)
        else:
            # Entraîner sur tout le dataset
            print(f"\nEntraînement sur tout le dataset ({len(X)} phrases)...")
            
            # Features
            X_train_vec, _ = self.prepare_features(X)
            
            # Train
            self.train(X_train_vec, y)
        
        # Save
        self.save_model()
        
        
        print("Entraînement terminé avec succès!")
        


def main():
    parser = argparse.ArgumentParser(
        description="Entraîner le modèle avec les meilleurs hyperparamètres trouvés"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Chemin du fichier CSV annoté (séparateur: point-virgule)"
    )
    parser.add_argument(
        "--best-params",
        type=Path,
        required=True,
        help="Chemin du fichier pickle contenant les meilleurs paramètres (de tune_verbalisation_hyperparams.py)"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Dossier de sauvegarde du modèle (défaut: models)"
    )
    parser.add_argument(
        "--with-eval",
        action="store_true",
        default=True,
        help="Évaluer sur un jeu test (défaut: True). Utiliser --no-eval pour entraîner sur tout"
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Entraîner sur tout le dataset sans jeu test (défaut: False)"
    )
    
    args = parser.parse_args()
    
    # Validate input files
    if not args.input.exists():
        print(f"ERREUR: Fichier introuvable: {args.input}")
        sys.exit(1)
    
    if not args.best_params.exists():
        print(f"ERREUR: Fichier de paramètres introuvable: {args.best_params}")
        sys.exit(1)
    
    # Determine if we evaluate
    with_eval = not args.no_eval
    
    # Run training
    trainer = VerbRegTrainer(model_dir=args.model_dir)
    trainer.run(args.input, args.best_params, with_eval=with_eval)


if __name__ == "__main__":
    main()