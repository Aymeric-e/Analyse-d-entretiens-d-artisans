"""
Train a regression model to predict verbalization difficulty (0-3 scale) from text.
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
    """Train and evaluate difficulty verbalization model"""
    
    def __init__(self, model_dir: Path = Path("models")):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.vectorizer = None
        self.scaler_params = None
    
    def load_data(self, csv_path: Path) -> pd.DataFrame:
        """Load annotated CSV with semicolon separator"""
        print(f"Chargement des données depuis {csv_path}...")
        df = pd.read_csv(csv_path, sep=";")
        
        # Validate required columns
        required_cols = ['text', 'difficulté_verbalisation']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes dans CSV: {missing}")
        
        print(f"Données chargées: {len(df)} phrases")
        print(f"Distribution des difficultés:\n{df['difficulté_verbalisation'].value_counts().sort_index()}")
        
        return df
    
    def prepare_features(self, X_train: pd.Series, X_test: pd.Series) -> tuple:
        """Vectorize text using TF-IDF"""
        print("\nVectorisation TF-IDF...")
        
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            lowercase=True,
            stop_words='english'
        )
        
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        print(f"Nombre de features TF-IDF: {X_train_vec.shape[1]}")
        
        return X_train_vec, X_test_vec
    
    def train(self, X_train_vec, y_train) -> None:
        """Train Ridge regression model"""
        print("\nEntraînement du modèle de régression...")
        
        self.model = Ridge(alpha=1.0, random_state=42)
        self.model.fit(X_train_vec, y_train)
        
        print("Modèle entraîné avec succès.")
    
    def evaluate(self, X_test_vec, y_test) -> dict:
        """Evaluate model performance"""
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
        
        model_path = self.model_dir / "verbalisation_regressor.pkl"
        vectorizer_path = self.model_dir / "verbalisation_vectorizer.pkl"
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        
        print(f"Modèle sauvegardé: {model_path}")
        print(f"Vectorizer sauvegardé: {vectorizer_path}")
    
    def run(self, csv_path: Path) -> None:
        """Complete training pipeline"""
        print("ENTRAINEMENT: Prédiction de la difficulté de verbalisation")
        
        # Load
        df = self.load_data(csv_path)
        X = df['text']
        y = df['difficulté_verbalisation'].astype(float)
        
        # Split
        print(f"\nSéparation train/test (80/20)...")
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
        
        # Save
        self.save_model()
        
        print("Entraînement terminé avec succès!")


def main():
    parser = argparse.ArgumentParser(
        description="Entraîner un modèle de prédiction de difficulté de verbalisation"
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        required=True,
        help="Chemin du fichier CSV annoté (séparateur: point-virgule)"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Dossier de sauvegarde du modèle (défaut: models)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.input_csv.exists():
        print(f"ERREUR: Fichier introuvable: {args.input_csv}")
        sys.exit(1)
    
    # Run training
    trainer = VerbRegTrainer(model_dir=args.model_dir)
    trainer.run(args.input_csv)


if __name__ == "__main__":
    main()