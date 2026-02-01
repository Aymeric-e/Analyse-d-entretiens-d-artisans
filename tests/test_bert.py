"""
Unit tests for BERT verbalization difficulty prediction.

This test module covers critical functionality of the BERT training pipeline
including hyperparameter loading, data handling, and model initialization.

Example:
    Run all tests:
    $ poetry run pytest tests/test_verbalisation_bert.py -v
    
    Run with coverage:
    $ poetry run pytest tests/test_verbalisation_bert.py --cov=src
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from processing.train_bert import BertFinalTrainer


class TestBertFinalTrainer(unittest.TestCase):
    """Test suite for BertFinalTrainer class and hyperparameter management"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.model_dir = Path(self.temp_dir.name)
        self.trainer = BertFinalTrainer(model_dir=self.model_dir)

    def tearDown(self):
        """Clean up after tests"""
        self.temp_dir.cleanup()

    def test_load_default_hyperparams_when_file_missing(self):
        """Test that default hyperparams are loaded when tuning file is missing"""
        missing_path = Path("nonexistent_tuning.csv")
        params = self.trainer.load_best_hyperparams(missing_path)

        self.assertIsInstance(params, dict)
        self.assertEqual(params["learning_rate"], 2e-5)
        self.assertEqual(params["batch_size"], 32)
        self.assertEqual(params["max_length"], 128)
        self.assertEqual(params["num_epochs"], 5)

    def test_load_hyperparams_from_csv(self):
        """Test loading hyperparameters from valid tuning CSV"""
        # Create mock tuning results
        csv_path = Path(self.temp_dir.name) / "tuning.csv"
        df = pd.DataFrame(
            {
                "learning_rate": [2e-5, 5e-5, 1e-5],
                "batch_size": [32, 64, 16],
                "max_length": [128, 256, 128],
                "num_epochs": [5, 10, 3],
                "r2": [0.82, 0.92, 0.75],  # Best R2 in second row
            }
        )
        df.to_csv(csv_path, sep=";", index=False)

        params = self.trainer.load_best_hyperparams(csv_path)

        # Should pick row with highest R2
        self.assertEqual(params["learning_rate"], 5e-5)
        self.assertEqual(params["batch_size"], 64)
        self.assertEqual(params["max_length"], 256)
        self.assertEqual(params["num_epochs"], 10)

    def test_load_hyperparams_with_empty_csv(self):
        """Test that default hyperparams are used when CSV is empty"""
        csv_path = Path(self.temp_dir.name) / "empty_tuning.csv"
        df = pd.DataFrame(columns=["learning_rate", "batch_size", "max_length", "num_epochs", "r2"])
        df.to_csv(csv_path, sep=";", index=False)

        params = self.trainer.load_best_hyperparams(csv_path)

        # Should return defaults since dataframe is empty
        self.assertEqual(params["learning_rate"], 2e-5)

    def test_save_hyperparams_creates_json(self):
        """Test that hyperparameters are saved to JSON file"""
        params = {"learning_rate": 2e-5, "batch_size": 32, "max_length": 128, "num_epochs": 5}

        self.trainer.save_hyperparams(params)

        json_path = self.model_dir / "difficulté_verbalisation" / "bert_final" / "hyperparams.json"
        self.assertTrue(json_path.exists())

        # Verify content

        with open(json_path, "r", encoding="utf-8") as f:
            loaded_params = json.load(f)

        self.assertEqual(loaded_params["learning_rate"], 2e-5)
        self.assertEqual(loaded_params["batch_size"], 32)


class TestDataLoading(unittest.TestCase):
    """Test suite for data loading functionality"""

    def setUp(self):
        """Set up test data"""
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.trainer = BertFinalTrainer(model_dir=Path(self.temp_dir.name))

    def tearDown(self):
        """Clean up"""
        self.temp_dir.cleanup()

    def test_load_data_valid_csv(self):
        """Test loading valid CSV with required columns"""
        csv_path = Path(self.temp_dir.name) / "data.csv"
        df = pd.DataFrame(
            {"text": ["Sample text 1", "Sample text 2", "Sample text 3"], "difficulté_verbalisation": [5.0, 7.5, 3.2]}
        )
        df.to_csv(csv_path, sep=";", index=False)

        X, y = self.trainer.load_data(csv_path)  # pylint: disable=invalid-name

        self.assertEqual(len(X), 3)
        self.assertEqual(len(y), 3)
        self.assertTrue(all(isinstance(val, str) for val in X))
        self.assertTrue(all(isinstance(val, (int, float)) for val in y))

    def test_load_data_missing_text_column_raises_error(self):
        """Test that missing 'text' column raises ValueError"""
        csv_path = Path(self.temp_dir.name) / "bad_data1.csv"
        df = pd.DataFrame(
            {
                "difficulté_verbalisation": [5.0, 7.5]
                # Missing 'text' column
            }
        )
        df.to_csv(csv_path, sep=";", index=False)

        with self.assertRaises(ValueError):
            self.trainer.load_data(csv_path)

    def test_load_data_missing_difficulty_column_raises_error(self):
        """Test that missing 'difficulté_verbalisation' column raises ValueError"""
        csv_path = Path(self.temp_dir.name) / "bad_data2.csv"
        df = pd.DataFrame(
            {
                "text": ["Sample text"]
                # Missing 'difficulté_verbalisation' column
            }
        )
        df.to_csv(csv_path, sep=";", index=False)

        with self.assertRaises(ValueError):
            self.trainer.load_data(csv_path)

    def test_load_data_converts_to_correct_types(self):
        """Test that data types are correctly converted"""
        csv_path = Path(self.temp_dir.name) / "types.csv"
        df = pd.DataFrame({"text": ["Text 1", "Text 2"], "difficulté_verbalisation": ["5.5", "3.2"]})  # String representation
        df.to_csv(csv_path, sep=";", index=False)

        X, y = self.trainer.load_data(csv_path)  # pylint: disable=invalid-name

        # Text should be string
        self.assertIsInstance(X.iloc[0], str)
        # Labels should be float
        self.assertIsInstance(y.iloc[0], (float, np.floating))


class TestHyperparamValidation(unittest.TestCase):
    """Test suite for hyperparameter validation"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.trainer = BertFinalTrainer(model_dir=Path(self.temp_dir.name))

    def tearDown(self):
        """Clean up"""
        self.temp_dir.cleanup()

    def test_hyperparams_are_numeric(self):
        """Test that loaded hyperparameters are numeric"""
        csv_path = Path(self.temp_dir.name) / "tuning.csv"
        df = pd.DataFrame({"learning_rate": [2e-5], "batch_size": [32], "max_length": [128], "num_epochs": [5], "r2": [0.85]})
        df.to_csv(csv_path, sep=";", index=False)

        params = self.trainer.load_best_hyperparams(csv_path)

        self.assertIsInstance(params["learning_rate"], float)
        self.assertIsInstance(params["batch_size"], int)
        self.assertIsInstance(params["max_length"], int)
        self.assertIsInstance(params["num_epochs"], int)

    def test_hyperparams_are_positive(self):
        """Test that hyperparameters have reasonable positive values"""
        csv_path = Path(self.temp_dir.name) / "tuning.csv"
        df = pd.DataFrame({"learning_rate": [2e-5], "batch_size": [32], "max_length": [128], "num_epochs": [5], "r2": [0.85]})
        df.to_csv(csv_path, sep=";", index=False)

        params = self.trainer.load_best_hyperparams(csv_path)

        self.assertGreater(params["learning_rate"], 0)
        self.assertGreater(params["batch_size"], 0)
        self.assertGreater(params["max_length"], 0)
        self.assertGreater(params["num_epochs"], 0)


if __name__ == "__main__":
    unittest.main()
