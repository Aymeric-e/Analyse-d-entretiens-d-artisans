"""
Unit tests for prediction merging utilities.

Tests the PredictionsMerger class from src/utils/merge_prediction_verbalisation.py.
Includes tests for:
- merge_predictions: Validates merging of regression and BERT predictions
- Verifies computation of average predictions and absolute differences
"""

import pandas as pd

from utils.merge_prediction_verbalisation import PredictionsMerger


def test_merge_predictions_basic():
    """
    Test basic merge functionality for regression and BERT predictions.

    Creates two DataFrames with regression and BERT predictions, merges them,
    and validates that:
    - Average predictions are correctly computed
    - Absolute differences between models are correctly calculated
    """
    reg = pd.DataFrame(
        {
            "filename": ["f1", "f2"],
            "text": ["phrase a", "phrase b"],
            "note_regression": [4.0, 6.0],
        }
    )

    bert = pd.DataFrame(
        {
            "filename": ["f1", "f2"],
            "text": ["phrase a", "phrase b"],
            "note_bert": [5.0, 7.0],
        }
    )

    merger = PredictionsMerger()
    merger.regression_df = reg
    merger.bert_df = bert

    merger.merge_predictions()

    m = merger.merged_df
    assert "moyenne" in m.columns
    assert list(m["moyenne"]) == [4.5, 6.5]
    assert list(m["difference_abs"]) == [1.0, 1.0]
