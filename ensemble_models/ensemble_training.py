"""Optimizes models and unifies them in a single fitted stacking ensemble.

Takes already-cleaned training data and returns a tuned and fitted StackingRegressor,
built from the base models in ensemble_optimizing_functions.
"""

from ensemble_models.ensemble_optimizing_functions import SeparateModelOptimization as SMO
import pandas as pd
from sklearn.ensemble import StackingRegressor

def ensemble_models_training(cleaned_training_df: pd.DataFrame) -> StackingRegressor:
    """Tunes the base models and fits a stacking regressor on training data.

    Parameters
    ----------
    cleaned_training_df : pd.DataFrame
        Output of clean_training_data, includes the SalePrice target column.

    Returns
    -------
    StackingRegressor
        The fitted ensemble.
    """
    X = cleaned_training_df.drop(columns=["SalePrice"])
    y = cleaned_training_df["SalePrice"]

    smo = SMO(X=X, y=y)

    return smo.stacking_regressor_model()