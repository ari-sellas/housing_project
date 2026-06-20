from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# These columns have too many missing values,
# and aren't worth the trouble of including them.
COLUMNS_TO_DROP = ["Alley", "MasVnrType", "PoolQC", "Fence", "MiscFeature"]
NUMERIC_DTYPES = ["int64", "float64"]

def clean_training_data(training_csv_file: str) -> tuple[pd.DataFrame, list, OneHotEncoder]:
    uncleaned_df = pd.read_csv(training_csv_file)
    uncleaned_df = uncleaned_df.drop(columns=["Id", *COLUMNS_TO_DROP])

    log_transformed_sale_prices = np.log(uncleaned_df["SalePrice"])
    uncleaned_df = uncleaned_df.drop(columns="SalePrice")

    ...

def _split_numerical_and_categorical(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    num_df = df.select_dtypes(include=NUMERIC_DTYPES)
    cat_df = df.select_dtypes(include="object")
    return num_df, cat_df