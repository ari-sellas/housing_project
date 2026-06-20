from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# These columns have too many missing values,
# and aren't worth the trouble of including them.
COLUMNS_TO_DROP = ["Alley", "MasVnrType", "PoolQC", "Fence", "MiscFeature"]
NUMERIC_DTYPES = ["int64", "float64"]
DEFAULT_KNN_NEIGHBORS = 5


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


def _fit_numeric_cleaners(num_df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    original_columns = num_df.columns

    numeric_imputer = _fit_knn_imputer(num_df)
    imputed_df = pd.DataFrame(numeric_knn_imputer.transform(num_df), columns=original_columns)

    ...

def _fit_knn_imputer(df: pd.DataFrame, n_neighbors:int = DEFAULT_KNN_NEIGHBORS) -> KNNImputer:
    return KNNImputer(n_neighbors=n_neighbors).fit(df)


def _fit_scaled_knn_imputer(df: pd.DataFrame, n_neighbors:int = DEFAULT_KNN_NEIGHBORS) -> Pipeline:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn_imputer", _fit_knn_imputer(df, n_neighbors))
    ])
    return pipeline.fit(df)