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

    num_df, cat_df = _split_numerical_and_categorical(uncleaned_df)

    num_df, numeric_imputers = _fit_numerical_cleaners(num_df)
    cat_df, fitted_ohe, categorical_imputer = _fit_categorical_cleaners(cat_df)

    cleaned_df = pd.concat([num_df, cat_df, log_transformed_sale_prices], axis=1)
    imputers = [*numeric_imputers, categorical_imputer]

    return cleaned_df, imputers, fitted_ohe


def clean_testing_data(
        testing_csv_file: str,
        imputers: list,
        fitted_ohe: OneHotEncoder
) -> tuple[pd.DataFrame, pd.Series]:
    uncleaned_df = pd.read_csv(testing_csv_file)
    house_ids = uncleaned_df["Id"]
    uncleaned_df = uncleaned_df.drop(columns=["Id", *COLUMNS_TO_DROP])

    num_df, cat_df = _split_numerical_and_categorical(uncleaned_df)

    numeric_imputer, scaled_numeric_imputer, categorical_imputer = imputers
    ...


def _split_numerical_and_categorical(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    num_df = df.select_dtypes(include=NUMERIC_DTYPES)
    cat_df = df.select_dtypes(include="object")
    return num_df, cat_df


def _fit_numerical_cleaners(num_df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    original_columns = num_df.columns

    numeric_imputer = _fit_knn_imputer(num_df)
    imputed_df = pd.DataFrame(numeric_imputer.transform(num_df), columns=original_columns)

    feature_engineered_df = _numerical_feature_engineering(imputed_df)
    new_columns_only = feature_engineered_df.drop(columns=original_columns)
    combined_df = pd.concat([num_df[original_columns], new_columns_only], axis=1)

    scaled_numeric_imputer = _fit_scaled_knn_imputer(combined_df)
    cleaned_df = pd.DataFrame(
        scaled_numeric_imputer.transform(combined_df), columns=combined_df.columns
    )

    return cleaned_df, [numeric_imputer, scaled_numeric_imputer]


def _use_numerical_cleaners(
        num_df: pd.DataFrame,
        numeric_imputer: KNNImputer,
        scaled_numeric_imputer: Pipeline
) -> pd.DataFrame:
    original_columns = num_df.columns

    imputed_df = pd.DataFrame(numeric_imputer.transform(num_df), columns=original_columns)

    feature_engineered_df = _numerical_feature_engineering(imputed_df)
    new_columns_only = feature_engineered_df.drop(columns=original_columns)
    combined_df = pd.concat([num_df[original_columns], new_columns_only], axis=1)

    cleaned_df = pd.DataFrame(
        scaled_numeric_imputer.transform(combined_df), columns=combined_df.columns
    )

    return cleaned_df


def _fit_categorical_cleaners(
        cat_df: pd.DataFrame
) -> tuple[pd.DataFrame, OneHotEncoder, KNNImputer]:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).set_output(
        transform="pandas"
    )

    encoded_df = ohe.fit_transform(cat_df)

    categorical_imputer = _fit_knn_imputer(encoded_df)
    cleaned_df = pd.DataFrame(
        categorical_imputer.transform(encoded_df), columns=encoded_df.columns
    )

    return cleaned_df, ohe, categorical_imputer


def _numerical_feature_engineering(num_df: pd.DataFrame) -> pd.DataFrame:
    num_df = num_df.copy()
    num_df["TotalLivingSF"] = num_df["TotalBsmtSF"] + num_df["GrLivArea"]
    num_df["TotalPropertySF"] = num_df["TotalLivingSF"] + num_df["GarageArea"]
    num_df["AboveGradeBath"] = num_df["FullBath"] + 0.5 * num_df["HalfBath"]
    num_df["BsmtBath"] = num_df["BsmtFullBath"] + 0.5 * num_df["BsmtHalfBath"]
    num_df["TotalBath"] = num_df["AboveGradeBath"] + num_df["BsmtBath"]
    num_df["QualCond"] = num_df["OverallQual"] + num_df["OverallCond"]
    num_df["TimeBeforeRemod"] = num_df["YearRemodAdd"] - num_df["YearBuilt"]
    num_df["TimeBeforeSell"] = num_df["YrSold"] - num_df["YearBuilt"]
    return num_df


def _fit_knn_imputer(df: pd.DataFrame, n_neighbors:int = DEFAULT_KNN_NEIGHBORS) -> KNNImputer:
    return KNNImputer(n_neighbors=n_neighbors).fit(df)


def _fit_scaled_knn_imputer(df: pd.DataFrame, n_neighbors:int = DEFAULT_KNN_NEIGHBORS) -> Pipeline:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn_imputer", _fit_knn_imputer(df, n_neighbors))
    ])
    return pipeline.fit(df)