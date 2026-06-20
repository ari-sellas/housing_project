"""Hyperparameter tuning and stacking for the housing price base models.

Each base regressor is tuned in its own Optuna study. All five are then
combined into one StackingRegressor with a Ridge metamodel.
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"

# Note to self: use these in future projects.
# @dataclass removes the need for an __init__ method
# since it will generate the constructor by default,
# and field can prevent shared mutable bugs.
from dataclasses import dataclass, field

# Note to self: also use this in future projects.
# It allows you to type hint a parameter that must be a function.
from collections.abc import Callable

import optuna
import optunahub
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

RANDOM_STATE = 42
CV_FOLDS = 10

@dataclass
class ModelSpec:
    """Everything needed to instantiate a base model.

    Attributes
    ----------
    name : str
        Name used for the Optuna study file, as well as the name
        the model is registered under in StackingRegressor.
    model_class : type
        The regressor class to be tuned.
    param_space : Callable[[optuna.trial.Trial], dict]
        The hyperparameter dict to use in Optuna trials.
    fixed_params : dict
        Extra kwargs applied on every trial and the final fit.
        These are not being tuned.
    """
    name: str
    model_class: type
    param_space: Callable[[optuna.trial.Trial], dict]
    fixed_params: dict = field(default_factory=dict)


def _xgb_param_space(trial: optuna.trial.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 4, 7),
        "min_child_weight": trial.suggest_int("min_child_weight", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.7, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "n_estimators": trial.suggest_int("n_estimators", 1000, 2000),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 10),
    }


def _lightgbm_param_space(trial: optuna.trial.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 1000, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "max_depth": trial.suggest_int("max_depth", 4, 7),
        "num_leaves": trial.suggest_int("num_leaves", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.7, 0.9),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.6),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 10),
    }


def _catboost_param_space(trial: optuna.trial.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 4, 7),
        "subsample": trial.suggest_float("subsample", 0.7, 0.9),
        "random_strength": trial.suggest_float("random_strength", 1e-8, 1, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "n_estimators": trial.suggest_int("n_estimators", 1000, 2000),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 20),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 0.8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
    }


def _random_forest_param_space(trial: optuna.trial.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 500, 1000),
        "max_depth": trial.suggest_int("max_depth", 10, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7]),
    }


def _ridge_param_space(trial: optuna.trial.Trial) -> dict:
    return {
        "alpha": trial.suggest_float("alpha", 1e-5, 1e3, log=True),
    }


BASE_MODEL_SPECS = [
    ModelSpec("xgb_regressor", XGBRegressor, _xgb_param_space, {"n_jobs": -1}),
    ModelSpec("lightgbm_regressor", LGBMRegressor, _lightgbm_param_space, {"n_jobs": -1, "verbose": -1}),
    ModelSpec("catboost_regressor", CatBoostRegressor, _catboost_param_space, {"thread_count": -1, "verbose": -1}),
    ModelSpec("random_forest", RandomForestRegressor, _random_forest_param_space, {"n_jobs": -1}),
    ModelSpec("ridge", Ridge, _ridge_param_space, {})
]

class SeparateModelOptimization:
    """Tunes a set of base regressors with Optuna, then stacks them.

    Attributes
    ----------
    X : pd.DataFrame
        Training data.
    y : pd.Series
        Training target.
    n_trials : int
        Number of Optuna trials run per base model.
        Note: 100 of these trials is extremely computationally expensive;
        if you decide to run this program yourself, feel free to turn that
        number down a bit.
    model_specs : list[ModelSpec], optional
        The list of base models defined earlier, or
        an optional alternative if you choose
        to clone the repo.
    """
    def __init__(
            self,
            X: pd.DataFrame,
            y: pd.DataFrame,
            model_specs: list[ModelSpec] | None = None,
            n_trials: int = 100
    ):
        self.X = X
        self.y = y
        self.model_specs = model_specs if model_specs is not None else BASE_MODEL_SPECS
        self.n_trials = n_trials
        self.optimized_models: list[tuple[str, object]] = []

    def _optimize_model(self, spec: ModelSpec) -> None:
        """Runs an Optuna study, and stores the best-fit model."""
        def objective(trial: optuna.trial.Trial) -> float:
            model = spec.model_class(
                **spec.param_space(trial), **spec.fixed_params, random_state=RANDOM_STATE
            )
            return cross_val_score(
                model, self.X, self.y, cv=CV_FOLDS, scoring="neg_mean_squared_error", n_jobs=-1
            ).mean()

        # To remind myself of what these do in the future:
        # Storage tracks experiment history in a database to prevent data loss
        # and share trial data across optimization workers.
        storage = optuna.storages.RDBStorage(url="sqlite:///../optuna_database/ensemble_models_study.db")

        # Auto sampler selects an appropriate sampler for the search space
        # (a sampler executes a specific hyperparameter optimization algorithm).
        sampler_module = optunahub.load_module(package="samplers/auto_sampler")

        # A study takes the objective you want to minimize or maximize
        # (in this case, maximize our MSE since it's negative), and conducts many
        # trials to find the best possible parameters.
        study = optuna.create_study(
            direction="maximize", storage=storage, sampler=sampler_module.AutoSampler(seed=RANDOM_STATE)
        )
        study.optimize(
            objective, n_trials=self.n_trials, show_progress_bar=True, gc_after_trial=True, n_jobs=-1
        )

        best_model = spec.model_class(
            **study.best_params, **spec.fixed_params, random_state=RANDOM_STATE
        ).fit(self.X, self.y)
        self.optimized_models.append((spec.name, best_model))

    def optimize_all_base_models(self) -> None:
        """Runs Optuna tuning for every model in self.model_specs."""
        for model_spec in self.model_specs:
            self._optimize_model(model_spec)

    def stacking_regressor_model(self) -> StackingRegressor:
        """Tunes all base models, and stacks them with a Ridge metamodel."""
        self.optimize_all_base_models()

        stacking_regressor = StackingRegressor(
            estimators=self.optimized_models,
            final_estimator=Ridge(random_state=RANDOM_STATE),
            cv=CV_FOLDS
        )
        return stacking_regressor.fit(self.X, self.y)