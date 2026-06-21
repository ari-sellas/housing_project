# Ames Housing Price Predictor
*A stacked ensemble (XGBoost, LightGBM, CatBoost, Random Forest, Ridge) tuned with Optuna*

## Problem & Dataset
This project was developed for the Kaggle "House Prices: Advanced Regression Techniques"
competition (Ames, Iowa housing data). The competition can be found [here](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

## Approach
First, the pipeline cleans and engineers features for both the training and testing data
(log-transforms the target, imputes missing values, etc.). Then, it tunes five base regressors independently with Optuna, and combines them
with a stacking regressor using a Ridge metamodel.

## Project Structure
```text
housing_project/
├── .gitignore
├── csv_files/                             # Contains competition CSV files
│   ├── test.csv                           # Testing data
│   └── train.csv                          # Training data
├── data_cleaning/                         # Cleaning, imputation, etc.
│   └── data_cleaning_functions.py         # Functions for cleaning data
├── ensemble_models/                       # Optuna tuning, stacking, training
│   ├── ensemble_optimizing_functions.py   # Tunes and stacks base models
│   └── ensemble_training.py               # Fits stacked ensemble
├── optuna_database/                       # Directory for Optuna study database
│   └── ensemble_models_study.db           # Stores data from Optuna studies
├── price_prediction/                      # Predicts and stores prices
│   ├── price_predictor.py                 # Entry point: runs the full pipeline
│   └── predicted_prices.csv               # Prices predicted by ensemble
└── README.md
```

## Installation
Requires Python 3.10+ and the following packages:
* pandas
* numpy
* scikit-learn
* xgboost
* lightgbm
* catboost
* optuna
* optunahub

Install them with:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna optunahub
```

## Usage
1. Create a `csv_files` and `optuna_database` folder in the project's root directory.
2. Download `train_csv` and `test_csv` from the [Kaggle competition page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) 
and place them in `csv_files/`.
3. Run the pipeline:
```bash
cd price_prediction
python price_predictor.py
```

Predictions are written to `predicted_prices.csv` in `price_prediction/`.

## Results
* Scored 0.121 RMSE on log-transformed prices. 
* When this project was originally created,
this netted me a top 2% finish in the [Kaggle Learn-oriented](https://www.kaggle.com/competitions/home-data-for-ml-course) 
version of the competition. 
* At the time of writing this, June 20, 2026, this is a top 6% finish in that same version, and a top 13% finish
in the Advanced Regression Techniques competition.

## Notes
* This project started about 8 months ago as a way to put everything I'd learned about ML into practice.
I've revisited it now to clean up the implementation; I've fixed minor bugs, restructured some messy functions, and added
proper documentation.
* I mention it in the ensemble_optimizing_functions document, but LightGBM is the only base model set to n_jobs=1, whereas the others
 utilize multithreading. This is because multithreading LightGBM caused Python to crash on my Mac; feel free to return
to multithreading if your system is more compatible.