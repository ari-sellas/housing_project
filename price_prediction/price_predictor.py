"""Cleans training and testing data. Tunes, stacks, and fits base models to predict prices.


clean_training_data is called to clean training data and fit imputers to be used
in clean_testing_data, which cleans testing data and stores house Ids for later use.
Then, the ensemble models are optimized with Optuna, stacked with a Ridge metamodel, and
fit on the clean training data. The clean testing data is then given to the stacked
ensemble to predict prices. House Ids are reattached so the CSV is ready for submission.
"""

import numpy as np
import pandas as pd
from ensemble_models.ensemble_training import ensemble_models_training
from data_cleaning.data_cleaning_functions import clean_training_data, clean_testing_data
import warnings
warnings.filterwarnings("ignore")


def main() -> None:
    """Executes the complete machine learning pipeline."""
    cleaned_training_df, imputers, fitted_ohe = clean_training_data("../csv_files/train.csv")
    cleaned_testing_df, house_ids = clean_testing_data("../csv_files/test.csv", imputers, fitted_ohe)

    final_predictor = ensemble_models_training(cleaned_training_df)
    price_prediction = {"Id": house_ids, "SalePrice": np.exp(final_predictor.predict(cleaned_testing_df))}

    pd.DataFrame(price_prediction).to_csv(path_or_buf="predicted_prices.csv", index=False)


if __name__ == "__main__":
    main()