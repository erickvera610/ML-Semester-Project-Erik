from operator import length_hint

from sklearn.model_selection import train_test_split

import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path
from .models.naive_bayes import naive_bayes
from .models.mlp import mlp
from .models.logistic_regression import logistic_regression


def model_testing(test_data):
    pass


def training_model(data) -> None:
    naive_bayes(data)
    mlp(data)
    logistic_regression(data)


def run_learning_at_size(training_size, training_data, validation_data):
    pass


def main() -> None:
    print("Hello from ml-semester-project!")
    print("harness starts here")

    DATA_DIR = Path(__file__).parent / "data"
    training_data = pl.read_parquet(DATA_DIR / "train-00000-of-00001.parquet")
    test_data = pl.read_parquet(DATA_DIR / "test-00000-of-00001.parquet")

    print(training_data.null_count())

    print(len(training_data))
    length_of_dataframe = len(training_data)
    # vaidation_data = training_data.slice(int(length_of_dataframe * 0.9))
    # training_subset = training_data.slice(0, int(length_of_dataframe * 0.9))

    labels = training_data["label"].to_numpy()
    row_indices = np.arange(training_data.height)

    train_indices, validation_indices = train_test_split(
        row_indices,
        test_size=0.1,
        random_state=420,
        stratify=labels,
    )

    print(validation_indices)

    train_pool = training_data[train_indices]
    validation_data = training_data[validation_indices]
    print(train_pool)
    print(validation_data)
    l = len(train_pool)
    for i in range(1, 11):
        subset = train_pool.slice(0, int(l * (i / 10)))
        print(len(subset))
        run_learning_at_size(i / 10, subset, validation_data)

    # for i in range(0, 110, 10):
    #     subset = training_data.slice(0, int(length_of_dataframe * (i / 100)))
    #     length_of_subset = len(subset)
    #     # training split
    #     training_subset = subset.slice(0, int(length_of_subset * 0.9))
    #     # validation split
    #     validation_subset = subset.slice(int(length_of_subset * 0.9 + 1))
    #     training_model(training_subset)
    #
    # for fraction in range(1, 11):
    #     size = len(training_subset) * fraction // 10
    #     model_training_data = training_subset.sample(
    #         size, with_replacement=False, shuffle=True, seed=420
    #     )
    # data ready to send into models
