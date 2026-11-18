from pathlib import Path

import pandas as pd

from .experiment_runner import (
    EXPERIMENT_PLAN,
    run_final_test_evaluations,
    run_validation_experiments,
)
from .visualize_results import save_result_charts


def main() -> None:
    data_dir = Path(__file__).parent / "data"
    training_data = pd.read_parquet(
        data_dir / "train-00000-of-00001.parquet"
    ).reset_index().rename(columns={"index": "review_id"})
    validation_results, validation_predictions = run_validation_experiments(
        training_data,
        EXPERIMENT_PLAN,
    )

    results_dir = Path(__file__).parent / "output"
    results_dir.mkdir(parents=True, exist_ok=True)
    validation_results.to_csv(results_dir / "validation-results.csv", index=False)
    validation_predictions.to_parquet(
        results_dir / "validation-predictions.parquet",
        index=False,
    )

    test_data = pd.read_parquet(
        data_dir / "test-00000-of-00001.parquet"
    ).reset_index().rename(columns={"index": "review_id"})
    test_results, test_predictions = run_final_test_evaluations(
        training_data,
        test_data,
        EXPERIMENT_PLAN,
    )
    test_results.to_csv(results_dir / "test-results.csv", index=False)
    test_predictions.to_parquet(
        results_dir / "test-predictions.parquet",
        index=False,
    )

    chart_paths = save_result_charts(
        validation_results,
        test_results,
        results_dir,
    )

    skipped_runs = (test_results["status"] == "not_implemented").sum()
    completed_runs = len(test_results) - skipped_runs
    print(
        f"Test evaluation finished for {completed_runs} model runs; "
        f"skipped {skipped_runs} unimplemented models."
    )
    print(f"Saved validation results to {results_dir / 'validation-results.csv'}")
    print(f"Saved test results to {results_dir / 'test-results.csv'}")
    for chart_path in chart_paths:
        print(f"Saved chart to {chart_path}")
