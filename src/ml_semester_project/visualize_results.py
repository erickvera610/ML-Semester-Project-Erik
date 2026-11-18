import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd


METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision_macro": "Macro precision",
    "recall_macro": "Macro recall",
    "f1_macro": "Macro F1",
}


def _metric_rows(results: pd.DataFrame, evaluation_set: str) -> pd.DataFrame:
    completed = results.loc[
        results["status"].eq("completed"),
        ["model_name", "train_fraction", *METRICS],
    ]
    group_columns = ["model_name", "train_fraction"]
    if evaluation_set == "test":
        group_columns = ["model_name"]

    means = completed.groupby(group_columns, as_index=False)[METRICS].mean()
    rows = means.melt(
        id_vars=group_columns,
        value_vars=METRICS,
        var_name="metric",
        value_name="score",
    )
    rows["metric"] = rows["metric"].map(METRIC_LABELS)
    return rows


def _validation_chart(data: pd.DataFrame, model: str | None = None) -> alt.Chart:
    return (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "train_fraction:Q",
                title="Fraction of training data",
                axis=alt.Axis(format=".0%"),
            ),
            y=alt.Y("score:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "model_name:N",
                title="Model",
                legend=None if model else alt.Undefined,
            ),
            strokeDash=alt.StrokeDash("metric:N", title="Metric"),
            tooltip=[
                alt.Tooltip("model_name:N", title="Model"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("train_fraction:Q", title="Training data", format=".0%"),
                alt.Tooltip("score:Q", title="Mean score", format=".3f"),
            ],
        )
        .properties(
            title="Validation metrics by training size",
            width=650,
            height=400,
        )
    )


def _test_chart(data: pd.DataFrame, model: str | None = None) -> alt.Chart:
    chart = alt.Chart(data).mark_bar()
    if model:
        chart = chart.encode(
            x=alt.X("metric:N", title="Metric"),
            y=alt.Y("score:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("metric:N", title="Metric", legend=None),
            tooltip=[
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("score:Q", title="Mean score", format=".3f"),
            ],
        )
        title = f"Test metrics for {model}"
    else:
        chart = chart.encode(
            x=alt.X("model_name:N", title="Model", sort="-y"),
            xOffset=alt.XOffset("metric:N"),
            y=alt.Y("score:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("metric:N", title="Metric"),
            tooltip=[
                alt.Tooltip("model_name:N", title="Model"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("score:Q", title="Mean score", format=".3f"),
            ],
        )
        title = "Test metrics by model"
    return chart.properties(title=title, width=650, height=400)


def _confusion_rows(
    validation_results: pd.DataFrame,
    test_results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    result_sets = [
        ("validation", validation_results.loc[validation_results["train_fraction"].eq(1.0)]),
        ("test", test_results),
    ]

    for evaluation_set, results in result_sets:
        completed = results.loc[results["status"].eq("completed")]
        for model_name, model_runs in completed.groupby("model_name"):
            matrices = model_runs["confusion_matrix"].dropna().map(json.loads)
            if matrices.empty:
                continue

            mean_matrix = np.mean(np.stack(matrices.to_list()), axis=0)
            for true_label, matrix_row in enumerate(mean_matrix):
                for predicted_label, count in enumerate(matrix_row):
                    rows.append(
                        {
                            "evaluation_set": evaluation_set,
                            "model_name": model_name,
                            "true_label": true_label,
                            "predicted_label": predicted_label,
                            "mean_count": count,
                        }
                    )

    return pd.DataFrame(rows)


def _confusion_chart(data: pd.DataFrame, model: str | None = None) -> alt.Chart:
    heatmap = alt.Chart(data).mark_rect().encode(
        x=alt.X("predicted_label:O", title="Predicted label"),
        y=alt.Y("true_label:O", title="True label"),
        color=alt.Color("mean_count:Q", title="Mean count"),
        tooltip=[
            alt.Tooltip("evaluation_set:N", title="Evaluation set"),
            alt.Tooltip("model_name:N", title="Model"),
            alt.Tooltip("true_label:O", title="True label"),
            alt.Tooltip("predicted_label:O", title="Predicted label"),
            alt.Tooltip("mean_count:Q", title="Mean count", format=".1f"),
        ],
    )
    counts = alt.Chart(data).mark_text(fontSize=16).encode(
        x="predicted_label:O",
        y="true_label:O",
        text=alt.Text("mean_count:Q", format=".0f"),
    )
    chart = (heatmap + counts).properties(
        title="Confusion matrices averaged over repeat seeds",
        width=220,
        height=180,
    )
    if model:
        return chart.facet(row=alt.Row("evaluation_set:N", title=None))
    return chart.facet(
        row=alt.Row("evaluation_set:N", title=None),
        column=alt.Column("model_name:N", title=None),
    )


def _save_chart(chart: alt.Chart, output_dir: Path, filename: str) -> Path:
    chart_path = output_dir / filename
    chart.save(chart_path)
    return chart_path


def save_result_charts(
    validation_results: pd.DataFrame,
    test_results: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_metrics = _metric_rows(validation_results, "validation")
    test_metrics = _metric_rows(test_results, "test")
    confusion_data = _confusion_rows(validation_results, test_results)
    model_names = sorted(
        set(validation_metrics["model_name"]) | set(test_metrics["model_name"])
    )
    chart_paths = []

    if not validation_metrics.empty:
        chart_paths.append(
            _save_chart(
                _validation_chart(validation_metrics),
                output_dir,
                "master-validation-metrics.html",
            )
        )
    if test_metrics["model_name"].nunique() >= 2:
        chart_paths.append(
            _save_chart(
                _test_chart(test_metrics),
                output_dir,
                "master-test-metrics.html",
            )
        )
    else:
        (output_dir / "master-test-metrics.html").unlink(missing_ok=True)
    if not confusion_data.empty:
        chart_paths.append(
            _save_chart(
                _confusion_chart(confusion_data),
                output_dir,
                "master-confusion-matrices.html",
            )
        )

    for model_name in model_names:
        validation_data = validation_metrics.loc[
            validation_metrics["model_name"].eq(model_name)
        ]
        test_data = test_metrics.loc[test_metrics["model_name"].eq(model_name)]
        model_confusion = confusion_data.loc[
            confusion_data["model_name"].eq(model_name)
        ]
        filename = model_name.replace("_", "-")

        if not validation_data.empty:
            chart_paths.append(
                _save_chart(
                    _validation_chart(validation_data, model_name),
                    output_dir,
                    f"{filename}-validation-metrics.html",
                )
            )
        if not test_data.empty:
            chart_paths.append(
                _save_chart(
                    _test_chart(test_data, model_name),
                    output_dir,
                    f"{filename}-test-metrics.html",
                )
            )
        if not model_confusion.empty:
            chart_paths.append(
                _save_chart(
                    _confusion_chart(model_confusion, model_name),
                    output_dir,
                    f"{filename}-confusion-matrices.html",
                )
            )

    return chart_paths
