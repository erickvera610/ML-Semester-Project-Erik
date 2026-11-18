# Team 1 model harness: requirements and current status

This document tracks what the harness needs to run and evaluate the required models for the Team 1 sentiment analysis project, and what is present in the repository now.

## Assignment requirements

### Models

- [ ] **Simple baseline:** implement and report a baseline such as majority class.
- [ ] **Classical model:** implement logistic regression.
- [ ] **Neural model:** implement an MLP with 1–3 hidden layers.
- [ ] **Track model:** implement Naive Bayes.
- [ ] Run every required model on the same train, validation, and test split and compare their results.
- [ ] Give each model a comparable hyperparameter tuning budget. Choose configurations using validation data only; never tune on the test set.

### Data and experiment conditions

- [ ] Load labeled review text and labels and check for missing/invalid values, label meanings, and class balance.
- [x] Establish fixed train, validation, and test partitions. Keep the test set untouched until final evaluation. The provided test file is kept separate, and a reproducible, stratified 90/10 train/validation split is created from the training file.
- [ ] Define a primary metric before evaluating models. Record any additional useful metrics, such as precision, recall, F1, and a confusion matrix.
- [ ] Compare models at multiple training-set sizes. Subsets should be reproducible and preserve class proportions; do not rely on taking the first *n* rows.
- [ ] Run one controlled vocabulary-related stress test, such as removing rare words or applying word dropout. State whether it changes training data, test data, or both, and use the same condition and severity for every model.
- [ ] Run the feature ablation: unigram versus unigram-plus-bigram features.
- [ ] Analyze representative errors involving negation, rare words, or ambiguous reviews, and connect those examples to quantitative results.
- [ ] Change one experimental factor at a time from a stated reference condition. Compare all required models in each experiment; testing every possible combination is not required.
- [ ] For stochastic models or procedures, report mean ± standard deviation over at least three runs. A deterministic result may be reported once.

### Shared harness behavior

- [ ] Use a consistent model interface so the harness can fit each model on training data and obtain predictions/scores for validation and test data.
- [ ] Fit text preprocessing and vocabulary only on the relevant training partition, then reuse the fitted transform for validation/test to avoid leakage.
- [ ] Support both unigram and unigram-plus-bigram feature configurations. Ensure the selected vocabulary/stress condition is applied consistently to all models.
- [ ] Centralize split seeds, training-size fractions, stress-test settings, feature settings, and model parameters in recorded configuration.
- [ ] Save structured results containing model, condition, training size, seed, selected settings, and metrics so the report can be reproduced.
- [ ] Handle class labels and predictions consistently across models; keep both labels and original review text available for error analysis.
- [ ] Provide a runnable entry point and README instructions, including environment/package versions and the commands needed to reproduce the experiments.

## Current repository state

### Present

- `src/ml_semester_project/data/train-00000-of-00001.parquet` and `test-00000-of-00001.parquet` are present. Both have 25,000 rows and columns `text` (string) and `label` (integer). The data appears to be a 25,000/25,000 review dataset; verify and record the dataset source and label mapping.
- `src/ml_semester_project/models/logistic_regression.py` exists, but its function currently returns `False` and does not train a model.
- `src/ml_semester_project/models/naive_bayes.py` exists, but its function is a placeholder (`pass`).
- `src/ml_semester_project/models/mlp.py` exists, but its function is a placeholder (`pass`).
- `src/ml_semester_project/__init__.py` has a `training_model` function that calls those three model functions, and a `model_testing` placeholder.
- `main()` loads the parquet files and starts iterating over slices of the training data in 10% increments.
- The project dependencies include scikit-learn, NumPy, and Polars.

### Missing or incomplete

- [ ] No baseline model is implemented.
- [ ] None of the three model functions currently implements training or prediction.
- [ ] `model_testing()` is unimplemented; there is no shared evaluation logic or metrics output.
- [ ] The current data-size loop makes nested 10%-to-100% prefixes from the randomized training partition, but does not train or evaluate models; the prefixes are not guaranteed to preserve class proportions at every size.
- [ ] A reproducible validation split exists, but there is no tuning workflow yet.
- [ ] No text vectorization/preprocessing pipeline, unigram/bigram ablation, or vocabulary stress condition exists yet.
- [ ] No error analysis, repeated-run uncertainty reporting, or structured results logging exists.
- [ ] The README contains only the project title and team label; it does not document setup or reproduction steps.
- [ ] Record the dataset citation/source and label mapping, then confirm whether it satisfies the team's dataset coordination with Team 2.

## Suggested harness run sequence

1. Load and validate the dataset; establish reproducible train/validation/test splits.
2. Fit train-only text vectorizers for unigram and unigram-plus-bigram conditions.
3. Tune each required model on validation data with a matched configuration budget.
4. Evaluate the reference condition for the baseline, logistic regression, MLP, and Naive Bayes.
5. Repeat evaluation for each training size and the selected vocabulary stress condition, changing one factor per run.
6. Run the unigram/bigram ablation, aggregate repeated runs where needed, and save metrics plus predictions for error analysis.
7. Write the chosen settings, seeds, package versions, result locations, and reproduction commands into the README.

## Project-level deliverables to remember

- Presentation: December 9, 2026 (10 minutes plus 5 minutes Q&A per team).
- Team report, code repository, and individual reports: December 18, 2026 at 11:59 p.m.
- The report is up to 8 pages excluding references and appendices, uses the ACL LaTeX style, keeps line numbers, includes a contribution statement, and presents the evidence needed for conclusions and limitations in the main text.
- Each individual submits a reflection of up to 250 words and a confidential peer evaluation.
- Cite the dataset, code, papers, and borrowed ideas; disclose generative AI/external assistance; verify reported results.
