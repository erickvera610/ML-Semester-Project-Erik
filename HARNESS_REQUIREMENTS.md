# Team 1 model harness: requirements and current status

This document tracks what the harness needs to run and evaluate the required models for the Team 1 sentiment analysis project, and what is present in the repository now.

## Assignment requirements

### Models

- [x] **Simple baseline:** the majority-class baseline ran; its results are saved in `src/ml_semester_project/output/validation-results.csv` and `test-results.csv`.
- [ ] **Classical model:** implement logistic regression.
- [ ] **Neural model:** implement an MLP with 1–3 hidden layers.
- [ ] **Track model:** implement Naive Bayes.
- [ ] Run every required model on the same train, validation, and test split and compare their results.
- [ ] Give each model a comparable hyperparameter tuning budget. Choose configurations using validation data only; never tune on the test set.

### Data and experiment conditions

- [ ] Load labeled review text and labels and check for missing/invalid values, label meanings, and class balance.
- [x] Establish fixed train, validation, and test partitions. Keep the test set untouched until final evaluation. The training file gets a reproducible, stratified 90/10 split; the separate test file is read only after validation runs.
- [ ] Define a primary metric before evaluating models. Record any additional useful metrics, such as precision, recall, F1, and a confusion matrix.
- [ ] Compare models at multiple training-set sizes. Subsets should be reproducible and preserve class proportions; do not rely on taking the first *n* rows.
- [ ] Run one controlled vocabulary-related stress test, such as removing rare words or applying word dropout. State whether it changes training data, test data, or both, and use the same condition and severity for every model.
- [ ] Run the feature ablation: unigram versus unigram-plus-bigram features.
- [ ] Analyze representative errors involving negation, rare words, or ambiguous reviews, and connect those examples to quantitative results.
- [ ] Change one experimental factor at a time from a stated reference condition. Compare all required models in each experiment; testing every possible combination is not required.
- [ ] For stochastic models or procedures, report mean ± standard deviation over at least three runs. A deterministic result may be reported once.

### Shared harness behavior

- [x] Use a consistent model-runner interface: each model module receives training and evaluation data, trains its model, and returns predictions.
- [ ] Each text model must fit preprocessing and vocabulary on its training partition, then apply the fitted transform to validation/test data to avoid leakage.
- [ ] Support both unigram and unigram-plus-bigram feature configurations and a vocabulary stress condition. Each model implementation will own its text preprocessing.
- [ ] Centralize split seeds, training-size fractions, stress-test settings, feature settings, and model parameters in recorded configuration. Split and experiment settings are in `ExperimentPlan`; model parameters still need the model implementations.
- [x] Save structured validation and final test results containing model, training fraction, repeat seed, split settings, and metrics so runs can be reproduced.
- [x] Generate Altair validation and test graphs for accuracy, macro precision, macro recall, macro F1, and confusion matrices. Save an all-model master report and a report for each model with completed runs.
- [x] Save each completed run's model, split settings, review ID and text, true label, and predicted label for error analysis.
- [ ] Provide a runnable entry point and README instructions, including environment/package versions and the commands needed to reproduce the experiments.

## Current repository state

### Present

- `src/ml_semester_project/data/train-00000-of-00001.parquet` and `test-00000-of-00001.parquet` are present. Both have 25,000 rows and columns `text` (string) and `label` (integer). The data appears to be a 25,000/25,000 review dataset; verify and record the dataset source and label mapping.
- `src/ml_semester_project/models/control.py` provides the majority-class baseline and owns its fit/predict calls.
- The logistic regression, Naive Bayes, and MLP model runners define the shared dataset-in/predictions-out interface but still need their model implementations.
- `src/ml_semester_project/experiment_runner.py` owns the experiment plan, stratified validation split, training subsets, model-runner calls, shared metrics, and result rows. Validation and test runs use the same sweep function. Each model module owns its preprocessing and fitting. The plan currently varies training size and repeat seed only.
- `main()` runs validation first, then evaluates models trained on the full training file against the separate test file. It writes result CSVs, per-review predictions, an all-model performance report, and model-specific reports under `src/ml_semester_project/output/`.
- Per-review validation and test predictions are also saved there as Parquet files, with review text and labels for error analysis.
- The current pipeline completed end to end for the implemented majority baseline. Runs for the other three models are recorded as `not_implemented` until teammates complete them.
- The project dependencies include pandas, PyArrow for Parquet support, Altair, scikit-learn, and NumPy.

### Missing or incomplete

- [ ] The three team-owned models still need implementations.
- [ ] Hyperparameter tuning is not implemented; the final test pass uses each model runner's configured defaults.
- [ ] Full model comparisons across all required models remain to be run after the team-owned models are implemented.
- [ ] The runner records each seed's metrics; mean and standard deviation aggregation and error analysis remain to be added.
- [ ] The README contains only the project title and team label; it does not document setup or reproduction steps.
- [ ] Record the dataset citation/source and label mapping, then confirm whether it satisfies the team's dataset coordination with Team 2.

## Suggested harness run sequence

1. Load and validate the dataset; establish reproducible train/validation/test splits.
2. Have each model fit its text vectorizer on its training partition for the unigram and unigram-plus-bigram conditions.
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
