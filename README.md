# Tennis Match Prediction — XGBoost

A small machine learning project for predicting tennis match outcomes using historical ATP-style match data. The pipeline loads match records, builds leak-safe features from prior results, trains an XGBoost classifier, evaluates it on a 2024 holdout set, and supports manual pre-match prediction.

This project is implemented across [training.ipynb](training.ipynb), [infer.py](infer.py), and [model.py](model.py).

## Project Goal

Predict which player is more likely to win a tennis match using:

- rank difference
- recent form difference
- surface win-rate difference
- Elo difference
- match format and tournament context

The model is trained on historical data from [datasets/matches_final.csv](datasets/matches_final.csv) and saved after training.

## Repository Structure

- [training.ipynb](training.ipynb) — complete data cleaning, feature engineering, model training, evaluation, and manual prediction examples
- [infer.py](infer.py) — reusable CLI-style inference script for entering new match data and predicting outcomes
- [model.py](model.py) — quick inspection script that loads the trained model bundle
- [datasets/matches_final.csv](datasets/matches_final.csv) — main dataset used for training
- [.gitignore](.gitignore) — excludes generated model/data files from version control

## Generated Files

The following files are created after running [training.ipynb](training.ipynb):

- [datasets/features.csv](datasets/features.csv) — engineered feature table
- [datasets/model.pkl](datasets/model.pkl) — trained model and feature columns saved with joblib

These are intentionally listed in [.gitignore](.gitignore) so they are not committed to source control.

## Requirements

Python 3.10+ is recommended.

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install pandas numpy joblib scikit-learn xgboost jupyter
```

## Quick Start

### 1) Train the model

Open and run all cells in [training.ipynb](training.ipynb).

This notebook does the following:

- loads the dataset
- cleans rank values and dates
- computes historical features in chronological order
- creates train/test splits by date
- trains an XGBoost classifier
- evaluates accuracy and probability metrics
- saves the trained model as [datasets/model.pkl](datasets/model.pkl)

### 2) Inspect the saved model

After training, run:

```bash
python model.py
```

This loads the bundle and prints:

- the saved model object
- the feature column names

### 3) Predict a match manually

Run:

```bash
python infer.py
```

Then enter:

- player names
- surface
- series
- round
- best of
- Elo
- ATP rank
- recent form
- surface win rate

The script prints each player’s win probability and the predicted winner.

## Model Details

The training notebook builds the following feature groups:

- numeric differences:
  - Rank_Diff
  - Recent_Form_Diff
  - Surface_WR_Diff
  - Elo_Diff
  - Best of
- categorical encodings:
  - Surface
  - Series
  - Round

The model used is:

- XGBClassifier
- trained with:
  - n_estimators=300
  - max_depth=4
  - learning_rate=0.05
  - subsample=0.8
  - colsample_bytree=0.8
  - eval_metric="logloss"

## Evaluation

The model is evaluated on 2024 matches using:

- accuracy
- log loss
- Brier score

The notebook also compares the model to a simple Elo-based baseline.

## Manual Prediction Example

The notebook includes examples like:

```python
predict_manual(
    "Zverev A.", "Shelton B.",
    surface="Hard", series="ATP Finals", round_="Round Robin", best_of=3,
    elo_A=2107, elo_B=2040,
    rank_A=1,   rank_B=4,
    form_A=0.79, form_B=0.68,
    wr_A=0.75,   wr_B=0.67,
)
```

This generates a probability estimate for each player and predicts the likely winner.

## Notes on Data Leakage

The feature engineering is intentionally leak-safe:

- recent form is computed using only prior matches
- surface win rate uses previous matches on the same surface
- Elo is updated sequentially from prior outcomes

This ensures the model is trained on information available before the match, rather than using future results.

## Troubleshooting

### Model file missing

If you run [infer.py](infer.py) before training, it will exit with an error explaining that [datasets/model.pkl](datasets/model.pkl) is missing.

Run [training.ipynb](training.ipynb) first.

### Import errors

Make sure your virtual environment is activated and all dependencies are installed:

```bash
pip install pandas numpy joblib scikit-learn xgboost jupyter
```

### Notebook not recognized

If Jupyter is not installed or the notebook does not open, run:

```bash
jupyter notebook
```

and open [training.ipynb](training.ipynb) from the interface.

## Suggested Next Improvements

- add a command-line wrapper for batch predictions
- compare several models (XGBoost, RandomForest, Logistic Regression)
- add feature importance plots
- include a saved metrics report
- support live or scraped player statistics

## License

This project does not include a license file. Add one if you plan to share or publish the repository publicly.

## Summary

This repository demonstrates a full tennis prediction workflow:

1. clean historical data
2. build prior-match features
3. train an XGBoost classifier
4. evaluate on recent matches
5. predict new matches using raw pre-game stats

The main workflow is defined in [training.ipynb](training.ipynb), while [infer.py](infer.py) provides a practical inference interface.
