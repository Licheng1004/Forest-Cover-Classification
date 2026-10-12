[![CI/CD](https://github.com/Licheng1004/Forest-Cover-Classification/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Licheng1004/Forest-Cover-Classification/actions/workflows/ci.yml)

# Forest Cover Type Classification

Final Machine Learning II course project. The repository contains the final four-base-learner Stacking model, its report, and report figures; the intermediate model iterations are omitted.

## Final model

The report describes 20 geographic feature additions (54 to 74 features) and a Stacking ensemble with Random Forest, Radius Neighbors, XGBoost, and Extra Trees as base learners, plus an XGBoost meta-learner with passthrough. The report records a Kaggle public leaderboard accuracy of **0.83966**.

## Data

Competition data are not included. Download them from the [Kaggle Forest Cover Type Prediction competition](https://www.kaggle.com/c/forest-cover-type-prediction) after signing in and accepting its rules. Place the files under `data/local/` with these names:

- `train.csv`
- `test-full.csv` (rename Kaggle's `test.csv` to this name)

The notebook writes its generated prediction file to `outputs/`, which is ignored by Git.

## Run

```bash
python -m pip install -r requirements.txt
jupyter lab
```

Start Jupyter from the repository root, open `notebooks/final_stacking.ipynb`, and run its cells in order.

## Report

- [Final report (Markdown)](report/final_report.md)
- [Final report (PDF)](report/final_report.pdf)

## CI/CD

The GitHub Actions workflow runs on pushes to `main`, pull requests, and manual dispatch. It validates the final notebook and fits a tiny XGBoost model on synthetic data; it does not need the Kaggle dataset. On a `v*` tag, it also publishes a GitHub Release with the source snapshot and the final report deliverables.
