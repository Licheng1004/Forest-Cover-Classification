# Forest Cover Type Classification

[![CI](https://github.com/Licheng1004/Forest-Cover-Classification/actions/workflows/ci.yml/badge.svg)](https://github.com/Licheng1004/Forest-Cover-Classification/actions/workflows/ci.yml)

Classify seven forest cover types using cartographic features and a heterogeneous stacking ensemble. This Machine Learning II course solution focuses on geographic feature engineering and complementary model behavior.

**Portfolio owner:** Licheng Xiong. **Stack:** Python, pandas, scikit-learn, XGBoost, Jupyter, GitHub Actions.

## Results and evidence

The [final report](report/final_report.md) records **0.83966 public leaderboard accuracy**, using **20 engineered geographic features (54 → 74)**. This is a historical report result, not a score recomputed by the public smoke test.

![Stacking architecture](report/report_figures/fig09_stacking_architecture.png)

- [Final stacking notebook](notebooks/final_stacking.ipynb)
- [Full report](report/final_report.md) · [PDF report](report/final_report.pdf)
- [Report figures](report/report_figures/)

## Project work

The repository publishes my final course implementation and its report, including the feature pipeline, stacking model, prediction export and reproducible repository packaging. The dataset/problem come from the course competition; I do not claim authorship of the competition or underlying ML libraries.

The base learners are Random Forest, Radius Neighbors, XGBoost and Extra Trees. An XGBoost meta-learner uses the base predictions and original features through passthrough. Geographic features encode terrain, illumination and distance relationships; the report explains the feature choices and class confusion.

## Reproduce the project

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then clone this repository. The committed `.python-version` selects Python 3.12.14; `pyproject.toml` and `uv.lock` lock the direct and transitive Python dependencies. First installation requires internet access.

```sh
git clone https://github.com/Licheng1004/Forest-Cover-Classification.git
cd Forest-Cover-Classification
uv sync --locked
uv run --locked python -m jupyterlab
```



### Verify the installation and published evidence

```sh
uv run --locked python scripts/verify.py
uv run --locked python scripts/ci_smoke.py
```

`--locked` rejects stale project metadata instead of silently changing the dependency versions. Select `.venv` as the Jupyter kernel in VS Code. Notebook path detection supports execution from the repository root or a notebook subdirectory. Runtime installation and synthetic checks do not reproduce a private-data competition score.

See [the reproducibility record](REPRODUCIBILITY.md) for the actual checks performed, their scope and any external prerequisites.

## Data and complete model run

Obtain the authorized challenge files using the [Kaggle Forest Cover source](https://www.kaggle.com/c/forest-cover-type-prediction), subject to its access and competition rules. The report references the DSAIB 2025–2026 course competition; exact course leaderboard reproduction requires that course's matching inputs and evaluation service.

```text
data/local/train.csv
data/local/test-full.csv   # rename the corresponding test.csv
```

Run `notebooks/final_stacking.ipynb` from the locked environment. It produces `outputs/submission_stacking_4models.csv`. The public repository includes code, reports and figures; data and generated submissions are ignored. Stacking cross-validation and multiple ensemble learners can require substantial CPU time and RAM.

## Repository map

| Path | Purpose |
| --- | --- |
| `notebooks/final_stacking.ipynb` | Final features, learners, stacking and export |
| `report/` | Methodology, final report and figures |
| `scripts/` | Notebook verification and synthetic XGBoost training |
| `data/local/`, `outputs/` | Local inputs and generated outputs |

## Limitations and reuse

The leaderboard score is tied to its original evaluation. The synthetic training check validates installation, not the ensemble's accuracy. Provider rules govern the data; this README grants no new reuse license.
