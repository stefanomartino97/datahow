# DataHow Coding Challenge

Predict the final monoclonal-antibody (mAb) titer of a simulated upstream bioprocess from its multivariate time-series. The dataset contains 100 experiments, each with daily measurements of process inputs (`W:`), state variables (`X:`), and a set of constant design parameters (`Z:`); the label is one scalar per experiment (`Y:Titer`).

## Project layout

```
datahow/
├── data/            # train/test CSVs, target CSVs (not tracked on GitHub)
├── notebooks/       # exploration, plots, modeling, endpoint smoke-test
├── src/             # library code
│   ├── api.py       # FastAPI app
│   ├── routes/      # /health, /predict
│   ├── config.py    # pydantic-settings
│   ├── data.py      # loading + preprocessing
│   ├── models.py    # candidate pipelines + grids
│   ├── train.py     # nested-CV training driver
│   ├── inference.py # load best MLflow model + predict
│   ├── validation.py
│   └── utils.py
├── tests/           # pytest suite (api, validation, utils)
├── Dockerfile
└── pyproject.toml
```

## Approach

The four notebooks document the path from raw CSVs to the chosen model - each one's findings shaped the next step.

I started in **`01_exploration.ipynb`** by auditing the dataset against the schema described in the challenge: `Z:` design constants, `W:` daily inputs, `X:` daily measurements, and a single `Y:Titer` scalar per experiment. The structural invariants all held - `Z:` is recorded only on day 0, `W:pH` / `W:temp` actually change on the scheduled `Z:phShift` / `Z:tempShift` days, `Y:Titer` is reported exactly once per experiment on its final day, and every other column is fully populated. With the data confirmed clean and well-formed, no cleaning step was needed and I could move straight to feature design.

In **`02_plots.ipynb`** I plotted per-experiment trajectories and target distributions to see what kind of regression problem this actually was. The decisive finding was the shape of the target: raw `Y:Titer` is heavily right-skewed (skew **+1.88**, excess kurtosis **+5.05**), while `log(Y:Titer)` is essentially Gaussian (skew **+0.05**, kurtosis **+0.43**). That fixed the modeling target - **train on `log(Y:Titer)` and exponentiate at prediction time**, which is what `src/data.py` and `src/inference.py` now do. A univariate correlation sweep against `log(Y:Titer)` also flagged `X:VCD_mean` (mean viable cell density across the run) as the single strongest predictor, which made physical sense (more producer cells integrated over the run → more product) and gave me a sanity baseline to beat.

That led into **`03_modeling.ipynb`**, where the model choice had to respect the shape of the problem: 100 sequence-to-scalar samples. That ruled out univariate forecasters (Prophet/ARIMA - wrong problem shape) and deep sequence models (LSTM/Transformer - too data-hungry), and pointed to classical regression on a per-experiment feature matrix (Z setpoints + W sum/mean + X final/max/mean/AUC + growth-phase rates). I benchmarked Ridge, PLS, SVR, Random Forest, XGBoost, and LightGBM under repeated nested 5-fold CV. **Ridge on the full feature matrix won at RMSE 0.149 ± 0.035, R² 0.904**, with PLS landing in the same place (within sampling noise) and clean residual diagnostics (predicted-vs-actual on the diagonal, residuals homoscedastic). Two ablations clarified where the signal lives: `Z`-only Ridge gets R² 0.25, `X`-only gets 0.88 - the observed trajectory carries almost all the predictive signal, the recipe alone barely moves the needle. That's the model registered in `src/models.py` and trained by `src/train.py`.

## Running the training

The project uses [`uv`](https://docs.astral.sh/uv/) for environment + dependency management.

```bash
# install dependencies
uv sync

# (optional) browse runs in MLflow
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

# train every candidate, log to MLflow, tag the CV-best run
uv run python -m src.train train_all

# refit the CV-best on train+test and log it as the production run
uv run python -m src.train retrain_best
```

`src/train.py` exposes two modes. `train_all` runs repeated nested 5-fold CV for each candidate in `src.models.model_candidates`, fits the final pipeline on the training split, and logs hyperparameters, metrics (`rmse_mean`, `rmse_std`, `r2_mean`, `r2_std`), the fitted sklearn pipeline, and the feature-column manifest to MLflow; the lowest-RMSE run is tagged `stage=cv_best`. `retrain_best` loads that CV-best pipeline, refits it on the concatenation of the train and test splits, and logs the result as a new run tagged `stage=production` so `src/inference.py` can resolve it at serving time.

## AI use

AI assistance was used in this project to improve markdown formatting in the notebooks and to format docstrings consistently across the `src/` modules.
