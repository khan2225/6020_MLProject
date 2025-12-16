# `models/` module

This folder contains everything needed to train, select, and interpret the
regression models used to approximate the legacy travel-reimbursement engine.

Contents:

- `pipelines.yaml` – configuration for all scikit-learn pipelines
- `predict.py` – trains **all** configured pipelines and saves the best one
- `xgb_model.py` – trains an XGBoost benchmark model and saves it
- `shap_analysis.py` – helper code for SHAP-based interpretability
- `final_model.pkl` – saved scikit-learn Gradient Boosting pipeline
- `final_model_xgb.pkl` – saved XGBoost model

---

## `pipelines.yaml`: model configuration

`pipelines.yaml` defines every scikit-learn pipeline used in the project.
Each top-level key under `pipelines:` is a short model name (e.g. `ols`,
`rf_12`, `gbr_shallow`) with the fields:

---

### `predict.py`: train all pipelines and save the best model

`predict.py` is the main entry point for running **all** scikit-learn
pipelines defined in `pipelines.yaml`, comparing their performance, and
saving the best performer.

1. Loads the project data via `ModelConfig`.
2. Builds every pipeline in `pipelines.yaml` using `BuildModels`.
3. Trains each pipeline and evaluates it on the held-out test set.
4. Prints a metrics summary for each model (one line per pipeline) of the form:

       ols             MAE= 167.014  RMSE= 208.790  MedAE= 153.310  R2= 0.781  P90= 302.117  MaxE=1058.583  MAPE= 14.98%
       rr_1            MAE= 166.982  RMSE= 208.724  MedAE= 153.870  R2= 0.781  P90= 301.846  MaxE=1058.397  MAPE= 14.99%
       ...
       gbr_shallow     MAE=  68.449  RMSE= 106.763  MedAE=  52.474  R2= 0.943  P90= 135.225  MaxE= 978.776  MAPE=  6.35%
       ...
       Best: gbr_shallow
       Saved final model 'gbr_shallow' to final_model.pkl

5. Picks the model with the lowest MAE as the “best” model
   (currently `gbr_shallow`).
6. Retrains that best pipeline on all available data and saves it to
   `models/final_model.pkl`.

#### How to run

From the **project root** (the folder that contains the `models/` directory):

    python -m models.predict

No command-line arguments are required. The script prints the metrics
block to the console and overwrites `models/final_model.pkl` with the
latest trained best model.

---

### `xgb_model.py`: XGBoost benchmark and pickle

`xgb_model.py` trains a separate XGBoost regressor using the three original
inputs (trip duration, miles traveled, total receipts). It is not controlled
by `pipelines.yaml`; its hyperparameters are specified directly in the code.

1. Loads the same data as `predict.py` via `ModelConfig`.
2. Fits an `xgboost.XGBRegressor` with a fixed, sensible configuration.
3. Evaluates it on the test set and prints a metrics line (MAE, RMSE, R²,
   etc.), comparable to the scikit-learn models.
4. Retrains XGBoost on all available data and saves it to

   - `models/final_model_xgb.pkl`

#### How to run

    python -m models.xgb_model

There are **no command-line arguments**. The script will overwrite
`models/final_model_xgb.pkl` if it already exists.

---

### `shap_analysis.py`: SHAP utilities

`shap_analysis.py` contains helper classes/functions to compute SHAP values
for the XGBoost model and generate interpretability plots.

Typical usage is via a separate script (e.g.
`scripts/generate_xgb_shap_plots.py`) that:

1. Loads `final_model_xgb.pkl` (or trains a fresh XGBoost model).
2. Loads the feature matrix used for training/testing.
3. Uses the utilities in `shap_analysis.py` to compute SHAP values and
   save figures such as:

   - `reports/figures/xgb_shap_summary.png`
   - `reports/figures/xgb_shap_dependence_input.total_receipts_amount.png`
   - `reports/figures/xgb_shap_bar.png` (mean |SHAP| bar plot)

You generally do **not** need to run `shap_analysis.py` directly; instead,
import its classes into a plotting script, or use the existing
`generate_xgb_shap_plots.py` script in `scripts/`.

---

### Regenerating the pickle files

To fully recreate both saved models from scratch:

**Delete old pickles (optional but clean)**

       rm models/final_model.pkl models/final_model_xgb.pkl

**Train and save the scikit-learn final model**

       python -m models.predict

   This prints metrics for all pipelines and saves the best one
   (currently `gbr_shallow`) to:

   - `models/final_model.pkl`

**Train and save the XGBoost benchmark**

       python -m models.xgb_model

   This trains the XGBoost surrogate and saves it to:

   - `models/final_model_xgb.pkl`

**At this point:**

- `final_model.pkl` – the chosen Gradient Boosting Regressor used by
  the production prediction script.
- `final_model_xgb.pkl` – the XGBoost surrogate used primarily for
  SHAP-based interpretability and benchmarking.

Both pickles can be loaded later with `joblib.load` or `pickle.load` in
any script that wants to make predictions or analyze the models.
