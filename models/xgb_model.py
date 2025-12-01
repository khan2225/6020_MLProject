import joblib
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

from predict import ModelConfig


class XGBModelRunner:
    def __init__(self, data_path: str = "../data/public_cases.json"):
        """
        Initialize w/ existing ModelConfig for data loading / splitting.
        """
        self.config = ModelConfig(data_path)
        self.best_model_ = None
        self.best_params_ = None
        self.best_cv_mae_ = None

    @staticmethod
    def default_model(random_state: int = 42) -> XGBRegressor:
        """
        Default XGBRegressor configuration for small tabular data
        """
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
        )

    def evaluate_default(self):
        """
        Fit / evaluate default XGBRegressor w/ ModelConfig.evaluate_model.
        """
        model = self.default_model()
        result = self.config.evaluate_model(model, _name="xgb_default")
        return result

    def tune_with_grid_search(self):
        """
        Hyperparameter tuning for XGBRegressor using GridSearchCV (MAE = score).
        Stores the best model and params on the instance.
        """
        base = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )

        param_grid = {
            "n_estimators": [300, 600],
            "learning_rate": [0.03, 0.05],
            "max_depth": [3, 4],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }

        grid = GridSearchCV(
            estimator=base,
            param_grid=param_grid,
            scoring="neg_mean_absolute_error",
            cv=5,
            n_jobs=-1,
            verbose=1,
        )

        grid.fit(self.config.X_train, self.config.y_train)

        self.best_model_ = grid.best_estimator_
        self.best_params_ = grid.best_params_
        self.best_cv_mae_ = -grid.best_score_

        print("\n[XGB] Best CV MAE:", self.best_cv_mae_)
        print("[XGB] Best params:", self.best_params_)

        # Evaluate tuned model on the held-out test set
        result = self.config.evaluate_model(self.best_model_, _name="xgb_tuned")
        return result

    def save_best(self, path: str = "final_model_xgb.pkl"):
        """
        Save best XGBoost model found (after tune_with_grid_search).
        """
        if self.best_model_ is None:
            raise ValueError("No best_model_ set. "
                             "Run tune_with_grid_search() first.")
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.best_model_, model_path)
        print(f"Saved XGBoost model to {model_path}")

    def train_full_and_save(self, path: str = "final_model_xgb_full.pkl"):
        """
        Retrain the tuned model on ALL data (X, y) and save.
        """
        if self.best_params_ is None:
            raise ValueError("No best_params_. "
                             "Run tune_with_grid_search() first.")
        full_model = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            **self.best_params_,
        )
        full_model.fit(self.config.X, self.config.y)
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(full_model, model_path)
        print(f"Saved full-data XGBoost model to {model_path}")


if __name__ == "__main__":
    runner = XGBModelRunner("../data/public_cases.json")

    print("\n * * * Default XGBoost * * * ")
    default_result = runner.evaluate_default()

    print("\n * * * Tuned XGBoost (GridSearchCV) * * * ")
    tuned_result = runner.tune_with_grid_search()

    # Save the tuned test-set model
    runner.save_best("final_model_xgb.pkl")

    # Retrain tuned params on full data and save separately
    # runner.train_full_and_save("models/final_model_xgb_full.pkl")
