import joblib
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from pathlib import Path
from predict import ModelConfig


class XGBModelRunner:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.best_model_ = None
        self.best_params_ = None
        self.best_cv_mae_ = None

    def default_model(self):
        """Train a default XGBoost model and return metrics dict"""
        model = xgb.XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
        )
        metrics = self.config.evaluate_model(model, _name="xgb_default")
        return metrics

    def tune_with_grid_search(self):
        """
        Run GridSearchCV for XGBoost, store best model, and
        return (metrics_dict, best_params, best_cv_mae)
        """
        base_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
        )
        param_grid = {
            "n_estimators": [300, 600],
            "learning_rate": [0.03, 0.05],
            "max_depth": [3, 4],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }
        grid = GridSearchCV(
            estimator=base_model,
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

        print(f"[XGB] Best CV MAE: {self.best_cv_mae_:.3f}")
        print(f"[XGB] Best params: {self.best_params_}")

        # Evaluate on held-out test set using helper
        metrics = self.config.evaluate_model(
            self.best_model_, _name="xgb_tuned")
        out_path = Path("models") / "final_model_xgb.pkl"
        joblib.dump(self.best_model_, out_path)
        print(f"Saved XGBoost model to {out_path}")

        return metrics, self.best_params_, self.best_cv_mae_

    def save_best(self, path: str | Path =
    "models/final_model_xgb.pkl") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.best_model_, path)
        print(f"Saved tuned XGBoost model to {path}")
        return path



if __name__ == "__main__":
    config = ModelConfig(r"../data/public_cases.json")
    runner = XGBModelRunner(config)

    print("\n * * * Default XGBoost * * * ")
    default_result = runner.default_model()

    # print("\n * * * Tuned XGBoost (GridSearchCV) * * * ")
    # tuned_result = runner.tune_with_grid_search()
    runner.save_best("final_model_xgb.pkl")