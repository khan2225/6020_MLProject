import json
import yaml
import joblib
from pathlib import Path
from pprint import pprint
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


class ModelConfig(object):
    def __init__(self, data):
        self.data = data
        self.file = self.load_file(self.data)
        self.df = self.json_to_df(self.file)

        self.X, self.y = self.organize()
        (self.X_train,self.X_test,self.y_train,self.y_test) = \
        self.split_into_subsets(self.X, self.y)

    @staticmethod
    def split_into_subsets(X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=test_size,
                             random_state=random_state)
        return X_train, X_test, y_train, y_test

    def evaluate_model(self, _model, _name="model"):
        """Fit on training data and evaluate on validation data."""
        _model.fit(self.X_train, self.y_train)
        preds = _model.predict(self.X_test)
        abs_err = np.abs(preds - self.y_test)
        # MSE
        mae = mean_absolute_error(self.y_test, preds)
        mse = mean_squared_error(self.y_test, preds)  # returns MSE
        rmse = np.sqrt(mse)  # convert to RMSE
        medae = median_absolute_error(self.y_test, preds)
        r2 = r2_score(self.y_test, preds)

        max_err = abs_err.max()
        p90_err = np.percentile(abs_err, 90)
        p95_err = np.percentile(abs_err, 95)

        denom = np.maximum(np.abs(self.y_test), 1e-8)
        mape = np.mean(abs_err / denom) * 100.0

        print(
            f"{_name:15s} "
            f"MAE={mae:8.3f}  "
            f"RMSE={rmse:8.3f}  "
            f"MedAE={medae:8.3f}  "
            f"R2={r2:6.3f}  "
            f"P90={p90_err:8.3f}  "
            f"MaxE={max_err:8.3f}  "
            f"MAPE={mape:6.2f}%"
        )

        return {
            "name": _name,
            "model": _model,
            "mae": mae,
            "rmse": rmse,
            "medae": medae,
            "r2": r2,
            "p90_err": p90_err,
            "p95_err": p95_err,
            "max_err": max_err,
            "mape": mape,
        }

    def organize(self):
        input_fields = [field for field in self.df.columns
                        if field.startswith("input.")]
        X = self.df[input_fields]       # feature_matrix
        y = self.df["expected_output"]  # target_vector
        return X, y

    @staticmethod
    def json_to_df(file):
        return pd.json_normalize(file)

    @staticmethod
    def load_file(data):
        with open(data, 'r') as fpath:
            file = json.load(fpath)
        return file


class BuildModels:
    linear_models = ["ols", "rr_1", "rr_10", "rr_0.1", "lasso", "enet"]
    tree_models = ["rf_6", "rf_12", "gbr", "dt_5", "dt_10"]


    def __init__(self, parent):
        self.parent = parent
        self.pldict = self.read_yaml()
        self.pipelines = self.create_pipeline(self.pldict)
        self.feat_names = self.parent.X_train.columns

    @staticmethod
    def read_yaml():
        # Resolve pipelines.yaml relative to this file
        here = Path(__file__).resolve().parent
        yaml_path = here / "pipelines.yaml"
        with open(yaml_path, "r") as file:
            pldict = yaml.safe_load(file)
        return pldict

    @staticmethod
    def create_pipeline(pldict):
        plobj = dict()
        for i in pldict:
            pld = pldict[i]
            pl_steps = list()
            for j in pld:
                params = pld[j]
                mod, submod = j.split(".")
                module = getattr(__import__(
                    f"sklearn.{mod}", fromlist=[submod]), submod)
                step = (str(submod), module(**params)) \
                    if params else (str(submod), module())
                pl_steps.append(step)
            plobj[i] = Pipeline(pl_steps)
        pprint(plobj)
        return plobj

    def summarize_linear(self, pipe, feature_names, top_k=10):
        pipe.fit(self.parent.X_train, self.parent.y_train)
        _model = list(pipe.named_steps.values())[-1]
        df = (pd.DataFrame({"feature": feature_names, "coef": model.coef_})
              .assign(abs_coef=lambda d: d["coef"].abs())
              .sort_values("abs_coef", ascending=False)
              .drop(columns="abs_coef")
              .head(top_k))
        return df

    def test_linear(self):
        for model in self.linear_models:
            print(f"\n=== {model} ===")
            df = self.summarize_linear(self.pipelines[model],
                                       self.feat_names, top_k=10)
            print(df)

    def summarize_tree(self, pipe, feature_names, top_k=15):
        pipe.fit(self.parent.X_train, self.parent.y_train)
        model = list(pipe.named_steps.values())[-1]
        imps = model.feature_importances_
        df = (pd.DataFrame({"feature": feature_names, "importance": imps})
              .sort_values("importance", ascending=False)
              .head(top_k))
        return df

class FinalModelTrainer:
    def __init__(
        self,
        config: ModelConfig,
        builder: BuildModels,
        final_model_name: str,
        model_path: str = "final_model.pkl",
    ):
        self.config = config
        self.builder = builder
        self.final_model_name = final_model_name
        self.model_path = Path(model_path)
        self.trained_pipeline = None

    def train_on_all_data(self):
        pipe = self.builder.pipelines[self.final_model_name]
        pipe.fit(self.config.X, self.config.y)
        self.trained_pipeline = pipe
        return pipe

    def save(self):
        if self.trained_pipeline is None:
            self.train_on_all_data()
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.trained_pipeline, self.model_path)
        print(f"Saved final model '{self.final_model_name}' "
              f"to {self.model_path}")

    def train_and_save(self):
        self.train_on_all_data()
        self.save()


if __name__ == '__main__':
    config = ModelConfig("../data/public_cases.json")
    build = BuildModels(config)
    results = list()
    for name, model in build.pipelines.items():
        res = config.evaluate_model(model, _name=name)
        results.append(res)

    best_name = min(results, key=lambda r: r["mae"])["name"]
    print("\nBest:", best_name)

    trainer = FinalModelTrainer(config, build, final_model_name=best_name)
    trainer.train_and_save()
