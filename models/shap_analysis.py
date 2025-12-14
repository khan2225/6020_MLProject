from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
import shap

from predict import ModelConfig, BuildModels


@dataclass
class ShapAnalyzer:
    """Wrapper for computing / plotting SHAP values
    (tree-based model)."""

    model_name: str
    data_path: str = "../data/public_cases.json"

    def __post_init__(self):
        self.config = ModelConfig(self.data_path)
        self.builder = BuildModels(self.config)

        self.model = self.builder.pipelines[self.model_name]

        self.X_train = self.config.X_train
        self.y_train = self.config.y_train
        self.X_test = self.config.X_test
        self.y_test = self.config.y_test

        self._fitted = False
        self.estimator_: Optional[object] = None
        self.explainer_: Optional[shap.TreeExplainer] = None
        self.shap_values_: Optional[np.ndarray] = None
        self.mean_abs_shap_: Optional[pd.DataFrame] = None

    def fit(self):
        if not self._fitted:
            self.model.fit(self.X_train, self.y_train)
            self._fitted = True

    def compute_shap(self):
        """Fit model (if needed) and compute SHAP values on X_test."""
        self.fit()
        # Use final estimator for pipelines; otherwise use model itself
        if hasattr(self.model, "named_steps"):
            self.estimator_ = list(self.model.named_steps.values())[-1]
        else:
            self.estimator_ = self.model
        self.explainer_ = shap.TreeExplainer(self.estimator_)
        self.shap_values_ = self.explainer_.shap_values(self.X_test)
        self.mean_abs_shap_ = (
            pd.DataFrame(
                {
                    "feature": self.X_test.columns,
                    "mean_abs_shap": np.abs(self.shap_values_).mean(axis=0),
                }
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        return self.mean_abs_shap_

    def plot_summary(self, bar: bool = False, show=True):
        """SHAP summary plot; bar=True for global importance bar chart."""
        if self.shap_values_ is None:
            self.compute_shap()

        plot_type = "bar" if bar else "dot"
        shap.summary_plot(
            self.shap_values_,
            self.X_test,
            plot_type=plot_type,
            show=show,
        )

    def plot_dependence(self, feature: str, show=True):
        """Dependence plot for a single feature."""
        if self.shap_values_ is None:
            self.compute_shap()
        shap.dependence_plot(
            feature,
            self.shap_values_,
            self.X_test,
            show=show,
        )

# Convenience constructor for your final model
def create_gbr_shallow_analyzer(
    data_path: str = "../data/public_cases.json",
) -> ShapAnalyzer:
    return ShapAnalyzer(model_name="gbr_shallow", data_path=data_path)
