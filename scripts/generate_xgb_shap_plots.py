from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import shap
from xgboost import XGBRegressor
from models.predict import ModelConfig


def get_paths():
    """
    Resolve project layout relative to this file.
    Assumes:
      project_root/
        data/public_cases.json
        models/
        reports/figures/
    """
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "public_cases.json"
    fig_dir = base_dir / "reports" / "figures"
    model_dir = base_dir / "models"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return base_dir, data_path, fig_dir, model_dir

def load_or_train_xgb(config: ModelConfig, model_dir: Path):
    """
    Load a previously saved XGBoost model if present,
    or train a single reasonably tuned default model
    """
    model_path = model_dir / "final_model_xgb.pkl"
    if model_path.exists():
        print(f"Loading XGBoost Model from {model_path}")
        model = joblib.load(model_path)
    else:
        print("Training Default XGBoost Model (no grid search)...")
        model = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
            tree_method="hist",
        )
        model.fit(config.X_train, config.y_train)
        joblib.dump(model, model_path)
        print(f"Saved XGBoost model to {model_path}")
    return model


def plot_xgb_feature_importance(model, feature_names, out_path: Path):
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    imp_sorted = importances[order]
    names_sorted = np.array(feature_names)[order]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names_sorted, imp_sorted)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance")
    ax.set_title("XGBoost Feature Importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")

def plot_xgb_pred_vs_actual(model, X_test, y_test, out_path: Path):
    y_pred = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.6)
    lo = min(y_test.min(), y_pred.min())
    hi = max(y_test.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "--")
    ax.set_xlabel("Actual Reimbursement")
    ax.set_ylabel("Predicted Reimbursement")
    ax.set_title("XGBoost: Predicted vs. Actual")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_xgb_residual_hist(model, X_test, y_test, out_path: Path):
    y_pred = model.predict(X_test)
    residuals = y_pred - y_test
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=30)
    ax.set_xlabel("Residual (Prediction - Actual)")
    ax.set_ylabel("Count")
    ax.set_title("XGBoost Residual Distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")

def shap_tree_model(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names,
    fig_dir: Path,
    prefix: str,
):
    """
    SHAP helper for XGBRegressor
    """
    if isinstance(X_train, pd.DataFrame):
        background = X_train.sample(
            n=min(200, len(X_train)), random_state=42)
    else:
        n_bg = min(200, X_train.shape[0])
        idx = np.random.RandomState(42).choice(
            X_train.shape[0], size=n_bg, replace=False)
        background = X_train[idx]
    explainer = shap.TreeExplainer(model, data=background)
    shap_values = explainer.shap_values(X_test)

    # SHAP summary (beeswarm)
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    out_path = fig_dir / f"{prefix}_shap_summary.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

    # SHAP bar (global importance)
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    out_path = fig_dir / f"{prefix}_shap_bar.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

    # SHAP dependence for most important feature
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_idx = int(np.argmax(mean_abs))
    top_feat = feature_names[top_idx]

    shap.dependence_plot(
        top_feat,
        shap_values,
        X_test,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    out_path = fig_dir / f"{prefix}_shap_dependence_{top_feat}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

def plot_shap_summary(model, X_train: pd.DataFrame,
                      X_test: pd.DataFrame, fig_dir: Path):
    """Global SHAP summary (beeswarm) for XGB model"""
    print("Computing SHAP values for summary plot...")
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    shap.summary_plot(
        shap_values.values,
        X_test,
        feature_names=X_test.columns,
        show=False,
    )
    fig = plt.gcf()
    fig.set_size_inches(10, 3)
    out_path = fig_dir / "xgb_shap_summary.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

def plot_shap_dependence(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    fig_dir: Path,
    main_feature: str = "total_receipts_amount",
    color_feature: str = "miles_traveled",
):
    """Dependence plot: SHAP vs a single feature"""
    if main_feature not in X_test.columns:
        raise ValueError(f"{main_feature} not in X_test columns "
                         f"{list(X_test.columns)}")
    if color_feature not in X_test.columns:
        raise ValueError(f"{color_feature} not in X_test columns "
                         f"{list(X_test.columns)}")
    print("Computing SHAP values for dependence plot...")
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    shap.dependence_plot(
        main_feature,
        shap_values.values,
        X_test,
        interaction_index=color_feature,
        show=False,
    )
    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    out_path = fig_dir / f"xgb_shap_dependence_{main_feature}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

def main():
    base_dir, data_path, fig_dir, model_dir = get_paths()
    # Load data & split using your existing config logic
    config = ModelConfig(str(data_path))
    # Load or train XGBoost model
    xgb_model = load_or_train_xgb(config, model_dir)
    # Basic XGB plots
    feature_names = list(config.X_train.columns)
    plot_xgb_feature_importance(
        xgb_model,
        feature_names,
        fig_dir / "xgb_feature_importance.png",
    )
    plot_xgb_pred_vs_actual(
        xgb_model,
        config.X_test,
        config.y_test,
        fig_dir / "xgb_pred_vs_actual.png",
    )
    plot_xgb_residual_hist(
        xgb_model,
        config.X_test,
        config.y_test,
        fig_dir / "xgb_residual_hist.png",
    )
    # SHAP plots for XGB
    shap_tree_model(
        xgb_model,
        config.X_train,
        config.X_test,
        feature_names=feature_names,
        fig_dir=fig_dir,
        prefix="xgb",
    )
    # SHAP on the input features ModelConfig uses
    X_train = (
        config.X_train
        if isinstance(config.X_train, pd.DataFrame)
        else pd.DataFrame(config.X_train, columns=config.feature_names)
    )
    X_test = (
        config.X_test
        if isinstance(config.X_test, pd.DataFrame)
        else pd.DataFrame(config.X_test, columns=config.feature_names)
    )
    plot_shap_summary(xgb_model, X_train, X_test, fig_dir)
    plot_shap_dependence(
        xgb_model, X_train, X_test, fig_dir,
        main_feature="input.total_receipts_amount",
        color_feature="input.miles_traveled",
    )


if __name__ == "__main__":
    main()
