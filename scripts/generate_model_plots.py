from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from models.predict import ModelConfig, BuildModels

MODEL_FAMILIES: dict[str, list[str]] = {
    "Linear / Regularized": ["ols", "rr_1", "rr_10", "rr_0.1", "lasso", "enet"],
    "Polynomial Ridge":     ["prr2", "prr3"],
    "Random Forest":        ["rf_6", "rf_12"],
    "Gradient Boosting":    ["gbr_base", "gbr_slow", "gbr_shallow", "gbr_stochastic"],
    "Decision Tree":        ["dt_5", "dt_10", "dt_25", "dt_50"],
    "XGBoost":              ["xgb_default", "xgb_tuned"],
}

def get_paths():
    """
    Resolve project layout relative to this file
    """
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "public_cases.json"
    fig_dir = base_dir / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return base_dir, data_path, fig_dir

def collect_metrics(data_path: Path, fig_dir: Path):
    """
    Re-run all YAML-defined models and collect performance metrics
    """
    config = ModelConfig(str(data_path))
    builder = BuildModels(config)

    results = []
    for name, pipe in builder.pipelines.items():
        res = config.evaluate_model(pipe, _name=name)
        results.append(res)

    df = pd.DataFrame(results).drop(columns=["model"])
    df = df.rename(
        columns={
            "name": "model",
            "mae": "MAE",
            "rmse": "RMSE",
            "medae": "MedAE",
            "r2": "R2",
            "p90_err": "P90",
            "p95_err": "P95",
            "max_err": "MaxE",
            "mape": "MAPE",
        }
    )

    out_csv = fig_dir / "model_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved metrics table to {out_csv}")

    return config, builder, df


def plot_mae_bar(df: pd.DataFrame, fig_dir: Path,
                 final_model: str = "gbr_shallow"):
    """
    Horizontal bar chart of MAE by model
    """
    df_plot = df.sort_values("MAE", ascending=True).copy()

    colors = []
    for name in df_plot["model"]:
        if name == final_model:
            colors.append("tab:blue")
        else:
            colors.append("lightgray")

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(df_plot["model"],
                   df_plot["MAE"],
                   color=colors)

    ax.set_xlabel("Mean Absolute Error (MAE)")
    ax.set_ylabel("Model")
    ax.set_title("Model comparison by MAE")
    ax.invert_yaxis()

    for bar, mae in zip(bars, df_plot["MAE"]):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{mae:.1f}",
            va="center",
            ha="left",
            fontsize=8,
        )

    # Vertical reference line at best (smallest) MAE
    best_mae = df_plot["MAE"].min()
    ax.axvline(best_mae, color="tab:blue",
               linestyle="--", linewidth=1, alpha=0.5)

    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()

    out_path = fig_dir / "model_mae_bar.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def get_model_family(name: str) -> str:
    """Return the family label for a given short model name"""
    for family, members in MODEL_FAMILIES.items():
        if name in members:
            return family
    return "Other"


def plot_mae_vs_r2_broken_axis(df: pd.DataFrame, fig_dir: Path,
                               cmap_name: str = "Paired"):
    """
    MAE vs R² with a broken x-axis so the linear models (large MAE)
    and the tree/boosted models (small MAE) are easier to see
    """
    # Drop XGBoost if present
    df_plot = df[~df["model"].str.startswith("xgb_")].copy()

    df_plot["family"] = df_plot["model"].apply(get_model_family)
    families = df_plot["family"].unique()

    cmap = plt.colormaps[cmap_name]
    print(f"Using colormap: {cmap_name} for families: {families}")

    family_colors = {fam: cmap(i) for i, fam in enumerate(families)}

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        sharey=True,
        figsize=(11, 7),
        gridspec_kw={"width_ratios": [3, 1]},
    )

    left_mask = df_plot["MAE"] < 140
    right_mask = ~left_mask

    def scatter_family(ax, mask):
        for fam in families:
            sub = df_plot[mask & (df_plot["family"] == fam)]
            if sub.empty:
                continue
            ax.scatter(
                sub["MAE"],
                sub["R2"],
                label=fam if ax is ax1 else None,
                color=family_colors[fam],
                alpha=0.7,
                edgecolors="none",
                s=45,
            )

    scatter_family(ax1, left_mask)
    scatter_family(ax2, right_mask)

    # Manually defined label offsets
    label_offsets = {
        "ols":            (0,   0.030),
        "rr_1":           (-3, -0.004),
        "rr_10":          (4,  -0.005),
        "rr_0.1":         (6,   0.004),
        "lasso":          (4,   0.020),
        "enet":           (-3,  0.025),

        "prr2":           (9,   0.020),
        "prr3":           (8,   0.020),

        "dt_5":           (-5, -0.020),
        "dt_10":          (6,  -0.015),
        "dt_25":          (-10, 0.010),
        "dt_50":          (-12,-0.010),

        "rf_6":           (4,  -0.020),
        "rf_12":          (2,  -0.030),

        "gbr_base":       (-6, -0.020),
        "gbr_slow":       (-3, -0.030),
        "gbr_shallow":    (-4, -0.007),
        "gbr_stochastic": (4,   0.010),
    }

    for _, row in df_plot.iterrows():
        name = row["model"]
        base_x = row["MAE"]
        base_y = row["R2"]

        dx, dy = label_offsets.get(name, (3.0, 0.003))
        label_x = base_x + dx
        label_y = base_y + dy

        ax = ax1 if base_x < 140 else ax2
        fam = row["family"]

        ax.annotate(
            name,
            xy=(base_x, base_y),
            xytext=(label_x, label_y),
            textcoords="data",
            fontsize=7,
            ha="center",
            va="center",
            arrowprops=dict(
                arrowstyle="-",
                color=family_colors[fam],
                alpha=0.6,
                lw=0.8,
            ),
        )

    # Axis limits and visual break
    ax1.set_xlim(60, 130)
    ax2.set_xlim(155, 175)

    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.yaxis.tick_right()

    d = 0.015
    kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    fig.supylabel(r"$R^2$")
    fig.supxlabel("MAE")
    fig.suptitle("Trade-off between MAE and $R^2$ across models", y=0.98)

    ax1.legend(loc="lower left", frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = fig_dir / "model_mae_vs_r2_broken_axis.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def get_best_model(config: ModelConfig,
                   builder: BuildModels,
                   df: pd.DataFrame):
    best_name = df.sort_values("MAE", ascending=True)["model"].iloc[0]
    best_pipe = builder.pipelines[best_name]

    best_pipe.fit(config.X_train, config.y_train)
    y_true = config.y_test
    y_pred = best_pipe.predict(config.X_test)
    return best_name, best_pipe, y_true, y_pred


def plot_pred_vs_actual(best_name: str,
                        y_true: np.ndarray,
                        y_pred: np.ndarray,
                        fig_dir: Path):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6)

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")

    plt.xlabel("Actual Reimbursement")
    plt.ylabel("Predicted Reimbursement")
    plt.title(f"{best_name}: Prediction vs. Actual")
    plt.tight_layout()

    out_path = fig_dir / f"{best_name}_pred_vs_actual.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


def plot_residual_hist(best_name: str,
                       y_true: np.ndarray,
                       y_pred: np.ndarray,
                       fig_dir: Path):
    residuals = y_pred - y_true

    plt.figure(figsize=(10, 5))
    plt.hist(residuals, bins=30)
    plt.xlabel("Residual (Prediction - Actual)")
    plt.ylabel("Count")
    plt.title(f"{best_name}: Residual Distribution")
    plt.tight_layout()

    out_path = fig_dir / f"{best_name}_residual_hist.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


def plot_calibration(best_name: str,
                     y_true: np.ndarray,
                     y_pred: np.ndarray,
                     fig_dir: Path,
                     n_bins: int = 10):
    """
    Bin predictions into quantiles, plot mean predicted vs. mean actual
    in each bin with error bars for variability
    """
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})

    df["bin"] = pd.qcut(df["y_pred"], q=n_bins,
                        labels=False, duplicates="drop")
    grouped = df.groupby("bin")
    mean_pred = grouped["y_pred"].mean()
    mean_actual = grouped["y_true"].mean()
    std_actual = grouped["y_true"].std()

    plt.figure(figsize=(8, 6))
    plt.errorbar(mean_pred, mean_actual, yerr=std_actual, fmt="o-")

    lo = min(mean_pred.min(), mean_actual.min())
    hi = max(mean_pred.max(), mean_actual.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")

    plt.xlabel("Mean Predicted Reimbursement (per bin)")
    plt.ylabel("Mean Actual Reimbursement (per bin)")
    plt.title(f"{best_name}: Calibration Plot (binned by prediction)")
    plt.tight_layout()

    out_path = fig_dir / f"{best_name}_calibration.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


def plot_residuals_vs_pred(best_name: str,
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           fig_dir: Path):
    residuals = y_pred - y_true

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0.0, linestyle="--")

    plt.xlabel("Predicted Reimbursement")
    plt.ylabel("Residual (Prediction - Actual)")
    plt.title(f"{best_name}: Residuals vs. Predicted")
    plt.tight_layout()

    out_path = fig_dir / f"{best_name}_residuals_vs_pred.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


def save_error_quantiles(best_name: str,
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         fig_dir: Path):
    abs_err = np.abs(y_pred - y_true)
    mae = abs_err.mean()
    rmse = np.sqrt(((y_pred - y_true) ** 2).mean())
    p50 = np.percentile(abs_err, 50)
    p90 = np.percentile(abs_err, 90)
    p95 = np.percentile(abs_err, 95)
    maxe = abs_err.max()

    df = pd.DataFrame(
        [
            {
                "model": best_name,
                "MAE": mae,
                "RMSE": rmse,
                "P50": p50,
                "P90": p90,
                "P95": p95,
                "MaxE": maxe,
            }
        ]
    )

    out_csv = fig_dir / f"{best_name}_error_quantiles.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")


def main():
    _, data_path, fig_dir = get_paths()
    # Run YAML-defined models and save model_metrics.csv
    config, builder, metrics_df = collect_metrics(data_path, fig_dir)
    # Model-comparison plots
    plot_mae_bar(metrics_df, fig_dir)
    plot_mae_vs_r2_broken_axis(metrics_df, fig_dir)
    # Focus on the single best model by MAE
    best_name, best_pipe, y_true, y_pred = get_best_model(
        config, builder, metrics_df
    )
    plot_pred_vs_actual(best_name, y_true, y_pred, fig_dir)
    plot_residual_hist(best_name, y_true, y_pred, fig_dir)
    plot_calibration(best_name, y_true, y_pred, fig_dir)
    plot_residuals_vs_pred(best_name, y_true, y_pred, fig_dir)
    save_error_quantiles(best_name, y_true, y_pred, fig_dir)


if __name__ == "__main__":
    main()
