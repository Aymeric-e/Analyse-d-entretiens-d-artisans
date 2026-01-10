"""
statistiques_correlation.py

Compute correlations (Pearson, Spearman) between a target score and candidate factor columns.
Advanced analysis included: VIF (Multicollinearity), Residual Analysis, SHAP Interactions, and SHAP Clustering.

Usage:
    poetry run python src/processing/statistiques_correlation.py \
        --input results/intimite/bert/all_scores_predictions.csv \
        --target note_bert_intimité \
        --output results/intimite/analyse_stat

Outputs saved in the output directory:
 - correlations.csv, heatmap.png, scatter plots
 - vif_analysis.csv (Variance Inflation Factor)
 - shap_summary.png, shap_bar.png
 - shap_interaction.png (Interaction matrix)
 - residuals_analysis.png (Model error analysis)
 - shap_clustering_profiles.png (Segmentation of intimacy types)
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress, pearsonr, spearmanr

# Try to import sklearn, shap, and statsmodels
try:
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import shap

    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant

    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

from utils.logger_config import setup_logger

logger = setup_logger(__name__, level="INFO")


def infer_default_columns(df: pd.DataFrame, target_hint: Optional[str] = None):
    """Infer a default target and factor columns from the dataframe columns if not provided."""
    cols = list(df.columns)
    if target_hint and target_hint in cols:
        target = target_hint
    else:
        candidates = [c for c in cols if "intimit" in c.lower() or "intimite" in c.lower()]
        target = candidates[0] if candidates else cols[3] if len(cols) > 3 else cols[0]

    factor_candidates = [c for c in cols if c.startswith("note_") and c != target]
    if not factor_candidates:
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        factor_candidates = [c for c in num_cols if c != target]

    return target, factor_candidates


def compute_correlations(df: pd.DataFrame, target: str, factors: List[str]):
    results = []
    for f in factors:
        try:
            x = df[target].values
            y = df[f].values
            pearson_val, pearson_p = pearsonr(x, y)
        except Exception:
            pearson_val, pearson_p = np.nan, np.nan
        try:
            spearman_val, spearman_p = spearmanr(x, y)
        except Exception:
            spearman_val, spearman_p = np.nan, np.nan
        try:
            lr = linregress(x, y)
            slope = float(lr.slope)
            intercept = float(lr.intercept)
            slope_p = float(lr.pvalue)
            rvalue = float(lr.rvalue)
        except Exception:
            slope, intercept, slope_p, rvalue = np.nan, np.nan, np.nan, np.nan

        results.append(
            {
                "factor": f,
                "pearson": pearson_val,
                "pearson_p": pearson_p,
                "spearman": spearman_val,
                "spearman_p": spearman_p,
                "slope": slope,
                "intercept": intercept,
                "slope_p": slope_p,
                "rvalue": rvalue,
            }
        )
    return pd.DataFrame(results)


def compute_vif(df: pd.DataFrame, factors: List[str], out_dir: Path):
    """Compute Variance Inflation Factor to detect multicollinearity."""
    if not STATSMODELS_AVAILABLE:
        logger.warning("statsmodels not available; skipping VIF analysis")
        return

    try:
        X = df[factors].copy()
        # VIF requires a constant term to be accurate
        X = add_constant(X)

        vif_data = pd.DataFrame()
        vif_data["factor"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        # Filter out the constant for the report
        vif_data = vif_data[vif_data["factor"] != "const"].sort_values(by="VIF", ascending=False)

        out_path = out_dir / "vif_analysis.csv"
        vif_data.to_csv(out_path, index=False, sep=";")
        logger.info("VIF analysis saved: %s", out_path)
    except Exception:
        logger.exception("Error computing VIF")


def plot_heatmap(corr_df: pd.DataFrame, target: str, out_path: Path):
    plt.figure(figsize=(6, len(corr_df) * 0.5 + 2))
    sns.set(style="whitegrid")
    mat = corr_df.set_index("factor")["pearson"].to_frame()
    sns.heatmap(mat, annot=True, cmap="coolwarm", vmin=-1, vmax=1, cbar_kws={"label": "Pearson r"})
    plt.title(f"Pearson correlation with {target}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_scatter(df: pd.DataFrame, target: str, factor: str, out_path: Path):
    plt.figure(figsize=(6, 4))
    sns.regplot(x=factor, y=target, data=df, scatter_kws={"s": 10, "alpha": 0.6}, line_kws={"color": "red"}, ci=None)
    plt.xlabel(factor)
    plt.ylabel(target)
    plt.title(f"{factor} vs {target}")
    try:
        lr = linregress(df[factor].values, df[target].values)
        text = f"slope={lr.slope:.3f}\nintercept={lr.intercept:.3f}\nr={lr.rvalue:.3f}"
        plt.gca().text(
            0.05,
            0.95,
            text,
            transform=plt.gca().transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
    except Exception:
        pass
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def analyze_residuals(y_test, y_pred, out_dir: Path):
    """Plot residuals to analyze model errors."""
    residuals = y_test - y_pred
    plt.figure(figsize=(12, 5))

    # Scatter: Predicted vs Residuals
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals (Real - Pred)")
    plt.title("Residuals vs Predictions")

    # Hist: Residual Distribution
    plt.subplot(1, 2, 2)
    sns.histplot(residuals, kde=True)
    plt.title("Distribution of Residuals")

    plt.tight_layout()
    plt.savefig(out_dir / "residuals_analysis.png", dpi=150)
    plt.close()


def analyze_shap_clustering(shap_values, factors, out_dir: Path):
    """Cluster SHAP values to identify 'Intimacy Profiles'."""
    try:
        # K-Means clustering on SHAP values
        # k=3 is often a good start for Low/Med/High or distinct styles
        k = 3
        kmeans = KMeans(n_clusters=k, random_state=42)
        _ = kmeans.fit_predict(shap_values.values)

        # Create a DataFrame for the cluster centers (Mean SHAP impact per cluster)
        cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=factors)
        cluster_centers["Cluster_ID"] = [f"Cluster {i}" for i in range(k)]
        cluster_centers = cluster_centers.set_index("Cluster_ID")

        plt.figure(figsize=(10, 6))
        sns.heatmap(cluster_centers, cmap="coolwarm", center=0, annot=True, fmt=".3f")
        plt.title("SHAP Clustering Profiles (Mean Impact per Factor)")
        plt.tight_layout()
        plt.savefig(out_dir / "shap_clustering_profiles.png", dpi=150)
        plt.close()

        logger.info("SHAP clustering analysis saved")
    except Exception:
        logger.exception("Error during SHAP clustering")


def compute_shap_advanced(df: pd.DataFrame, target: str, factors: List[str], out_dir: Path):
    if not (SKLEARN_AVAILABLE and SHAP_AVAILABLE):
        logger.warning("Libraries for SHAP/ML missing.")
        return None

    X = df[factors]
    y = df[target]

    # 1. Train Model (The 'Analysis Model')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 2. Residual Analysis
    y_pred = model.predict(X_test)
    analyze_residuals(y_test, y_pred, out_dir)

    # 3. Standard SHAP
    # Use TreeExplainer for exact interaction values with Random Forest
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # Beeswarm
    try:
        plt.figure(figsize=(8, 6))
        shap.plots.beeswarm(shap_values, show=False)
        plt.title("SHAP summary (beeswarm)")
        plt.tight_layout()
        plt.savefig(out_dir / "shap_summary.png", dpi=150)
        plt.close()
    except Exception:
        pass

    # Bar plot
    try:
        mean_abs = np.abs(shap_values.values).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        features = np.array(factors)[order]
        vals = mean_abs[order]
        plt.figure(figsize=(8, max(4, len(factors) * 0.4)))
        sns.barplot(x=vals, y=features, palette="viridis")
        plt.xlabel("Mean |SHAP value|")
        plt.title("Feature importance (SHAP)")
        plt.tight_layout()
        plt.savefig(out_dir / "shap_bar.png", dpi=150)
        plt.close()
    except Exception:
        pass

    # 4. SHAP Interaction Values
    try:
        # Calculate interaction values (shape: n_samples x n_features x n_features)
        shap_interaction_values = explainer.shap_interaction_values(X_test)

        # Plot summary of interaction
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_interaction_values, X_test, show=False)
        plt.title("SHAP Interaction Summary")
        plt.tight_layout()
        plt.savefig(out_dir / "shap_interaction.png", dpi=150)
        plt.close()
    except Exception:
        logger.exception("Error computing SHAP interactions")

    # 5. SHAP Clustering (Segmentation)
    analyze_shap_clustering(shap_values, factors, out_dir)

    return True


def main():
    parser = argparse.ArgumentParser(description="Compute correlations, VIF, and advanced SHAP analysis")
    parser.add_argument(
        "--input", type=Path, default=Path("results/intimite/bert/all_scores_predictions.csv"), help="CSV input file"
    )
    parser.add_argument("--target", type=str, default=None, help="Target column name")
    parser.add_argument("--factors", type=str, nargs="*", default=None, help="Factor column names")
    parser.add_argument("--output-dir", type=Path, default=Path("results/intimite/analyse_stat"), help="Output folder")
    parser.add_argument("--no-shap", action="store_true", help="Disable SHAP explanation")

    args = parser.parse_args()

    if not args.input.exists():
        logger.error("Input CSV not found: %s", args.input)
        sys.exit(1)

    df = pd.read_csv(args.input, sep=";")
    logger.info("Input loaded: %d rows", len(df))

    target, default_factors = infer_default_columns(df, target_hint=args.target)
    factors = args.factors if args.factors else default_factors

    logger.info("Target: %s", target)
    logger.info("Factors: %s", factors)

    missing = [c for c in [target] + factors if c not in df.columns]
    if missing:
        logger.error("Colonnes manquantes: %s", missing)
        sys.exit(1)

    use_cols = [target] + factors
    df_clean = df[use_cols].copy()
    df_clean = df_clean.apply(pd.to_numeric, errors="coerce")
    df_clean = df_clean.dropna()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Correlations
    corr_df = compute_correlations(df_clean, target, factors)
    corr_csv = out_dir / "correlations.csv"
    corr_df.to_csv(corr_csv, index=False, sep=";")

    # 2. VIF Analysis
    compute_vif(df_clean, factors, out_dir)

    # 3. Visuals
    try:
        plot_heatmap(corr_df, target, out_dir / "heatmap.png")
    except Exception:
        pass

    for f in factors:
        try:
            plot_scatter(df_clean, target, f, out_dir / f"scatter_{f}.png")
        except Exception:
            pass

    # 4. Advanced SHAP Analysis
    if not args.no_shap:
        if SHAP_AVAILABLE and SKLEARN_AVAILABLE:
            compute_shap_advanced(df_clean, target, factors, out_dir)
        else:
            logger.warning("SHAP/Sklearn missing, skipping advanced analysis.")

    # Summary Text
    summary_txt = out_dir / "correlation_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Correlation and Analysis Summary\n")
        f.write(f"Target: {target}\n")
        f.write("\n--- Correlations ---\n")
        for _, row in corr_df.iterrows():
            f.write(f"{row['factor']}: r={row['pearson']:.3f}, rho={row['spearman']:.3f}\n")

        f.write("\n--- Generated Files ---\n")
        f.write("- vif_analysis.csv: Check for multicollinearity (VIF > 5 is high)\n")
        f.write("- residuals_analysis.png: Check model errors\n")
        f.write("- shap_clustering_profiles.png: Segmentation of intimacy types\n")
        f.write("- shap_interaction.png: Interaction effects between factors\n")

    logger.info("Analysis complete. Results in %s", out_dir)


if __name__ == "__main__":
    main()
