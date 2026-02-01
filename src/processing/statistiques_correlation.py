"""
statistiques_correlation.py

Calcule les corrélations (Pearson, Spearman) entre un score cible et des colonnes de facteurs candidats.
Analyse avancée incluse : VIF (Multicolinéarité), Analyse des Résidus, Interactions SHAP, et Clustering SHAP.

Utilisation :
    poetry run python src/processing/statistiques_correlation.py \
        --input results/intimite/bert/all_scores_predictions.csv \
        --target note_bert_intimité \
        --output results/intimite/analyse_stat \
        --k-clusters 5
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

# Essayer d'importer sklearn, shap, et statsmodels
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
    """Infère une cible par défaut et des colonnes de facteurs à partir des colonnes du dataframe si non fournies."""
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
    """Calcule les corrélations entre la cible et les facteurs."""
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


def compute_vif(df: pd.DataFrame, factors: List[str], out_dir: Path = None):
    """Calcule le Facteur d'Inflation de Variance pour détecter la multicolinéarité."""
    if not STATSMODELS_AVAILABLE:
        logger.warning("statsmodels non disponible ; analyse VIF ignorée")
        return None

    try:
        X = df[factors].copy()
        # VIF nécessite un terme constant pour être précis
        X = add_constant(X)

        vif_data = pd.DataFrame()
        vif_data["factor"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        # Filtrer le constant pour le rapport
        vif_data = vif_data[vif_data["factor"] != "const"].sort_values(by="VIF", ascending=False)

        if out_dir:
            out_path = out_dir / "vif_analysis.csv"
            vif_data.to_csv(out_path, index=False, sep=";")
            logger.info("Analyse VIF sauvegardée : %s", out_path)
        return vif_data
    except Exception:
        logger.exception("Erreur lors du calcul VIF")
        return None


def plot_heatmap(corr_df: pd.DataFrame, target: str, out_path: Path):
    """Pour la heatmap de correlation pearson"""
    plt.figure(figsize=(6, len(corr_df) * 0.5 + 2))
    sns.set_theme(style="whitegrid")
    mat = corr_df.set_index("factor")["pearson"].to_frame()
    sns.heatmap(mat, annot=True, cmap="coolwarm", vmin=-1, vmax=1, cbar_kws={"label": "Pearson r"})
    plt.title(f"Corrélation Pearson avec {target}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_scatter(df: pd.DataFrame, target: str, factor: str, out_path: Path):
    """Pour tracer les notes en fonction de chaque facteur + régression"""
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
    """Trace les résidus pour analyser les erreurs du modèle."""
    residuals = y_test - y_pred
    plt.figure(figsize=(12, 5))

    # Scatter : Prédit vs Résidus
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Valeurs Prédites")
    plt.ylabel("Résidus (Réel - Préd)")
    plt.title("Résidus vs Prédictions")

    # Hist : Distribution des Résidus
    plt.subplot(1, 2, 2)
    sns.histplot(residuals, kde=True)
    plt.title("Distribution des Résidus")

    plt.tight_layout()
    plt.savefig(out_dir / "residuals_analysis.png", dpi=150)
    plt.close()


def analyze_shap_clustering(shap_values, factors, out_dir: Path, k: int = 5):
    """Clusterise les valeurs SHAP pour identifier les 'Profils d'Intimité'."""
    try:
        # Clustering K-Means sur les valeurs SHAP
        kmeans = KMeans(n_clusters=k, random_state=42)
        _ = kmeans.fit_predict(shap_values.values)

        # Créer un DataFrame pour les centres de cluster (Impact SHAP moyen par cluster)
        cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=factors)
        cluster_centers["Cluster_ID"] = [f"Cluster {i}" for i in range(k)]
        cluster_centers = cluster_centers.set_index("Cluster_ID")

        plt.figure(figsize=(10, 6))
        sns.heatmap(cluster_centers, cmap="coolwarm", center=0, annot=True, fmt=".3f")
        plt.title("Profils de Clustering SHAP (Impact Moyen par Facteur)")
        plt.tight_layout()
        plt.savefig(out_dir / "shap_clustering_profiles.png", dpi=150)
        plt.close()

        logger.info("Analyse de clustering SHAP sauvegardée")
        return kmeans
    except Exception:
        logger.exception("Erreur lors du clustering SHAP")
        return None


def compute_shap_advanced(
    df_clean: pd.DataFrame, df_original: pd.DataFrame, target: str, factors: List[str], out_dir: Path, k: int = 5
):
    """Fonction réalisant tous les calculs pour l'analyse SHAP"""
    if not (SKLEARN_AVAILABLE and SHAP_AVAILABLE):
        logger.warning("Bibliothèques pour SHAP/ML manquantes.")
        return None

    X = df_clean[factors]
    y = df_clean[target]

    # Entraîner le Modèle (Le 'Modèle d'Analyse')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Analyse des Résidus
    y_pred = model.predict(X_test)
    analyze_residuals(y_test, y_pred, out_dir)

    # SHAP Standard
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # Beeswarm
    try:
        plt.figure(figsize=(8, 6))
        shap.plots.beeswarm(shap_values, show=False)
        plt.title("Résumé SHAP (beeswarm)")
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
        # Correction warning palette: assigner hue=y et legend=False
        sns.barplot(x=vals, y=features, hue=features, palette="viridis", legend=False)
        plt.xlabel("Moyenne |valeur SHAP|")
        plt.title("Importance des caractéristiques (SHAP)")
        plt.tight_layout()
        plt.savefig(out_dir / "shap_bar.png", dpi=150)
        plt.close()
    except Exception:
        pass

    # Valeurs d'Interaction SHAP
    try:
        shap_interaction_values = explainer.shap_interaction_values(X_test)

        # CORRECTION ICI:
        # Au lieu de passer feature_names comme liste (ce qui cause l'erreur d'indexation numpy),
        # on renomme le DataFrame directement. SHAP gère très bien les DataFrames.
        X_test_renamed = X_test.rename(columns=lambda x: x.replace("note_bert_", ""))

        plt.figure(figsize=(12, 10))
        # On passe le DataFrame renommé, et on retire l'argument feature_names
        shap.summary_plot(shap_interaction_values, X_test_renamed, show=False)

        plt.title("Résumé des Interactions SHAP", fontsize=16, pad=20, loc="center")
        plt.tight_layout()
        plt.savefig(out_dir / "shap_interaction.png", dpi=150)
        plt.close()
    except Exception:
        logger.exception("Erreur lors du calcul des interactions SHAP")

    # Clustering SHAP (Segmentation)
    kmeans = analyze_shap_clustering(shap_values, factors, out_dir, k)

    # Ajouter les clusters et résidus
    if kmeans is not None:
        shap_values_all = explainer(df_clean[factors])
        clusters = kmeans.predict(shap_values_all.values)

        # Mise à jour du dataframe original pour sauvegarde
        df_original.loc[df_clean.index, "cluster"] = clusters

        # Mise à jour du dataframe de travail 'df_clean' pour le groupby suivant
        df_clean = df_clean.copy()
        df_clean["cluster"] = clusters

        # Calculer les résidus
        y_pred_all = model.predict(df_clean[factors])
        residuals_all = df_clean[target] - y_pred_all

        # Sauvegarde dans original
        df_original.loc[df_clean.index, "residual"] = residuals_all
        df_original.to_csv(out_dir / "data_with_stats.csv", index=False, sep=";")
        logger.info("CSV avec clusters et résidus sauvegardé")

        # Matrice des moyennes par cluster (heatmap)
        try:
            cluster_means = df_clean.groupby("cluster")[factors + [target]].mean()

            # Créer les annotations avec moyenne et erreur moyenne absolue
            annot_df = pd.DataFrame(index=cluster_means.index, columns=factors + [target])
            for cluster in cluster_means.index:
                group = df_clean[df_clean["cluster"] == cluster]
                for var in factors + [target]:
                    mean_val = cluster_means.loc[cluster, var]
                    mae = (group[var] - mean_val).abs().mean()
                    annot_df.loc[cluster, var] = f"{mean_val:.3f}\n({mae:.3f})"

            plt.figure(figsize=(12, 8))
            sns.heatmap(cluster_means.T, cmap="coolwarm", center=0, annot=annot_df.T, fmt="")
            plt.title("Moyennes des Facteurs et Intimité par Cluster\n(Erreur Moyenne Absolue en parenthèses)")
            plt.tight_layout()
            plt.savefig(out_dir / "cluster_means_heatmap.png", dpi=150)
            plt.close()
            logger.info("Heatmap des moyennes par cluster sauvegardée")
        except Exception:
            logger.exception("Erreur lors de la création de la heatmap des clusters")

    return kmeans


def compute_stats_per_filename(df_with_stats: pd.DataFrame, target: str, factors: List[str], k: int, out_dir: Path):
    """Calcule les stats par filename : moyennes, clusters, corrélations, VIF."""
    # Vérification si filename existe
    if "filename" not in df_with_stats.columns:
        logger.warning("Colonne 'filename' manquante, saut de l'analyse par entretien.")
        return

    grouped = df_with_stats.groupby("filename")
    stats_list = []

    for filename, group in grouped:
        # Gestion du texte (concaténation si présent)
        text_concat = ""
        if "text" in group.columns:
            text_concat = " ".join(group["text"].astype(str).tolist())

        # Conversion forcée en numérique pour les calculs de moyenne/corr
        # (df_with_stats vient de df_original qui peut avoir des types mixtes)
        group_numeric = group.copy()
        cols_to_convert = [target] + factors
        for col in cols_to_convert:
            group_numeric[col] = pd.to_numeric(group[col], errors="coerce")

        # Moyennes
        mean_target = group_numeric[target].mean()
        mean_factors = {f: group_numeric[f].mean() for f in factors}

        # Compte des clusters (si la colonne existe)
        cluster_cols = {}
        if "cluster" in group.columns:
            cluster_counts = group["cluster"].value_counts()
            cluster_cols = {f"cluster_{i}": cluster_counts.get(i, 0) for i in range(k)}

        # Corrélations sur les données numériques nettoyées
        corr_df = compute_correlations(group_numeric.dropna(subset=cols_to_convert), target, factors)
        corr_dict = {}
        for _, row in corr_df.iterrows():
            f = row["factor"]
            corr_dict[f"pearson_{f}"] = row["pearson"]
            corr_dict[f"spearman_{f}"] = row["spearman"]

        # VIF sur les données numériques nettoyées
        # On utilise une fonction VIF qui retourne un dataframe ou None
        # Attention: compute_vif recalcule les VIF globaux, ici on veut local par fichier ?
        # Si le fichier a peu de lignes, VIF plantera ou sera infini. On tente quand même.
        vif_dict = {}
        if len(group_numeric) > len(factors) + 1:  # Minimum de lignes pour VIF
            vif_df_local = compute_vif(group_numeric.dropna(subset=factors), factors)
            if vif_df_local is not None:
                for _, row in vif_df_local.iterrows():
                    f = row["factor"]
                    vif_dict[f"vif_{f}"] = row["VIF"]

        stat_row = {
            "filename": filename,
            "text": text_concat,
            f"mean_{target}": mean_target,
            **{f"mean_{f}": mean_factors[f] for f in factors},
            **cluster_cols,
            **corr_dict,
            **vif_dict,
        }
        stats_list.append(stat_row)

    if stats_list:
        stats_df = pd.DataFrame(stats_list)
        stats_df.to_csv(out_dir / "stat_par_entretien.csv", index=False, sep=";")
        logger.info("Stats par entretien sauvegardées")
    else:
        logger.warning("Aucune stat par entretien générée.")


def main():
    parser = argparse.ArgumentParser(description="Calcule les corrélations, VIF, et analyse SHAP avancée")
    parser.add_argument(
        "--input", type=Path, default=Path("results/intimite/bert/all_scores_predictions.csv"), help="Fichier CSV d'entrée"
    )
    parser.add_argument("--target", type=str, default=None, help="Nom de la colonne cible")
    parser.add_argument("--factors", type=str, nargs="*", default=None, help="Noms des colonnes de facteurs")
    parser.add_argument("--output-dir", type=Path, default=Path("results/intimite/analyse_stat"), help="Dossier de sortie")
    parser.add_argument("--no-shap", action="store_true", help="Désactiver l'explication SHAP")
    parser.add_argument("--k-clusters", type=int, default=5, help="Nombre de clusters pour SHAP")

    args = parser.parse_args()

    if not args.input.exists():
        logger.error("Fichier CSV d'entrée non trouvé : %s", args.input)
        sys.exit(1)

    df = pd.read_csv(args.input, sep=";")
    logger.info("Entrée chargée : %d lignes", len(df))

    target, default_factors = infer_default_columns(df, target_hint=args.target)
    factors = args.factors if args.factors else default_factors

    logger.info("Cible : %s", target)
    logger.info("Facteurs : %s", factors)

    missing = [c for c in [target] + factors if c not in df.columns]
    if missing:
        logger.error("Colonnes manquantes : %s", missing)
        sys.exit(1)

    df_original = df.copy()
    use_cols = [target] + factors
    df_clean = df[use_cols].copy()
    df_clean = df_clean.apply(pd.to_numeric, errors="coerce")
    df_clean = df_clean.dropna()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Corrélations
    corr_df = compute_correlations(df_clean, target, factors)
    corr_csv = out_dir / "correlations.csv"
    corr_df.to_csv(corr_csv, index=False, sep=";")

    # Analyse VIF
    compute_vif(df_clean, factors, out_dir)

    # Visuels
    try:
        plot_heatmap(corr_df, target, out_dir / "heatmap.png")
    except Exception:
        pass

    for f in factors:
        try:
            plot_scatter(df_clean, target, f, out_dir / f"scatter_{f}.png")
        except Exception:
            pass

    # Analyse SHAP Avancée
    kmeans = None
    if not args.no_shap:
        if SHAP_AVAILABLE and SKLEARN_AVAILABLE:
            kmeans = compute_shap_advanced(df_clean, df_original, target, factors, out_dir, args.k_clusters)
        else:
            logger.warning("SHAP/Sklearn manquants, analyse avancée ignorée.")

    # Stats par entretien
    # Utiliser df_original (qui a le filename) mais filtré sur les index propres
    if kmeans is not None:
        df_with_stats = df_original.loc[df_clean.index].copy()
        compute_stats_per_filename(df_with_stats, target, factors, args.k_clusters, out_dir)

    # Résumé Texte
    summary_txt = out_dir / "correlation_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Résumé des Corrélations et Analyses\n")
        f.write(f"Cible : {target}\n")
        f.write("\n--- Corrélations ---\n")
        for _, row in corr_df.iterrows():
            f.write(f"{row['factor']}: r={row['pearson']:.3f}, rho={row['spearman']:.3f}\n")

        f.write("\n--- Fichiers Générés ---\n")
        f.write("- vif_analysis.csv : Vérifier la multicolinéarité (VIF > 5 est élevé)\n")
        f.write("- residuals_analysis.png : Vérifier les erreurs du modèle\n")
        f.write("- shap_clustering_profiles.png : Segmentation des types d'intimité\n")
        f.write("- shap_interaction.png : Effets d'interaction entre facteurs\n")
        f.write("- data_with_clusters.csv : CSV d'entrée avec colonnes cluster et residual\n")
        f.write("- cluster_means_heatmap.png : Matrice des moyennes des facteurs et intimité par cluster\n")
        f.write("- stat_par_entretien.csv : Stats agrégées par filename\n")

    logger.info("Analyse terminée. Résultats dans %s", out_dir)


if __name__ == "__main__":
    main()
