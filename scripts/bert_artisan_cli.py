"""
bert_artisan_cli.py

Interface en ligne de commande pour utiliser rapidement les principales étapes 
du projet Analyse d'entretiens des artisans : nettoyage, détection d'outils,
augmentation, tuning, entraînement, prédiction et génération de rapports.

But : permettre à un utilisateur non-expert d'exécuter le pipeline complet
ou des étapes isolées avec une seule commande lisible.

Usage (exemples) :
  # Nettoyer tous les .docx du dossier data/raw (mode sentence)
  python scripts/bert_artisan_cli.py clean --input data/raw --output data/processed --mode sentence

  # Lancer la détection d'outils par comparaison à la liste Wikipedia
  # python scripts/bert_artisan_cli.py tools --processed-dir data/processed --tool-output-dir data/tool_detection --results-dir results/tool_detection

  # Augmenter un fichier annoté
  python scripts/bert_artisan_cli.py augment --input data/annotation/sentences_annotated.csv --output data/annotation_augmented --num-aug 2

  # Pipeline verbalisation complet (tuning + train + predict + report) pour BERT
  python scripts/bert_artisan_cli.py verbalisation --input data/annotation/sentences_annotated.csv --mode bert --run full

Notes :
- Le script ajoute automatiquement 'src' au PYTHONPATH pour importer les modules du projet.
- Les messages et logs sont en français.
- La détection d'outils utilise la comparaison avec la liste Wikipedia.
- Les dictionnaires par artisanat/matériau sont optionnels (nécessitent recap_entretien.csv).
"""

# pylint: skip-file

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project 'src' to sys.path so imports work when launching from project root
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Imports from the project's source modules
from utils.logger_config import setup_logger

# Import lazily heavy classes later inside functions to avoid importing heavy deps at CLI startup
logger = setup_logger(__name__, level="INFO")


def cmd_clean(args: argparse.Namespace) -> None:
    """
    Nettoyer des fichiers .docx et produire CSV nettoyés.
    """
    from preprocessing.text_cleaning import InterviewCleaner

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        logger.error("Le dossier d'entrée n'existe pas : %s", input_dir)
        sys.exit(1)

    cleaner = InterviewCleaner()
    modes = [args.mode] if args.mode else ["paragraph"]
    for mode in modes:
        out_name = {
            "paragraph": "cleaned_paragraph.csv",
            "sentence": "cleaned_sentence.csv",
            "full": "cleaned_full.csv",
        }.get(mode, f"cleaned_{mode}.csv")
        output_path = output_dir / out_name
        logger.info("Lancement du nettoyage (mode=%s)...", mode)
        cleaner.batch_process(input_dir, output_path, mode=mode)
    logger.info("Nettoyage terminé.")


def cmd_tools(args: argparse.Namespace) -> None:
    """
    Pipeline de détection d'outils :
      - Extraction Wikipedia (si absent ou --force-extract)
      - Détection par comparaison stricte sur CSV nettoyés (les CSV sources doivent être dans --processed-dir)
      - Optionnel : création des dictionnaires par matériau/artisanat (si --build-dicts)
      - Résultats organisés en deux dossiers : `--tool-output-dir` (intermédiaires) et `--results-dir` (csv + html)
    """
    import shutil

    from preprocessing.extract_tool_wiki import extract_tools
    from processing.build_tool_dicts import generate_tool_dicts
    from processing.tool_detection_word_comparaison_strict import process_all_csvs
    from utils.csv_to_html import csv_to_html
    from utils.csv_to_html_highlight_tool import process_csv_to_html as process_csv_highlight

    processed_dir = Path(args.processed_dir)
    tool_output_dir = Path(args.tool_output_dir)
    results_dir = Path(args.results_dir)

    # Prepare directories
    tool_output_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "csv").mkdir(parents=True, exist_ok=True)
    (results_dir / "html" / "csv").mkdir(parents=True, exist_ok=True)
    (results_dir / "html" / "highlight").mkdir(parents=True, exist_ok=True)

    list_tool_path = tool_output_dir / "list_tool_wiki.csv"

    # 1) Extraction Wikipedia
    if list_tool_path.exists() and not args.force_extract:
        logger.info("Fichier %s trouvé, extraction Wikipedia SKIPÉE (utilisez --force-extract pour forcer).", list_tool_path)
    else:
        logger.info("Fichier %s absent ou extraction forcée ; lancement de l'extraction Wikipedia...", list_tool_path)
        try:
            extract_tools(tool_output_dir)
        except Exception:
            logger.exception("Échec de l'extraction Wikipedia. Vérifiez votre connexion réseau ou l'URL.")
            sys.exit(1)

    # 2) Détection par comparaison sur les CSV nettoyés
    default_inputs = [
        processed_dir / "cleaned_paragraph.csv",
        processed_dir / "cleaned_sentence.csv",
        processed_dir / "cleaned_full.csv",
    ]

    logger.info("Début de la détection d'outils par comparaison sur les CSV nettoyés...")
    process_all_csvs(input_csvs=default_inputs, output_dir=tool_output_dir, tool_csv=list_tool_path)

    # 3) Copier les CSV importants dans results/csv
    patterns = ["list_tool_wiki.csv", "*_with_tools.csv", "*_tool_dict.csv", "dict_outils_*.csv"]
    copied = []
    for pat in patterns:
        for f in sorted(tool_output_dir.glob(pat)):
            try:
                dest = results_dir / "csv" / f.name
                shutil.copy(f, dest)
                copied.append(dest)
                logger.info("Copié: %s -> %s", f, dest)
            except Exception:
                logger.exception("Erreur lors de la copie de %s vers %s", f, results_dir / "csv")

    # 4) Générer HTML des CSV de résultats si demandé
    if args.generate_html:
        # Convertir tous les CSV copiés en HTML (résultats csv)
        for csv_file in copied:
            try:
                logger.info("Conversion CSV->HTML pour %s", csv_file.name)
                csv_to_html(str(csv_file), str(results_dir / "html" / "csv"), separator=",")
            except Exception:
                logger.exception("Erreur lors de la conversion en HTML pour %s", csv_file)

        # Générer les highlights pour tous les fichiers *_with_tools.csv (version texte surlignée)
        for with_tools in sorted(tool_output_dir.glob("*_with_tools.csv")):
            try:
                logger.info("Génération du HTML highlight pour %s", with_tools.name)
                process_csv_highlight(with_tools, results_dir / "html" / "highlight")
            except Exception:
                logger.exception("Erreur lors de la génération du highlight HTML pour %s", with_tools)

    # 5) Construire dictionnaires par artisanat/matériau (OPTIONNEL)
    if args.build_dicts:
        entretien_csv = Path(args.recap_entretien)
        if entretien_csv.exists():
            logger.info("Génération des dictionnaires d'outils par catégorie (matériau/artisanat)...")

            # On cherche de préférence cleaned_full_with_tools.csv
            csv_tools = list(tool_output_dir.glob("cleaned_full_with_tools.csv"))
            if not csv_tools:
                # fallback to any _with_tools.csv
                csv_tools = list(tool_output_dir.glob("*_with_tools.csv"))

            if not csv_tools:
                logger.error(
                    "Aucun fichier *_with_tools.csv trouvé dans %s. Impossible de créer les dictionnaires.", tool_output_dir
                )
            else:
                tool_csv_choice = csv_tools[0]
                try:
                    generate_tool_dicts(entretien_csv, tool_csv_choice, tool_output_dir)
                    # copy generated dicts to results csv
                    for dict_file in tool_output_dir.glob("dict_outils_*.csv"):
                        shutil.copy(dict_file, results_dir / "csv" / dict_file.name)
                        logger.info("Dictionnaire copié: %s", dict_file)
                except Exception:
                    logger.exception("Erreur lors de la création des dictionnaires. Vérifiez les formats des CSV.")
        else:
            logger.warning("Fichier recap_entretien (%s) absent ; dictionnaires non générés (optionnel).", entretien_csv)

    logger.info("Pipeline 'tools' terminé. Résultats finaux dans %s", results_dir)


def cmd_augment(args: argparse.Namespace) -> None:
    """
    Data augmentation pour un fichier CSV annoté.
    """
    from preprocessing.text_augmentation import process_csv as augment_process_csv

    input_path = Path(args.input)
    output_path = Path(args.output)
    text_column = args.text_column
    augmenter_types = args.augmenter_types or ["contextual", "translation", "swap"]
    num_aug = args.num_aug

    if not input_path.exists():
        logger.error("Fichier d'entrée introuvable : %s", input_path)
        sys.exit(1)

    logger.info("Lancement de l'augmentation (%s) sur %s", ", ".join(augmenter_types), input_path)
    augment_process_csv(str(input_path), str(output_path), text_column, augmenter_types, num_aug)
    logger.info("Augmentation terminée. Sortie : %s", output_path)


def cmd_verbalisation(args: argparse.Namespace) -> None:
    """
    Pipeline verbalisation (augment, tune, train, predict, full).

    - Tuning / training work on the annotated (possibly augmented) CSVs.
    - Predictions are run on the *sentences* CSV (cleaned sentences file).
    - Hyperparams and prediction CSVs are written to --verb-data.
    - Final results and HTML reports are copied to --results-dir with subfolders csv and html.
    """
    mode = args.mode  # 'bert' | 'reg' | 'both'
    action = args.run

    input_csv = Path(args.input)
    augmented_csv = Path(args.augmented_csv)
    clean_sentences = Path(args.clean_sentences)
    clean_full = Path(args.clean_full)

    verb_data = Path(args.verb_data)
    results_dir = Path(args.results_dir)

    verb_data.mkdir(parents=True, exist_ok=True)
    (results_dir / "csv").mkdir(parents=True, exist_ok=True)
    (results_dir / "html" / "csv").mkdir(parents=True, exist_ok=True)
    (results_dir / "html" / "reports").mkdir(parents=True, exist_ok=True)

    # 0) Augment (when asked)
    if action in ("augment", "full") and args.do_augment:
        logger.info("Étape d'augmentation demandée...")
        cmd_augment(
            argparse.Namespace(
                input=str(input_csv),
                output=str(augmented_csv),
                text_column=args.text_column,
                augmenter_types=args.augmenter_types,
                num_aug=args.num_aug,
            )
        )

    # 1) Tuning
    if action in ("tune", "full"):
        if mode in ("bert", "both"):
            logger.info("Tuning hyperparamètres pour BERT... (col=%s)", args.target_col)
            from processing.tune_verbalisation_bert import BertHyperparameterTuner

            bert_tune_out = verb_data / f"{args.target_col}_bert_tuning_results.csv"
            tuner = BertHyperparameterTuner(
                model_name=args.bert_model_name, score_col=args.target_col, score_scale=args.score_scale
            )
            tuner.run(augmented_csv, bert_tune_out)

        if mode in ("reg", "both"):
            logger.info("Tuning hyperparamètres pour la régression (TF-IDF + Ridge)...")
            from processing.tune_verbalisation_hyperparams import VerbRegHyperparameterTuner

            reg_tune_out = verb_data / "regression_hyperparams.csv"
            reg_tuner = VerbRegHyperparameterTuner(model_dir=Path(args.model_reg_dir))
            reg_tuner.run(augmented_csv, reg_tune_out, cv=args.cv)

    # 2) Train
    if action in ("train", "full"):
        if mode in ("bert", "both"):
            logger.info("Entraînement final BERT... (col=%s)", args.target_col)
            from processing.train_verbalisation_bert import BertFinalTrainer

            trainer = BertFinalTrainer(
                model_name=args.bert_model_name,
                model_dir=Path(args.model_bert_dir),
                score_col=args.target_col,
                score_scale=args.score_scale,
            )
            trainer.run(augmented_csv, Path(verb_data / f"{args.target_col}_bert_tuning_results.csv"))

        if mode in ("reg", "both"):
            logger.info("Entraînement modèle de régression...")
            from processing.train_verbalisation_regression import VerbRegTrainer

            reg_trainer = VerbRegTrainer(model_dir=Path(args.model_reg_dir))
            reg_trainer.run(augmented_csv, Path(args.model_reg_dir) / "verbalisation_best_params.pkl", with_eval=not args.no_eval)

    # 3) Predict (predictions ALWAYS run on the sentences CSV)
    pred_bert_out = verb_data / "verbalisation_bert_predictions.csv"
    pred_reg_out = verb_data / "verbalisation_reg_predictions.csv"

    if action in ("predict", "full"):
        if mode in ("bert", "both"):
            logger.info("Prédiction BERT sur le fichier de phrases : %s (col=%s)", clean_sentences, args.target_col)
            from processing.predict_verbalisation_bert import BertPredictor

            pred_bert_out = verb_data / f"{args.target_col}_verbalisation_bert_predictions.csv"
            predictor = BertPredictor(
                model_dir=Path(args.model_bert_dir), score_col=args.target_col, score_scale=args.score_scale
            )
            predictor.run(clean_sentences, pred_bert_out, max_length=int(args.max_length))

        if mode in ("reg", "both"):
            logger.info("Prédiction regression sur le fichier de phrases : %s", clean_sentences)
            from processing.predict_verbalisation_regression import VerbRegPredicter

            reg_pred = VerbRegPredicter(model_dir=Path(args.model_reg_dir))
            reg_pred.run(clean_sentences, pred_reg_out)

    # 4) Merge + Reports + HTML export
    # Copy CSV predictions to results and create HTMLs
    import shutil

    for csv_path in [p for p in [pred_bert_out, pred_reg_out] if p.exists()]:
        try:
            shutil.copy(csv_path, results_dir / "csv" / csv_path.name)
            # convert to HTML
            from utils.csv_to_html import csv_to_html

            csv_to_html(str(csv_path), str(results_dir / "html" / "csv"), separator=",")
        except Exception:
            logger.exception("Erreur lors de la copie/conversion du fichier %s", csv_path)

    # If both and both preds exist, merge
    merged_out = verb_data / "verbalisation_merged.csv"
    if mode == "both" and (pred_bert_out.exists() and pred_reg_out.exists()):
        logger.info("Fusion des prédictions BERT + regression...")
        from utils.merge_prediction_verbalisation import PredictionsMerger

        merger = PredictionsMerger()
        merger.run(pred_reg_out, pred_bert_out, merged_out)
        shutil.copy(merged_out, results_dir / "csv" / merged_out.name)
        from utils.csv_to_html import csv_to_html

        csv_to_html(str(merged_out), str(results_dir / "html" / "csv"), separator=",")

    # Generate both versions of the verbalisation report
    if action in ("predict", "full"):
        # Choose which phrases/pred file to use for reporting
        report_phrases = merged_out if merged_out.exists() and args.report_on == "merged" else pred_bert_out
        if not report_phrases.exists():
            logger.warning("Fichier de prédiction pour le rapport introuvable: %s", report_phrases)
        else:
            from utils.generate_verbalisation_report import DifficultyReportGenerator as ReportV1
            from utils.generate_verbalisation_report_v2 import DifficultyReportGenerator as ReportV2

            report_v1 = ReportV1()
            report_v2 = ReportV2()

            report_out_v1 = results_dir / "html" / "reports" / "verbalisation_report_v1.html"
            report_out_v2 = results_dir / "html" / "reports" / "verbalisation_report_v2.html"

            interviews_csv = clean_full

            if interviews_csv.exists():
                try:
                    difficulty_col = f"note_bert_{args.target_col}" if args.report_on == "bert" else "moyenne"
                    report_v1.run(report_phrases, interviews_csv, report_out_v1, difficulty_col=difficulty_col)
                    report_v2.run(report_phrases, interviews_csv, report_out_v2, difficulty_col=difficulty_col)
                    logger.info("Rapports générés: %s , %s", report_out_v1, report_out_v2)
                except Exception:
                    logger.exception("Erreur lors de la génération des rapports HTML")
            else:
                logger.warning("Fichier interviews (%s) introuvable, imposs. générer rapports.", interviews_csv)

    logger.info("Pipeline verbalisation terminé. Résultats: %s", results_dir)


def build_parser() -> argparse.ArgumentParser:
    """
    Construire l'argument parser principal avec les sous-commandes.
    Tous les messages destinés à l'utilisateur sont en français.
    """
    parser = argparse.ArgumentParser(description="Outils CLI pour Analyse d'entretiens des artisans")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: clean
    p_clean = subparsers.add_parser("clean", help="Nettoyer des fichiers .docx et générer des CSV nettoyés")
    p_clean.add_argument("--input", type=str, default="data/raw", help="Dossier contenant les .docx (défaut: data/raw)")
    p_clean.add_argument(
        "--output", type=str, default="data/processed", help="Dossier de sortie pour les CSV (défaut: data/processed)"
    )
    p_clean.add_argument(
        "--mode",
        type=str,
        choices=["paragraph", "sentence", "full"],
        default="paragraph",
        help="Mode de segmentation (défaut: paragraph)",
    )

    # Subcommand: tools
    p_tools = subparsers.add_parser("tools", help="Détection d'outils par comparaison avec liste Wikipedia")
    p_tools.add_argument(
        "--processed-dir", type=str, default="data/processed", help="Dossier contenant les CSV nettoyés (défaut: data/processed)"
    )
    p_tools.add_argument(
        "--tool-output-dir",
        type=str,
        default="data/tool_detection",
        help="Dossier pour stocker tous les CSV produits (défaut: data/tool_detection)",
    )
    p_tools.add_argument(
        "--results-dir",
        type=str,
        default="results/tool_detection",
        help="Dossier de résultats finaux (csv + html) (défaut: results/tool_detection)",
    )
    p_tools.add_argument("--generate-html", action="store_true", help="Générer les fichiers HTML (CSV->HTML et highlights)")
    p_tools.add_argument(
        "--build-dicts", action="store_true", help="Créer les dictionnaires par artisanat/matériau (requiert recap_entretien.csv)"
    )
    p_tools.add_argument(
        "--recap-entretien",
        type=str,
        default="data/recap_entretien.csv",
        help="CSV de métadonnées des entretiens (Nom Fichier, Matériau, Artisanat)",
    )
    p_tools.add_argument("--force-extract", action="store_true", help="Forcer le re-téléchargement de la liste Wikipedia")

    # Subcommand: augment
    p_aug = subparsers.add_parser("augment", help="Effectuer la data augmentation sur un CSV annoté")
    p_aug.add_argument("--input", type=str, required=True, help="CSV annoté d'entrée (séparateur ';')")
    p_aug.add_argument("--output", type=str, required=True, help="Dossier ou CSV de sortie")
    p_aug.add_argument("--text-column", type=str, default="text", help="Nom de la colonne contenant le texte (défaut: text)")
    p_aug.add_argument(
        "--augmenter-types", type=str, nargs="*", default=["contextual", "translation", "swap"], help="Types d'augmentation"
    )
    p_aug.add_argument("--num-aug", type=int, default=1, help="Nombre d'augmentations par type (défaut: 1)")

    # Subcommand: verbalisation
    p_verb = subparsers.add_parser("verbalisation", help="Pipeline pour la détection de difficulté de verbalisation")
    p_verb.add_argument(
        "--input",
        type=str,
        default="data/annotation/sentences_annoted_verb.csv",
        help="CSV annoté d'entrée (défaut: data/annotation/sentences_annoted_verb.csv)",
    )
    p_verb.add_argument(
        "--augmented-csv",
        type=str,
        default="data/annotation/sentences_annoted_verb_augmented.csv",
        help="CSV annoté augmenté (défaut)",
    )
    p_verb.add_argument(
        "--clean-sentences",
        type=str,
        default="data/processed/cleaned_sentence.csv",
        help="CSV des phrases (défaut: data/processed/cleaned_sentence.csv)",
    )
    p_verb.add_argument(
        "--clean-full",
        type=str,
        default="data/processed/cleaned_full.csv",
        help="CSV des entretiens complets (défaut: data/processed/cleaned_full.csv)",
    )
    p_verb.add_argument(
        "--mode",
        type=str,
        choices=["bert", "reg", "both"],
        default="bert",
        help="Mode: bert, reg ou both (défaut: bert)",
    )
    p_verb.add_argument(
        "--run",
        type=str,
        choices=["augment", "tune", "train", "predict", "full"],
        default="full",
        help="Action à effectuer (défaut: full)",
    )
    p_verb.add_argument(
        "--verb-data",
        type=str,
        default="data/verbalisation",
        help="Dossier pour hyperparams et prédictions (défaut: data/verbalisation)",
    )
    p_verb.add_argument(
        "--results-dir",
        type=str,
        default="results/verbalisation",
        help="Dossier de résultat final (défaut: results/verbalisation)",
    )
    p_verb.add_argument("--text-column", type=str, default="text", help="Colonne texte (défaut: text)")
    p_verb.add_argument("--cv", type=int, default=5, help="Nombre de folds pour tuning (défaut: 5)")
    p_verb.add_argument("--max-length", type=int, default=128, help="Max token length pour BERT predictor (défaut:128)")
    p_verb.add_argument("--num-aug", type=int, default=1, help="Nombre d'augmentations par type (défaut:1)")
    p_verb.add_argument(
        "--augmenter-types",
        type=str,
        nargs="*",
        default=["contextual", "translation", "swap"],
        help="Types d'augmentation (défaut: contextual translation swap)",
    )
    p_verb.add_argument("--model-reg-dir", type=str, default="models/verbalisation/regression", help="Dossier modèles regression")
    p_verb.add_argument(
        "--model-bert-dir",
        type=str,
        default="models",
        help="Dossier racine pour stocker les modèles bert par colonne (défaut: models)",
    )
    p_verb.add_argument(
        "--bert-model-name",
        type=str,
        default="distilbert-base-multilingual-cased",
        help="Nom du modèle Hugging Face à utiliser pour BERT",
    )
    p_verb.add_argument(
        "--target-col",
        type=str,
        default="difficulté_verbalisation",
        help="Nom de la colonne cible (ex: 'intimité')",
    )
    p_verb.add_argument(
        "--score-scale",
        type=float,
        default=10.0,
        help="Échelle maximale du score annoté (utilisé pour normalisation)",
    )
    p_verb.add_argument(
        "--report-on",
        type=str,
        choices=["bert", "merged"],
        default="bert",
        help="Sur quel modèle générer le rapport (défaut: bert)",
    )
    p_verb.add_argument("--do-augment", action="store_true", help="Effectuer l'augmentation avant les autres étapes")
    p_verb.add_argument("--no-eval", action="store_true", help="Ne pas évaluer lors de l'entraînement regression")

    return parser


def main() -> None:
    """Point d'entrée du CLI"""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "clean":
        cmd_clean(args)
    elif args.command == "tools":
        cmd_tools(args)
    elif args.command == "augment":
        cmd_augment(args)
    elif args.command == "verbalisation":
        cmd_verbalisation(args)
    else:
        logger.error("Commande non reconnue : %s", args.command)
        parser.print_help()


if __name__ == "__main__":
    main()
