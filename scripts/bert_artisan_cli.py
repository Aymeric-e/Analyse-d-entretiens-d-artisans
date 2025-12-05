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
  python scripts/bert_artisan_cli.py tools --processed-dir data/processed --output-dir data/processed_tools

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
    Pipeline de détection d'outils par comparaison avec la liste Wikipedia :
      - Extraction Wikipedia (si absent)
      - Détection par comparaison stricte sur CSV nettoyés
      - Génération de fichiers HTML avec mise en évidence (optionnel)
      - Création de dictionnaires par artisanat/matériau (optionnel, si recap_entretien.csv présent)
    """
    from preprocessing.extract_tool_wiki import extract_tools
    from processing import build_tool_dicts
    from processing.tool_detection_word_comparaison_strict import process_all_csvs
    from utils.csv_to_html_highlight_tool import process_csv_to_html

    list_tool_path = ROOT / "data" / "list_tool.csv"

    # 1) Extraction Wikipedia si nécessaire
    if list_tool_path.exists() and not args.force_extract:
        logger.info("Fichier %s trouvé, extraction Wikipedia SKIPÉE (utilisez --force-extract pour forcer).", list_tool_path)
    else:
        logger.info("Fichier %s absent ou extraction forcée ; lancement de l'extraction Wikipedia...", list_tool_path)
        try:
            extract_tools()
        except Exception:
            logger.exception("Échec de l'extraction Wikipedia. Vérifiez votre connexion réseau ou l'URL.")
            sys.exit(1)

    # 2) Détection par comparaison

    # Determine input CSVs
    processed_dir = Path(args.processed_dir)
    default_inputs = [
        processed_dir / "cleaned_paragraph.csv",
        processed_dir / "cleaned_sentence.csv",
        processed_dir / "cleaned_full.csv",
    ]

    logger.info("Début de la détection d'outils par comparaison sur les CSV nettoyés...")
    process_all_csvs(input_csvs=default_inputs, output_dir=Path(args.output_dir), tool_csv="tool_dictionary.csv")

    # 3) Générer HTML highlight pour chaque CSV produit (optionnel)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.generate_html:
        for csv_file in sorted(output_dir.glob("*.csv")):
            try:
                logger.info("Génération HTML de mise en évidence pour %s", csv_file.name)
                process_csv_to_html(csv_file, Path(args.html_output_dir))
            except Exception:
                logger.exception("Erreur lors de la génération HTML pour %s", csv_file)

    # 4) Construire dictionnaires par artisanat/matériau (OPTIONNEL - si recap_entretien.csv existe)
    if args.build_dicts:
        entretien_csv = Path(args.recap_entretien)
        if entretien_csv.exists():
            logger.info("Génération des dictionnaires d'outils par catégorie (matériau/artisanat)...")
            try:
                file_to_materiau, file_to_artisanat = build_tool_dicts.load_entretien_file(entretien_csv)
                tools_csv = output_dir / "tool_dictionary.csv"
                if tools_csv.exists():
                    file_to_tools = build_tool_dicts.load_tools_file(tools_csv)
                    dict_materiau = build_tool_dicts.count_tools_by_category(file_to_materiau, file_to_tools)
                    dict_artisanat = build_tool_dicts.count_tools_by_category(file_to_artisanat, file_to_tools)
                    out_dir = Path(args.dict_output_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    build_tool_dicts.write_output_csv(out_dir / "dict_outils_materiau.csv", "materiau", dict_materiau)
                    build_tool_dicts.write_output_csv(out_dir / "dict_outils_artisanat.csv", "artisanat", dict_artisanat)
                    logger.info("Dictionnaires créés dans %s", out_dir)
                else:
                    logger.warning("Fichier tool_dictionary.csv non trouvé, impossible de créer les dictionnaires.")
            except Exception:
                logger.exception("Erreur lors de la création des dictionnaires. Vérifiez les formats des CSV.")
        else:
            logger.warning("Fichier recap_entretien (%s) absent ; dictionnaires non générés (c'est optionnel).", entretien_csv)


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
    Pipeline verbalisation (augmentation, tuning, train, predict, report).
    Modes supportés : 'bert', 'regression', 'both'
    Actions : 'augment' | 'tune' | 'train' | 'predict' | 'full'
    """
    mode = args.mode  # 'bert' | 'regression' | 'both'
    action = args.run  # 'augment'|'tune'|'train'|'predict'|'full'
    input_csv = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Helper flags and paths
    augmented_csv = Path(args.augmented_csv) if args.augmented_csv else input_csv
    bert_tune_out = Path(args.bert_tune_out)
    bert_model_dir = Path(args.bert_model_dir)
    reg_best_params = Path(args.reg_best_params)
    reg_model_dir = Path(args.reg_model_dir)
    reports_dir = Path(args.reports_dir)

    # 0) If augment requested as part of a full run
    if action in ("augment", "full") and args.do_augment:
        logger.info("Étape d'augmentation demandée...")
        cmd_augment(
            argparse.Namespace(
                input=str(input_csv),
                output=args.augment_output,
                text_column=args.text_column,
                augmenter_types=args.augmenter_types,
                num_aug=args.num_aug,
            )
        )
        # Update path to augmented CSV
        augmented_csv = Path(args.augment_output)

    # 1) Tuning
    if action in ("tune", "full"):
        if mode in ("bert", "both"):
            logger.info("Tuning hyperparamètres pour BERT...")
            from processing.tune_verbalisation_bert import BertHyperparameterTuner

            tuner = BertHyperparameterTuner(model_name=args.bert_model_name)
            tuner.run(augmented_csv, bert_tune_out)

        if mode in ("regression", "both"):
            logger.info("Tuning hyperparamètres pour la régression (TF-IDF + Ridge)...")
            from processing.tune_verbalisation_hyperparams import VerbRegHyperparameterTuner

            reg_tuner = VerbRegHyperparameterTuner(model_dir=reg_model_dir)
            reg_tuner.run(augmented_csv, Path(args.reg_tune_out), cv=args.cv)

    # 2) Train
    if action in ("train", "full"):
        if mode in ("bert", "both"):
            logger.info("Entraînement final BERT...")
            from processing.train_verbalisation_bert import BertFinalTrainer

            trainer = BertFinalTrainer(model_name=args.bert_model_name, model_dir=bert_model_dir)
            trainer.run(augmented_csv, Path(bert_tune_out))

        if mode in ("regression", "both"):
            logger.info("Entraînement modèle de régression...")
            from processing.train_verbalisation_regression import VerbRegTrainer

            reg_trainer = VerbRegTrainer(model_dir=reg_model_dir)
            reg_trainer.run(augmented_csv, reg_best_params, with_eval=not args.no_eval)

    # 3) Predict
    # Create prediction output paths
    pred_bert_out = output_dir / "verbalisation_bert.csv"
    pred_reg_out = output_dir / "verbalisation_reg.csv"

    if action in ("predict", "full"):
        if mode in ("bert", "both"):
            logger.info("Prédiction avec BERT...")
            from processing.predict_verbalisation_bert import BertPredictor

            predictor = BertPredictor(model_dir=bert_model_dir)
            predictor.run(augmented_csv, pred_bert_out, max_length=args.max_length)

        if mode in ("regression", "both"):
            logger.info("Prédiction avec modèle de régression...")
            from processing.predict_verbalisation_regression import VerbRegPredicter

            pred = VerbRegPredicter(model_dir=reg_model_dir)
            pred.run(augmented_csv, pred_reg_out)

    # 4) Merge + report if both
    if action in ("full",) and mode == "both":
        logger.info("Fusion des prédictions BERT + regression...")
        from utils.merge_prediction_verbalisation import PredictionsMerger

        merger = PredictionsMerger()
        merged_out = output_dir / "verbalisation_merged.csv"
        merger.run(pred_reg_out, pred_bert_out, merged_out)

        # Generate report (use merged or one of the predictions depending on choice)
        logger.info("Génération du rapport HTML...")
        from utils.generate_verbalisation_report import DifficultyReportGenerator

        report_generator = DifficultyReportGenerator()
        report_html = reports_dir / "verbalisation_report.html"
        interviews_csv = Path(args.interviews_csv)
        if interviews_csv.exists():
            difficulty_col = "note_bert" if args.report_on == "bert" else "moyenne"
            report_generator.run(merged_out, interviews_csv, report_html, difficulty_col=difficulty_col)
            logger.info("Rapport disponible: %s", report_html)
        else:
            logger.warning("Fichier interviews (%s) introuvable, impossible de générer le rapport HTML.", interviews_csv)

    logger.info("Pipeline verbalisation terminée.")


def build_parser() -> argparse.ArgumentParser:
    """
    Construire l'argument parser principal avec les sous-commandes.
    Tous les messages destinés à l'utilisateur sont en français.
    """
    parser = argparse.ArgumentParser(description="Outils CLI pour Analyse d'entretiens des artisans (messages en français)")
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
    p_tools.add_argument("--output-dir", type=str, default="data/processed_tools", help="Dossier de sortie pour CSV détectés")
    p_tools.add_argument("--generate-html", action="store_true", help="Générer les fichiers HTML de mise en évidence")
    p_tools.add_argument(
        "--html-output-dir", type=str, default="results/tool_highlight_html", help="Dossier de sortie pour les fichiers HTML"
    )
    p_tools.add_argument(
        "--build-dicts", action="store_true", help="Créer les dictionnaires par artisanat/matériau (requiert recap_entretien.csv)"
    )
    p_tools.add_argument(
        "--recap-entretien",
        type=str,
        default="data/recap_entretien.csv",
        help="CSV de métadonnées des entretiens (Nom Fichier, Matériau, Artisanat)",
    )
    p_tools.add_argument(
        "--dict-output-dir", type=str, default="results/tool_comparaison", help="Dossier de sortie pour les dictionnaires"
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
    p_verb.add_argument("--input", type=str, required=True, help="CSV annoté d'entrée (séparateur ';')")
    p_verb.add_argument(
        "--mode",
        type=str,
        choices=["bert", "regression", "both"],
        default="bert",
        help="Mode: bert, regression ou both (défaut: bert)",
    )
    p_verb.add_argument(
        "--run",
        type=str,
        choices=["augment", "tune", "train", "predict", "full"],
        default="full",
        help="Action à effectuer (défaut: full)",
    )
    p_verb.add_argument(
        "--output-dir", type=str, default="results/verbalisation", help="Dossier de sortie pour predictions/rapports"
    )
    p_verb.add_argument("--augmented-csv", type=str, default="", help="Chemin du CSV déjà augmenté (si existe)")
    p_verb.add_argument("--do-augment", action="store_true", help="Effectuer l'augmentation avant les autres étapes")
    p_verb.add_argument(
        "--augment-output", type=str, default="data/annotation_augmented", help="Chemin de sortie pour l'augmentation"
    )
    p_verb.add_argument("--text-column", type=str, default="text", help="Colonne texte pour l'augmentation")
    p_verb.add_argument(
        "--augmenter-types", type=str, nargs="*", default=["contextual", "translation", "swap"], help="Types d'augmentation"
    )
    p_verb.add_argument("--num-aug", type=int, default=1, help="Nombre d'augmentations par type")
    p_verb.add_argument(
        "--bert-tune-out", type=str, default="results/bert_tuning_results.csv", help="Fichier de sortie du tuning BERT"
    )
    p_verb.add_argument("--bert-model-dir", type=str, default="models", help="Dossier modèle BERT")
    p_verb.add_argument(
        "--bert-model-name", type=str, default="distilbert-base-multilingual-cased", help="Nom du modèle Hugging Face pour BERT"
    )
    p_verb.add_argument(
        "--reg-tune-out", type=str, default="results/hyperparams_tuning.csv", help="Fichier de sortie du tuning regression"
    )
    p_verb.add_argument(
        "--reg-best-params",
        type=str,
        default="models/verbalisation_best_params.pkl",
        help="Chemin du pickle des meilleurs params regression",
    )
    p_verb.add_argument("--reg-model-dir", type=str, default="models", help="Dossier de sauvegarde pour le modèle de regression")
    p_verb.add_argument("--reports-dir", type=str, default="results/verbalisation", help="Dossier de sortie pour rapports HTML")
    p_verb.add_argument(
        "--interviews-csv", type=str, default="data/processed/cleaned_full.csv", help="CSV des entretiens pour rapports"
    )
    p_verb.add_argument(
        "--report-on", type=str, choices=["bert", "merged"], default="bert", help="Sur quel modèle générer le rapport"
    )
    p_verb.add_argument("--max-length", type=int, default=128, help="Max token length pour BERT predictor")
    p_verb.add_argument("--cv", type=int, default=5, help="Nombre de folds pour tuning regression")
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
