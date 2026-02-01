# Tutoriel Complet - Analyse d'entretiens d'Artisans

## Table des matières
1. [Installation](#installation)
2. [Workflow complet pas à pas](#workflow-complet-pas-à-pas)
3. [Phase 0 : Préparation des données](#phase-0--préparation-des-données)
4. [Phase 1 : Détection des outils](#phase-1--détection-des-outils)
5. [Phase 2 : Difficulté de verbalisation](#phase-2--difficulté-de-verbalisation)
6. [Phase 3 : Intimité multi-facteurs](#phase-3--intimité-multi-facteurs)
7. [Exemples d'utilisation](#exemples-dutilisation)
8. [Dépannage](#dépannage)

---

## Installation

### Prérequis
- **Python 3.10+** 
- **Git** pour cloner le dépôt
- **Poetry** pour la gestion des dépendances

### Installation locale

```bash
# Cloner le dépôt
git clone https://github.com/Aymeric-e/Analyse-d-entretiens-d-artisans.git
cd Analyse-d-entretiens-d-artisans

# Installer les dépendances
poetry install

```

### Vérifier l'installation
```bash
# Vérifier que les modules importent correctement
python -c "from preprocessing.text_cleaning import InterviewCleaner; print(' Installation OK')"
```

---

## Phase 0 : Préparation des Données

### Étape 1 : Nettoyage des fichiers .docx

Le module `InterviewCleaner` extrait et nettoie les entretiens au format `.docx`.

#### Structure attendue du fichier .docx
```
[Chercheur] Introduction...
[Interviewé] Réponse de l'artisan...
[Chercheur] Nouvelle question...
[Interviewé] Nouvelle réponse...
```

#### Utilisation
```python
from pathlib import Path
from preprocessing.text_cleaning import InterviewCleaner

# Créer une instance du nettoyeur
cleaner = InterviewCleaner()

# Traiter les fichiers (trois niveaux de segmentation possibles)
# Mode 'full' : entretien complet = 1 ligne
df_full = cleaner.batch_process(
    input_dir=Path('data/raw'),
    output_path=Path('data/processed/cleaned_full.csv'),
    mode='full'
)

# Mode 'paragraph' : paragraphe = 1 ligne (recommandé pour la plupart des analyses)
df_paragraph = cleaner.batch_process(
    input_dir=Path('data/raw'),
    output_path=Path('data/processed/cleaned_paragraph.csv'),
    mode='paragraph'
)

# Mode 'sentence' : phrase = 1 ligne (plus granulaire)
df_sentence = cleaner.batch_process(
    input_dir=Path('data/raw'),
    output_path=Path('data/processed/cleaned_sentence.csv'),
    mode='sentence'
)
```

#### Résultat
Fichiers CSV avec colonnes :
- `filename` : nom du fichier source
- `text` : texte nettoyé
- `word_count` : nombre de mots

**Exemple d'exécution en ligne de commande :**
```bash
# Nettoyage via CLI (exécuter depuis la racine du projet)
python src/preprocessing/text_cleaning.py --input data/raw --output data/processed --mode all

# Modes individuels :
python src/preprocessing/text_cleaning.py --input data/raw --output data/processed --mode paragraph
python src/preprocessing/text_cleaning.py --input data/raw --output data/processed --mode sentence
python src/preprocessing/text_cleaning.py --input data/raw --output data/processed --mode full
```

---

## Phase 1 : Détection des Outils

### Étape 1 : Extraction de la liste d'outils depuis Wikipedia

```python
from pathlib import Path
from preprocessing.extract_tool_wiki import extract_tools

# Extraire automatiquement la liste des outils depuis Wikipedia
output_dir = Path('data/tool_detection')
extract_tools(output_dir)

# Crée un fichier : data/tool_detection/list_tool_wiki.csv
# Contenant tous les outils de la page Wikipedia "Liste d'outils"
```

Ou en ligne de commande :
```bash
python src/preprocessing/extract_tool_wiki.py --output data/tool_detection
```


### Étape 2 : Détection des outils dans les textes

```python
from pathlib import Path
from processing.tool_detection_word_comparaison_strict import process_directory

# Détecter les outils dans les fichiers CSV nettoyés
process_directory(
    tool_csv=Path('data/tool_detection/list_tool_wiki.csv'),
    input_dir=Path('data/processed'),  # Dossier contenant cleaned_*.csv
    output_dir=Path('data/tool_detection')
)

# Résultats générés :
# - <filename>_with_tools.csv : textes avec colonne 'tools_found'
# - <filename>_tool_dict.csv : dictionnaire des outils par fichier
```

Ou en ligne de commande :
```bash
python src/processing/tool_detection_word_comparaison_strict.py \
  --tool-csv data/tool_detection/list_tool_wiki.csv \
  --input-dir data/processed \
  --output-dir data/tool_detection

# Optionnel : traiter uniquement certains fichiers
python src/processing/tool_detection_word_comparaison_strict.py --input-dir data/processed --csv-list cleaned_paragraph.csv cleaned_sentence.csv
```


### Étape 3 : Visualisation (optionnel)

```python
from pathlib import Path
from utils.csv_to_html_highlight_tool import highlight_tools_in_csv

# Générer des fichiers HTML avec surlignage des outils (Python API)
highlight_tools_in_csv(
    input_csv=Path('data/tool_detection/cleaned_paragraph_with_tools.csv'),
    output_html=Path('results/tool_detection/highlight.html')
)
```

Ou en ligne de commande :
```bash
python src/utils/csv_to_html_highlight_tool.py --input data/tool_detection/cleaned_paragraph_with_tools.csv --output-dir results/tool_detection
```


**Exemple complet (CLI) :**
```bash
# Extraire la liste depuis Wikipedia
python src/preprocessing/extract_tool_wiki.py --output data/tool_detection

# Détecter les outils dans les CSV nettoyés
python src/processing/tool_detection_word_comparaison_strict.py \
  --tool-csv data/tool_detection/list_tool_wiki.csv \
  --input-dir data/processed \
  --output-dir data/tool_detection

# Générer un HTML avec surlignage (optionnel)
python src/utils/csv_to_html_highlight_tool.py --input data/tool_detection/cleaned_paragraph_with_tools.csv --output-dir results/tool_detection
```

---

## Phase 2 : Difficulté de Verbalisation

Cette phase implémente une comparaison entre deux approches de modélisation :

1. **Ridge Regression** : basée sur TF-IDF et régularisation L2
2. **BERT Fine-tuned** : modèle de deep learning multilingual

### Étape 1 : Préparation des données annotées

Créer un fichier CSV annoté avec les colonnes suivantes :
```
text;difficulté_verbalisation
"c'est difficile à expliquer";1.0
"il faut le faire pour le comprendre";0.8
...
```

### Étape 2 : Augmentation des données (optionnel mais recommandé)

```python
from pathlib import Path
from preprocessing.text_augmentation import augment_csv

# Augmenter les données avec 3 techniques :
# - contextual : substitutions contextuelles (CamemBERT)
# - translation : back-translation (FR→DE→FR)
# - swap : permutation aléatoire de mots

augment_csv(
    input_path=Path('data/annotation/sentences_annotated_verb.csv'),
    output_path=Path('data/annotation/sentences_annotated_verb_augmented.csv'),
    text_column='text',
    augmenter_types=['contextual', 'translation', 'swap'],
    num_aug=1  # 1 augmentation par type par ligne
)

# Résultat : CSV avec ~4x plus de lignes (original + 3 augmentations)
```

Ou en ligne de commande :
```bash
python src/preprocessing/text_augmentation.py \
  --input data/annotation/sentences_annotated_verb.csv \
  --output data/annotation/sentences_annotated_verb_augmented.csv \
  --augmenter-types contextual translation swap \
  --num_aug 1
```


### Étape 3 : Pipeline BERT complet (Tuning → Train → Predict)

Utiliser le script `scripts/multiple_bert_models.py` pour automatiser le pipeline :

```bash
python scripts/multiple_bert_models.py \
  --annotated-csv data/annotation/sentences_annotated_verb_augmented.csv \
  --predict-csv data/processed/cleaned_sentence.csv \
  --output-csv results/verbalisation/bert_predictions.csv
```

Ou exécuter les étapes individuellement en CLI :
```bash
# 1) Tuning
python src/processing/tune_bert.py --input data/annotation/sentences_annotated_verb_augmented.csv --output data/verbalisation/bert_tuning_results.csv

# 2) Train
python src/processing/train_bert.py --input data/annotation/sentences_annotated_verb_augmented.csv --tuning-results data/verbalisation/bert_tuning_results.csv --model-dir models

# 3) Predict
python src/processing/predict_bert.py --input data/processed/cleaned_sentence.csv --model-dir models --output results/verbalisation/bert_predictions.csv
```

**Ou en Python :**

```python
from pathlib import Path
from processing.tune_bert import BertHyperparameterTuner
from processing.train_bert import BertFinalTrainer
from processing.predict_bert import BertPredictor
import pandas as pd

score_col = 'difficulté_verbalisation'

# 1) Tuning des hyperparamètres
print("1. Tuning BERT...")
tuner = BertHyperparameterTuner(
    model_name='distilbert-base-multilingual-cased',
    score_col=score_col,
    score_scale=10.0
)
tuning_results = Path('data/verbalisation/bert_tuning_results.csv')
tuner.run(
    csv_path=Path('data/annotation/sentences_annotated_verb_augmented.csv'),
    output_path=tuning_results
)

# 2) Entraînement avec les meilleurs hyperparamètres
print("2. Entraînement BERT...")
trainer = BertFinalTrainer(
    model_name='distilbert-base-multilingual-cased',
    model_dir=Path('models'),
    score_col=score_col,
    score_scale=10.0
)
trainer.run(
    csv_path=Path('data/annotation/sentences_annotated_verb_augmented.csv'),
    tuning_results_path=tuning_results
)

# 3) Prédictions
print("3. Prédictions BERT...")
predictor = BertPredictor(
    model_dir=Path('models'),
    score_col=score_col,
    score_scale=10.0
)

# Charger les données à prédire
df_pred = pd.read_csv('data/processed/cleaned_sentence.csv')
predictions = predictor.predict(df_pred['text'].astype(str))

# Sauvegarder les résultats
df_pred['difficulté_verbalisation_bert'] = predictions
df_pred.to_csv('results/verbalisation/bert_predictions.csv', index=False)

print(" Pipeline BERT terminé")
```

### Étape 4 : Pipeline Ridge Regression (alternatif)

```python
from pathlib import Path
from processing.train_regression import RegressionTrainer
from processing.predict_regression import RegressionPredictor
import pandas as pd

# Entraînement Ridge
trainer = RegressionTrainer(score_col='difficulté_verbalisation')
trainer.run(
    csv_path=Path('data/annotation/sentences_annotated_verb_augmented.csv'),
    model_save_path=Path('models/verbalisation/ridge_final')
)

# Prédictions Ridge
predictor = RegressionPredictor(model_path=Path('models/verbalisation/ridge_final'))
df_pred = pd.read_csv('data/processed/cleaned_sentence.csv')
predictions = predictor.predict(df_pred['text'].astype(str))

df_pred['difficulté_verbalisation_ridge'] = predictions
df_pred.to_csv('results/verbalisation/ridge_predictions.csv', index=False)
```

---

## Phase 3 : Intimité Multi-Facteurs

Cette phase entraîne 7 modèles BERT indépendants, un pour chaque dimension d'intimité :

1. **Fertilité du langage** : richesse vocabulaire et diversité syntaxique
2. **Fluidité** : continuité et fluidité de l'expression
3. **État physique** : description des transformations matérielles
4. **Distance physique** : proxémique artisan-matière
5. **Temps d'attente** : phases d'attente sans intervention
6. **Vulnerability** : indices de vulnérabilité
7. **Imaginaire** : usage d'images et métaphores

### Préparation des données annotées

Créer un fichier CSV avec les 7 colonnes de scores :
```
text;fertilité;fluidité;état_physique;distance;temps_attente;vulnerability;imaginaire
"le bois devient vraiment fluide...";0.8;0.9;0.7;0.5;0.3;0.2;0.6
...
```

### Exécution du pipeline multi-modèles

```bash
python scripts/multiple_bert_models.py \
  --annotated-csv data/annotation/intimité_augmented.csv \
  --predict-csv data/processed/cleaned_paragraph.csv \
  --output-csv results/intimité/all_scores.csv \
  --columns fertilité fluidité état_physique distance temps_attente vulnerability imaginaire \
  --n-scores 7
```

**Ou en Python :**
```python
from pathlib import Path
from processing.tune_bert import BertHyperparameterTuner
from processing.train_bert import BertFinalTrainer
from processing.predict_bert import BertPredictor
import pandas as pd

# Définir les colonnes à traiter
columns = ['fertilité', 'fluidité', 'état_physique', 'distance', 'temps_attente', 'vulnerability', 'imaginaire']

# Charger données prédiction (récurrent)
df_pred = pd.read_csv('data/processed/cleaned_paragraph.csv')

# Pour chaque colonne, run tuning → train → predict
for col in columns:
    print(f"\n Traitement de : {col} ")
    
    # 1) Tuning
    tuner = BertHyperparameterTuner(score_col=col, score_scale=10.0)
    tuning_file = Path(f'data/verbalisation/{col}_tuning.csv')
    tuner.run(Path('data/annotation/intimité_augmented.csv'), tuning_file)
    
    # 2) Train
    trainer = BertFinalTrainer(model_dir=Path('models'), score_col=col)
    trainer.run(Path('data/annotation/intimité_augmented.csv'), tuning_file)
    
    # 3) Predict
    predictor = BertPredictor(model_dir=Path('models'), score_col=col)
    preds = predictor.predict(df_pred['text'].astype(str))
    
    col_name = f'note_bert_{col}'
    df_pred[col_name] = preds
    print(f" {col} prédite (colonne: {col_name})")

# Sauvegarder tous les résultats
df_pred.to_csv('results/intimité/all_scores.csv', index=False)
print("\n Toutes les prédictions sauvegardées")
```

---

## Exemples d'Utilisation

### Exemple 1 : Pipeline complet (Phase 0 + 1 + 2)

```bash
# Script bash complet
#!/bin/bash

PROJECT_DIR="/usr/bert_artisan_nlp"
cd "$PROJECT_DIR"

echo " PIPELINE COMPLET "

# Phase 0 : Nettoyage
echo "Phase 0 : Nettoyage des .docx..."
python -c "
from pathlib import Path
from preprocessing.text_cleaning import InterviewCleaner
cleaner = InterviewCleaner()
cleaner.batch_process(Path('data/raw'), Path('data/processed/cleaned_full.csv'), mode='full')
cleaner.batch_process(Path('data/raw'), Path('data/processed/cleaned_paragraph.csv'), mode='paragraph')
cleaner.batch_process(Path('data/raw'), Path('data/processed/cleaned_sentence.csv'), mode='sentence')
print(' Phase 0 terminée')
"

# Phase 1 : Détection outils
echo "Phase 1 : Détection des outils..."
python -c "
from pathlib import Path
from preprocessing.extract_tool_wiki import extract_tools
from processing.tool_detection_word_comparaison_strict import process_directory
extract_tools(Path('data/tool_detection'))
process_directory(
    Path('data/tool_detection/list_tool_wiki.csv'),
    Path('data/processed'),
    Path('data/tool_detection')
)
print(' Phase 1 terminée')
"

# Phase 2 : Verbalisation
echo "Phase 2 : Verbalisation BERT..."
python scripts/multiple_bert_models.py \
  --annotated-csv data/annotation/sentences_annotated_verb_augmented.csv \
  --predict-csv data/processed/cleaned_sentence.csv \
  --output-csv results/verbalisation/predictions.csv

echo " Pipeline complet terminé"
```

### Exemple 2 : Focus sur une seule phase

```python
# Script Python pour tester une phase spécifique
from pathlib import Path
from preprocessing.extract_tool_wiki import extract_tools
from processing.tool_detection_word_comparaison_strict import process_directory

# Télécharger et détecter les outils uniquement
extract_tools(Path('data/tool_detection'))
process_directory(
    Path('data/tool_detection/list_tool_wiki.csv'),
    Path('data/processed'),
    Path('data/tool_detection')
)

# Vérifier les résultats
import pandas as pd
results = pd.read_csv('data/tool_detection/cleaned_paragraph_tool_dict.csv')
print(f"Total outils détectés: {len(results)}")
print(results.head())
```

### Exemple 3 : Personnalisation des hyperparamètres

```python
from pathlib import Path
from processing.tune_bert import BertHyperparameterTuner

# Tuning avec modèle différent
tuner = BertHyperparameterTuner(
    model_name='bert-base-multilingual-cased',  # Plus lourd mais potentiellement meilleur
    score_col='difficulté_verbalisation',
    score_scale=10.0
)

tuner.run(
    Path('data/annotation/sentences_annotated_verb_augmented.csv'),
    Path('data/verbalisation/bert_tuning_results.csv')
)

# Afficher les meilleurs hyperparamètres
import pandas as pd
results = pd.read_csv('data/verbalisation/bert_tuning_results.csv', sep=';')
best = results.loc[results['r2'].idxmax()]
print("Meilleurs hyperparamètres:")
print(f"  LR: {best['learning_rate']}")
print(f"  Batch: {best['batch_size']}")
print(f"  Max Length: {best['max_length']}")
print(f"  Epochs: {best['num_epochs']}")
print(f"  R²: {best['r2']:.4f}")
```

---

## Dépannage

### Issue 1 : "ModuleNotFoundError: No module named 'preprocessing'"

**Solution :**
```bash
# Vérifier que vous êtes dans le bon répertoire
cd Analyse-d-entretiens-d-artisans

# Vérifier que Poetry a bien installé les packages
poetry install

# Si le problème persiste, régénérer l'environnement
poetry env remove <env_name>
poetry install
```

### Issue 2 : CUDA Out of Memory

**Solution :**
```python
# Réduire la taille du batch
tuner = BertHyperparameterTuner(...)
# Les hyperparamètres incluent batch_size
# ou modifier dans le tuning

# Alternative : utiliser CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

### Issue 3 : Fichiers Wikipedia non téléchargés

**Solution :**
```python
from preprocessing.extract_tool_wiki import extract_tools
from pathlib import Path

# Forcer le téléchargement avec retry
try:
    extract_tools(Path('data/tool_detection'))
except Exception as e:
    print(f"Erreur: {e}")
    print("Vérifier la connexion internet et réessayer")
```

### Issue 4 : Logs trop verbeux

**Solution :**
```python
# Configurer le niveau de log
from utils.logger_config import setup_logger

logger = setup_logger(__name__, level="WARNING")  # Au lieu de INFO
```

### Issue 5 : Modèle pas trouvé après entraînement

**Vérifier :**
```bash
# Vérifier la structure des dossiers
find models/ -type d -name "bert_final"

# Les modèles doivent être dans :
# models/<score_col>/bert_final/model/
# models/<score_col>/bert_final/tokenizer/
```

### Issue 6 : Scripts CLI et imports

Les scripts supportent des arguments en ligne de commande 

- Pour obtenir la liste des arguments :
```bash
python src/preprocessing/text_cleaning.py --help
```

---

## Ressources Supplémentaires

- [Documentation Hugging Face Transformers](https://huggingface.co/transformers/)
- [Documentation NLTK](https://www.nltk.org/)
- [Documentation scikit-learn](https://scikit-learn.org/)
- [NLPaug pour la data augmentation](https://github.com/makcedward/nlpaug)

## Support

Pour toute question ou problème, consultez les fichiers de log :
```bash
# Logs détaillés
tail -f logs/bert_artisan.log

# Erreurs uniquement
tail -f logs/errors.log
```
