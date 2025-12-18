# Tutoriel Complet - Analyse d'entretiens des artisans

## Table des matières
1. [Installation](#installation)
2. [Commandes disponibles](#commandes-disponibles)
3. [Workflow complet pas à pas](#workflow-complet-pas-à-pas)
4. [Exemples d'utilisation](#exemples-dutilisation)
5. [Dépannage](#dépannage)

---

## Installation

### Prérequis
- **Python 3.10+** 
- **Git** pour cloner le dépôt
- **Poetry** pour la gestion des dépendances et de l'environnement

### Installation locale

1. Cloner le dépôt :
```bash
git clone https://github.com/Aymeric-e/Analyse-d-entretiens-d-artisans.git

cd Analyse-d-entretiens-d-artisans
```

2. Installer les dépendances :
```bash
poetry install
```

### Vérifier l'installation
```bash
python scripts/bert_artisan_cli.py --help
```

Vous devriez voir la liste de toutes les commandes disponibles.

---

## Commandes disponibles

### 1. `clean` - Nettoyage des fichiers .docx

Transforme les fichiers Word (.docx) en fichiers CSV nettoyés.

**Syntaxe :**
```bash
python scripts/bert_artisan_cli.py clean \
  --input data/raw \
  --output data/processed \
  --mode paragraph
```

**Arguments :**
- `--input` : Dossier contenant les fichiers .docx (défaut: `data/raw`)
- `--output` : Dossier de sortie pour les CSV nettoyés (défaut: `data/processed`)
- `--mode` : Mode de segmentation
  - `paragraph` : Segmente par paragraphes (recommandé)
  - `sentence` : Segmente par phrases
  - `full` : Pas de segmentation (garde le texte entier)

**Résultat :**
- `cleaned_paragraph.csv` : Un paragraphe = une ligne
- `cleaned_sentence.csv` : Une phrase = une ligne
- `cleaned_full.csv` : Un document = une ligne

**Exemple :**
```bash
python scripts/bert_artisan_cli.py clean --input data/raw --output data/processed --mode sentence
```

---

### 2. `tools` - Détection des outils

Pipeline complet pour détecter les outils mentionnés dans les textes par comparaison avec la liste Wikipedia.

Ce processus produit deux zones de sortie :
- `--tool-output-dir` (par défaut `data/tool_detection`) : contient tous les CSV intermédiaires générés (liste Wikipedia, `*_with_tools.csv`, `*_tool_dict.csv`, et `dict_outils_*.csv` si générés).
- `--results-dir` (par défaut `results/tool_detection`) : contient les résultats finaux organisés sous `csv/` (les CSV listés plus haut) et `html/` (versions HTML des CSV et fichiers highlight pour les fichiers `*_with_tools.csv`).

**Syntaxe simple (détection uniquement) :**
```bash
python scripts/bert_artisan_cli.py tools \
  --processed-dir data/processed \
  --tool-output-dir data/tool_detection \
  --results-dir results/tool_detection
```

**Syntaxe complète (avec tous les optionnels) :**
```bash
python scripts/bert_artisan_cli.py tools \
  --processed-dir data/processed \
  --tool-output-dir data/tool_detection \
  --results-dir results/tool_detection \
  --generate-html \
  --build-dicts \
  --recap-entretien data/recap_entretien.csv
```

**Étapes exécutées automatiquement :**
1. Extraction de la liste Wikipedia (dans `--tool-output-dir/list_tool_wiki.csv`) (si absent ou forcée)
2. Détection par comparaison stricte sur les CSV nettoyés (écrit `*_with_tools.csv` et `*_tool_dict.csv` dans `--tool-output-dir`)
3. Copie des fichiers utiles dans `--results-dir/csv`
4. (optionnel) Conversion CSV→HTML dans `--results-dir/html/csv` et génération des highlight HTML (avec surlignage des outils) dans `--results-dir/html/highlight` pour chaque `*_with_tools.csv` (si `--generate-html`)
5. (optionnel) Création de dictionnaires par artisanat/matériau (si `--build-dicts` and `--recap-entretien` fourni). Les dictionnaires sont copiés dans `--results-dir/csv`.

**Arguments :**
- `--processed-dir` : Dossier contenant les CSV nettoyés (défaut: `data/processed`)
- `--tool-output-dir` : Dossier pour tous les CSV produits (défaut: `data/tool_detection`)
- `--results-dir` : Dossier final de résultats (contient `csv/` et `html/`) (défaut: `results/tool_detection`)
- `--generate-html` : (optionnel) Générer les fichiers HTML (CSV→HTML et highlights)
- `--build-dicts` : (optionnel) Créer dictionnaires par artisanat/matériau (nécessite `--recap-entretien`)
- `--recap-entretien` : CSV de métadonnées (Nom Fichier, Matériau, Artisanat) (défaut: `data/recap_entretien.csv`)
- `--force-extract` : Force le re-téléchargement de Wikipedia même si le fichier existe

**Résultat (minimal) :**
- `--tool-output-dir/` : CSV intermédiaires (`list_tool_wiki.csv`, `*_with_tools.csv`, `*_tool_dict.csv`)
- `--results-dir/csv/` : Copie des fichiers utiles

**Résultat (avec --generate-html) :**
- `--results-dir/html/csv/` : versions HTML des CSV
- `--results-dir/html/highlight/` : versions HTML avec texte surligné pour chaque `*_with_tools.csv`

**Exemples :**
```bash
# Détection simple
python scripts/bert_artisan_cli.py tools --processed-dir data/processed

# Avec HTML et dictionnaires (nécessite data/recap_entretien.csv)
python scripts/bert_artisan_cli.py tools --processed-dir data/processed --generate-html --build-dicts

# Forcer la réextraction de Wikipedia
python scripts/bert_artisan_cli.py tools --force-extract
```

---

### 3. `augment` - Augmentation des données

Augmente un dataset annoté pour améliorer l'entraînement des modèles.

**Syntaxe :**
```bash
python scripts/bert_artisan_cli.py augment \
  --input data/annotation/sentences_annotated.csv \
  --output data/annotation_augmented
```

**Arguments :**
- `--input` : Fichier CSV annoté d'entrée (OBLIGATOIRE)
- `--output` : Dossier ou fichier de sortie (OBLIGATOIRE)
- `--text-column` : Nom de la colonne texte (défaut: `text`)
- `--augmenter-types` : Types d'augmentation
  - `contextual` : Substitution contextuelle
  - `translation` : Back-translation
  - `swap` : Permutation de mots
- `--num-aug` : Nombre d'augmentations par type (défaut: 1)

**Résultat :**
- Dataset augmenté (*nb_augmenter-types\*num-aug* plus gros que l'original)

**Exemple :**
```bash
python scripts/bert_artisan_cli.py augment \
  --input data/annotation/sentences_annotated.csv \
  --output data/annotation_augmented \
  --num-aug 2 \
  --augmenter-types contextual translation
```

---

### 4. `verbalisation` - Pipeline de difficulté de verbalisation

Workflow complet pour l'entraînement, le tuning, la prédiction et la génération de rapports.

Principes clés :
- Le fichier d'entrée annoté (par défaut `data/annotation/sentences_annoted_verb.csv`) est utilisé pour l'augmentation/tuning/entraînement.
- Les prédictions sont exécutées sur le fichier de *phrases* nettoyées (par défaut `data/processed/cleaned_sentence.csv`).
- Les hyperparamètres et fichiers de prédiction sont écrits dans `--verb-data` (défaut `data/verbalisation`).
- Les résultats finaux et HTML sont copiés dans `--results-dir` (défaut `results/verbalisation`) avec sous-dossiers `csv/` et `html/`.

**Syntaxe :**
```bash
python scripts/bert_artisan_cli.py verbalisation \
  --input data/annotation/sentences_annoted_verb.csv \
  --mode bert \
  --run full \
  --verb-data data/verbalisation \
  --results-dir results/verbalisation
```

**Arguments principaux :**
- `--input` : CSV annoté d'entrée (défaut: `data/annotation/sentences_annoted_verb.csv`)
- `--augmented-csv` : CSV annoté augmenté (défaut: `data/annotation/sentences_annoted_verb_augmented.csv`)
- `--clean-sentences` : Fichier des phrases (prédiction) (défaut: `data/processed/cleaned_sentence.csv`)
- `--clean-full` : Fichier des entretiens complets (pour rapports) (défaut: `data/processed/cleaned_full.csv`)
- `--mode` : `bert` | `reg` | `both` (défaut: `bert`)
- `--run` : `augment` | `tune` | `train` | `predict` | `full` (défaut: `full`)
- `--verb-data` : Dossier pour hyperparams & prédictions (défaut: `data/verbalisation`)
- `--results-dir` : Dossier final de résultats (défaut: `results/verbalisation`)

**Options importantes :**
- `--text-column` : Colonne texte (défaut: `text`)
- `--cv` : folds CV pour tuning (défaut: 5)
- `--max-length` : Longueur tokens BERT (défaut: 128)
- `--num-aug` : Nombre d'augment par type (défaut: 1)
- `--augmenter-types` : Types d'augmentation (défaut: contextual translation swap)
- `--model-reg-dir` : Dossier modèles regression (défaut: `models/verbalisation/regression`)
- `--model-bert-dir` : Dossier modèles BERT (défaut: `models/verbalisation/bert`)

**Résultat attendu :**
- `--verb-data/` : hyperparams CSV et fichiers de prédictions (`verbalisation_bert_predictions.csv`, `verbalisation_reg_predictions.csv`, `verbalisation_merged.csv`)
- `--results-dir/csv/` : copies des CSV de prédiction
- `--results-dir/html/csv/` : versions HTML des CSV
- `--results-dir/html/reports/` : deux versions de rapports HTML (`verbalisation_report_v1.html` et `_v2.html`) générées par les deux programmes de rapport

**Exemple - Pipeline complet avec BERT :**
```bash
python scripts/bert_artisan_cli.py verbalisation \
  --mode bert \
  --run full \
  --verb-data data/verbalisation \
  --results-dir results/verbalisation
```
**Exemple - Pipeline complet avec BERT :**
```bash
python scripts/bert_artisan_cli.py verbalisation \
  --input data/annotation/sentences_annotated.csv \
  --mode bert \
  --run full \
  --output-dir results/verbalisation
```

**Exemple - Pipeline avec augmentation :**
```bash
python scripts/bert_artisan_cli.py verbalisation \
  --input data/annotation/sentences_annotated.csv \
  --mode regression \
  --run full \
  --do-augment \
  --augment-output data/augmented \
  --num-aug 2
```

**Exemple - Seulement tuning :**
```bash
python scripts/bert_artisan_cli.py verbalisation \
  --input data/annotation/sentences_annotated.csv \
  --mode both \
  --run tune \
  --cv 10
```

---

## Workflow complet pas à pas

### Scénario : Analyser des entretiens d'artisans

#### Étape 1 : Préparation des données
1. Placez vos fichiers `.docx` dans `data/raw/`
2. (Optionnel) Si vous voulez des dictionnaires par catégorie : préparez `data/recap_entretien.csv` avec les colonnes :
   - `Nom Fichier` : Nom du fichier .docx (ex: `entretien_01.docx`)
   - `Matériau` : Type de matériau (ex: `Bois`)
   - `Artisanat` : Type d'artisanat (ex: `Menuiserie`)

#### Étape 2 : Nettoyage
```bash
python scripts/bert_artisan_cli.py clean \
  --input data/raw \
  --output data/processed \
  --mode sentence
```

Cela crée `data/processed/cleaned_sentence.csv` avec une phrase par ligne.

#### Étape 3 : Détection des outils (minimal)
```bash
python scripts/bert_artisan_cli.py tools \
  --processed-dir data/processed \
  --tool-output-dir data/tool_detection \
  --results-dir results/tool_detection
```

Cela crée :
- `data/tool_detection/` : CSV intermédiaires (par ex. `list_tool_wiki.csv`, `*_with_tools.csv`, `*_tool_dict.csv`)
- `results/tool_detection/csv/` : copies des CSV utiles
- `results/tool_detection/html/` : versions HTML (si `--generate-html`)

#### Étape 3b : Détection des outils (avec options avancées)
```bash
python scripts/bert_artisan_cli.py tools \
  --processed-dir data/processed \
  --tool-output-dir data/tool_detection \
  --results-dir results/tool_detection \
  --generate-html \
  --build-dicts \
  --recap-entretien data/recap_entretien.csv
```

Cela crée en plus :
- `results/tool_detection/html/csv/` : Visualisations HTML pour les CSV de résultats
- `results/tool_detection/html/highlight/` : fichiers highlight (texte avec outils surlignés) pour chaque `*_with_tools.csv`
- `results/tool_detection/csv/` : dictionnaires par catégorie (si `--build-dicts` et `recap_entretien.csv` fourni)

#### Étape 4 : Annotation (MANUEL)
Vous devez manuellement annoter le fichier CSV pour indiquer la difficulté de verbalisation :
- Ouvrez `data/processed/cleaned_sentence.csv`
- Ajoutez une colonne `difficulte_verbalisation` avec des scores (0-10)
- Sauvegardez sous `data/annotation/sentences_annotated.csv`

#### Étape 5 : Pipeline verbalisation complet
```bash
python scripts/bert_artisan_cli.py verbalisation \
  --input data/annotation/sentences_annotated.csv \
  --mode both \
  --run full \
  --do-augment \
  --num-aug 1
```

Cela :
1. Augmente les données (x3)
2. Tune les hyperparamètres
3. Entraîne BERT et Regression
4. Fait des prédictions
5. Fusionne les résultats
6. Génère un rapport HTML

#### Étape 6 : Consultation des résultats
- Rapport : `results/verbalisation/verbalisation_report.html` (ouvrir dans un navigateur)
- Prédictions détaillées : `results/verbalisation/verbalisation_merged.csv`
- Logs : `logs/` (pour déboguer si erreurs)

---

## Exemples d'utilisation

### Exemple 1 : Quick Start (5 minutes)
```bash
# Supposant les .docx sont dans data/raw/
python scripts/bert_artisan_cli.py clean --input data/raw --mode paragraph

python scripts/bert_artisan_cli.py tools --processed-dir data/processed --tool-output-dir data/tool_detection --results-dir results/tool_detection

# Ouvrez les résultats générés
```

### Exemple 2 : Détection simple des outils
```bash
# Juste extraire les outils sans rien d'autre
python scripts/bert_artisan_cli.py tools \
  --processed-dir data/processed \
  --tool-output-dir data/tool_detection \
  --results-dir results/tool_detection
```

**Résultat :** CSV avec colonne `tools_detected` remplie

### Exemple 3 : Détection complète avec visualisations
```bash
# Outils + HTML + dictionnaires
python scripts/bert_artisan_cli.py tools \
  --processed-dir data/processed \
  --tool-output-dir data/tool_detection \
  --results-dir results/tool_detection \
  --generate-html \
  --build-dicts \
  --recap-entretien data/recap_entretien.csv
```

**Résultat :** CSV + HTML + dictionnaires par artisanat/matériau

### Exemple 4 : Entraîner un modèle de régression simple
```bash
# Supposant data/annotation/sentences_annotated.csv existe
python scripts/bert_artisan_cli.py verbalisation \
  --input data/annotation/sentences_annotated.csv \
  --mode regression \
  --run full
```

**Temps estimé :** 2-5 minutes

### Exemple 5 : Entraîner BERT avec augmentation
```bash
python scripts/bert_artisan_cli.py verbalisation \
  --input data/annotation/sentences_annotated.csv \
  --mode bert \
  --run full \
  --do-augment \
  --augment-output data/augmented \
  --num-aug 3 \
  --bert-model-name "distilbert-base-multilingual-cased"
```

**Temps estimé :** 30 minutes à 2 heures (dépend du GPU)

### Exemple 6 : Seulement faire des prédictions (modèle déjà entraîné)
```bash
python scripts/bert_artisan_cli.py verbalisation \
  --input data/annotation/new_sentences.csv \
  --mode bert \
  --run predict
```

**Temps estimé :** 5-10 minutes

### Exemple 7 : Comparaison BERT vs Regression
```bash
python scripts/bert_artisan_cli.py verbalisation \
  --input data/annotation/sentences_annotated.csv \
  --mode both \
  --run full \
  --do-augment \
  --num-aug 2 \
  --report-on merged
```

Génère un rapport qui compare les deux modèles.

### Exemple 8 : Pipeline personnalisé - seulement tuning
```bash
python scripts/bert_artisan_cli.py verbalisation \
  --input data/annotation/sentences_annotated.csv \
  --mode both \
  --run tune \
  --cv 10
```

Génère :
- `results/bert_tuning_results.csv`
- `results/hyperparams_tuning.csv`

---

## Dépannage

### Erreur : "Fichier introuvable"
**Message :** `Le dossier d'entrée n'existe pas`

**Solution :**
```bash
# Vérifiez le chemin
ls data/raw/

# Si vide, créez le dossier
mkdir -p data/raw
```

### Erreur : "ModuleNotFoundError: No module named 'src'"
**Message :** `ModuleNotFoundError: No module named 'src'`

**Solution :**
```bash
# Assurez-vous d'exécuter depuis la racine du projet
cd /chemin/vers/Analyse-entretiens-artisans
```

### Erreur : "Modèle BERT non trouvé"
**Message :** `ConnectionError: Unable to download model from Hugging Face`

**Solution :**
```bash
# Vérifiez votre connexion internet
ping huggingface.co

# Ou téléchargez le modèle manuellement :
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased')"
```

### Erreur : "Out of memory" (GPU/RAM)
**Message :** `CUDA out of memory` ou `MemoryError`

**Solutions :**
```bash
# Réduisez la taille des batches
python scripts/bert_artisan_cli.py verbalisation \
  --input data/annotation/sentences_annotated.csv \
  --mode bert \
  --run full \
  --max-length 64  # Au lieu de 128

# Ou utilisez regression au lieu de BERT
python scripts/bert_artisan_cli.py verbalisation \
  --input data/annotation/sentences_annotated.csv \
  --mode regression \
  --run full
```

### Erreur : "Colonne manquante dans CSV"
**Message :** `KeyError: 'colonne_attendue'`

**Solution :**
```bash
# Vérifiez les colonnes de votre CSV
head -1 data/annotation/sentences_annotated.csv

# Assurez-vous d'avoir les colonnes obligatoires :
# - 'text' ou autre nom spécifié avec --text-column
# - 'difficulte_verbalisation' (pour verbalisation)

# Renommez si nécessaire
python scripts/bert_artisan_cli.py verbalisation \
  --input data/annotation/sentences_annotated.csv \
  --text-column mon_nom_de_colonne
```

### Erreur : "Fichier .docx corrompu"
**Message :** `FileNotFoundError` ou erreur lors du nettoyage

**Solution :**
```bash
# Vérifiez le fichier
file data/raw/mon_fichier.docx

# Essayez de le convertir manuellement avec LibreOffice
libreoffice --headless --convert-to csv data/raw/mon_fichier.docx

# Ou recréez-le proprement dans Word
```

### Logs : Où voir ce qui se passe ?
```bash
# Regarder les logs en temps réel
tail -f logs/*.log

# Ou les logs de la dernière exécution
ls -ltrh logs/ | tail -1 | awk '{print $NF}' | xargs cat
```

---

## FAQ

### Q: Je n'ai pas d'annotations. Que faire ?
R: Vous devez manuellement annoter les phrases pour indiquer la difficulté de verbalisation. Ouvrez le CSV et ajoutez une colonne "difficulte_verbalisation" avec des valeurs 0-10 (0 = facile, 10 = difficile).

### Q: Combien d'annotations sont nécessaires ?
R: Au minimum 100-500 phrases annotées. Idéalement 1000+. Plus vous en avez, meilleur est le modèle.

### Q: Puis-je utiliser d'autres modèles que BERT ?
R: Actuellement non via le CLI. Mais vous pouvez modifier le code source dans `src/processing/`.

### Q: Puis-je paralléliser le nettoyage ?
R: Pas directement via le CLI, mais l'augmentation et les prédictions utilisent déjà le multi-processing.

### Q: Comment partager les modèles entraînés ?
R: Copier le répertoire `models/` et le dossier `results/` à quelqu'un d'autre.

### Q: Puis-je combiner plusieurs datasets annotés ?
R: Oui, fusionnez les CSV avant de lancer verbalisation.

### Q: Je dois vraiment avoir recap_entretien.csv ?
R: Non ! Les dictionnaires par artisanat/matériau sont optionnels. Vous pouvez juste faire de la détection d'outils sans ce fichier. Utilisez simplement `--build-dicts` seulement si vous avez ce fichier.

---

## Support 

Pour les erreurs ou améliorations, consultez :
- `logs/` pour voir les messages détaillés
- `src/` pour le code source
- `docs/` pour la documentation technique et le fichier de référence des modules (ouvrir index.html dans un navigateur)

