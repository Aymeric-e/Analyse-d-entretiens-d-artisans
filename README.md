# Projet Analyse Automatique d'Entretiens d'Artisans

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

Série d'outils d'analyse pour explorer la transmission de conaissance d'artisans de manière implicite à travers d'indices linguistiques présents dans les entretiens réalisés.

## Contexte

Ce projet s'inscrit dans un travail de recherche qui explore comment transmettre les connaissances et savoirs des artisans. L'objectif est d'identifier des indices linguistiques qui caractérisent cette transmission au travers d'outils d'analyses statistiques et d'outils NLP.

**Référence :** https://theses.fr/s394689

---

## Architecture du Projet

Ce projet implémente une série d'outil qu'on peut séparer en 4 phases :

### Phase 0 : Préparation des Données
- Extraction et nettoyage des fichiers `.docx` d'entretiens
- Segmentation par paragraphes, phrases ou entretiens complets
- Standardisation en fichiers CSV nettoyés

### Phase 1 : Détection des Outils
- Extraction automatique d'une liste d'outils depuis Wikipedia
- Détection par comparaison stricte avec les textes d'entretiens
- Génération de statistiques d'occurrence des outils
- Création optionnelle de dictionnaires par métier

### Phase 2 : Difficulté de Verbalisation
- Annotation manuelle des passages verbalement difficiles
- Augmentation de données (traduction, substitution contextuelle, permutation)
- Entraînement de deux modèles comparatifs :
  - **Ridge Regression** : approche basée sur TF-IDF
  - **BERT Fine-tuned** : approche deep learning (multilingual)
- Prédiction sur l'ensemble des entretiens

### Phase 3 : Analyse Multi-Facteurs d'Intimité (7 dimensions)
- **Fertilité du langage** : richesse vocabulaire et diversité syntaxique
- **Fluidité** : continuité et fluidité de l'expression
- **État physique de la matière** : description des transformations
- **Distance physique** : proxémique artisan-matière
- **Temps d'attente** : phases d'attente sans intervention
- **Vulnerability** : indices de vulnérabilité
- **Imaginaire** : usage d'images et métaphores

Entraînement de 7 modèles BERT indépendants + explications statistiques et avec SHAP

---

## Installation

### Prérequis
- **Python 3.10+** 
- **Poetry** pour la gestion des dépendances
- **Git** pour cloner le dépôt

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
python -c "from preprocessing.text_cleaning import InterviewCleaner; print('OK')"
```

---

## Utilisation

Voir le [tutoriel détaillé](docs/tutorials/tutorial.md).

---

## Tests

Exécuter la suite de tests :
```bash
poetry run pytest -q
```

Exécuter avec couverture de code :
```bash
poetry run pytest --cov=src --cov-report=html
```

---

## Logging

Le projet utilise une **configuration centralisée du logging** via [src/utils/logger_config.py](src/utils/logger_config.py).

### Fichiers de log générés
- `logs/bert_artisan.log` — Logs détaillés (INFO et supérieur, rotation)
- `logs/errors.log` — Erreurs uniquement (ERROR et supérieur)
- Console — INFO seulement

---

## Configuration

### Dépendances clés (voir [pyproject.toml](pyproject.toml))

```toml
[tool.poetry.dependencies]
python = "^3.10"
transformers = "^4.35.0"
torch = "^2.2.0"
pandas = "^2.1.0"
scikit-learn = "^1.4.0"
nlpaug = "^1.1.10"
nltk = "^3.8.0"
shap = "^0.44.0"
```

### Outils de qualité
- **Black** : formatage (line-length: 130)
- **isort** : organisation des imports
- **Pylint** : linting *Pour le moment désactiver car trop contraignant*
- **Pytest** : tests avec couverture

---

## Licence

Ce projet est sous licence **MIT**. Voir [LICENCE.md](LICENCE.md) pour les détails.

## Auteur

Aymeric Eyer  
https://github.com/Aymeric-e
