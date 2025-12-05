**EN COURS DE CONSTRUCTION**

# Projet Analyse d'entretien d'artisans

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)


Ceci est un projet réalisé dans le cadre de mes études qui a pour objectif de concevoir et expérimenter une méthode d’analyse automatique des entretiens d’artisans en utilisant des techniques classiques de comaparaisons et des techniques de nlp. Ce projet vise à identifier des indices linguistiques caractérisant la relation entre l'artisan et la matière.


## Contexte

Ce projet s'inscrit dans une exploration des relations entre **l'artisan** et **la matière**, en s'appuyant sur les travaux de recherche concernant les degrés d'intimité entre l'artisan et ses matériaux. L'objectif est de mieux comprendre ces relations afin de contribuer à la conception de nouveaux outils ou méthodes de formation.

Référence : https://theses.fr/s394689

---

## Axes de travail

### 1. Détection de la présence d'outils
- Construction d'un **dictionnaire d'outils** mentionnés par les artisans
- Organisation optionnelle **par métiers**
- Extraction automatique depuis Wikipedia

### 2. Détection de la difficulté de verbalisation
Repérage des moments où l'artisan exprime une difficulté à expliquer un geste ou un savoir-faire (indicateur d'intimité matière-artisan).

Techniques :
- Processus d'annotation manuelle sur corpus cible
- Augmentation de données (traduction aller/retour, reformulation)
- Fine-tuning BERT sur tâche de classification

Formulations détectées :
- "C'est difficile à expliquer"
- "Il faut le faire pour le comprendre"
- "C'est mieux si je vous montre"

### 3. Distance physique artisan / matière
Identification d'indices verbaux décrivant la proximité ou l'éloignement physique.

### 4. Temps d'attente
Repérage des passages évoquant des **temps d'attente sans intervention directe** (révélateurs de phases particulières).

### 5. État physique de la matière
Détection des termes décrivant transformations, propriétés et conditions (texture, température, réaction, etc.).

---

## Installation

### Prérequis
- **Python 3.10+** 
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

## Usage

Voir le [tutoriel](docs/tutorials/tutorial.md)

## Tests

Exécuter la suite de tests :
```bash
poetry run pytest -q
```

Exécuter les tests avec couverture de code :
```bash
poetry run pytest --cov=src --cov-report=html
```

Fichiers de test disponibles :
- `test_text_cleaning.py` — Tests du nettoyage de texte
- `test_csv_highlight.py` — Tests de mise en évidence d'outils dans CSV
- `test_merge_predictions.py` — Tests de fusion de prédictions
- `test_bert.py` — Tests du pipeline BERT



## Logging

Le projet utilise une **configuration centralisée du logging** via `src/utils/logger_config.py`.

### Fichiers de log

- `logs/bert_artisan.log` — Logs détaillés (INFO et supérieur, rotation)
- `logs/errors.log` — Erreurs uniquement (ERROR et supérieur)
- Console — INFO seulement


## Configuration

### Poetry

Extrait important du `pyproject.toml` :

    [tool.poetry]
    name = "bert-artisan-nlp"
    version = "0.3.0"
    python = "^3.10"

    [tool.poetry.dependencies]
    transformers = "^4.35.0"
    torch = "^2.2.0"
    pandas = "^2.1.0"
    scikit-learn = "^1.4.0"

### Black (formatage)

    [tool.black]
    line-length = 130
    target-version = ['py310', 'py311']

### isort (organisation des imports)

    [tool.isort]
    profile = "black"
    line_length = 130
    src_paths = ["src"]

### Pytest

    [tool.pytest.ini_options]
    testpaths = ["tests"]
    addopts = "--cov=src --cov-report=html"

### Pylint

La configuration est fournie dans le [.pylintrc](.pylintrc)



## License

Ce projet est sous licence **MIT**. Voir `LICENCE.md` pour les détails.

## Auteur

Aymeric Eyer  
https://github.com/Aymeric-e

