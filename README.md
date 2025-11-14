# Projet bert-artisan-nlp

Projet industriel de 3A ICM qui a pour objectif de concevoir et expérimenter une méthode d’analyse automatique des entretiens d’artisans, en s’appuyant sur BERT et  des  techniques  de  NLP  et  NER. L’objectif est d’extraire des thèmes et entités pertinentes pour qualifier la relation  artisan/matière.

---

## Prérequis

- Python 3.9+ (idéalement géré avec pyenv)
- Poetry pour gérer dépendances, environnement virtuel et build

---

## Installation des dépendances

```bash
poetry install
```

Active l'environnement virtuel :

```bash
poetry env activate
```

Ou lance une commande sans activer manuellement :

```bash
poetry run python scripts/1_clean.py
```

---

## Gestion des versions

La version est configurée dans `pyproject.toml` :

```toml
[tool.poetry]
version = "0.1.0"
```

Pour mettre à jour la version :

```bash
poetry version patch  # ou minor, major
```

---

## Compilation et packaging

Pour construire les packages installables (.whl et .tar.gz) :

```bash
poetry build
```

Les fichiers sont générés dans `dist/` :

- bert_artisan_nlp-0.1.0-py3-none-any.whl
- bert_artisan_nlp-0.1.0.tar.gz

Installation locale possible via :

```bash
pip install dist/bert_artisan_nlp-0.1.0-py3-none-any.whl
```

---

## Build complet réplicable

Pour cloner et utiliser ce projet reproduisible :

```bash
git clone https://github.com/Aymeric-e/Analyse-d-entretiens-d-artisans.git
cd bert-artisan-nlp
poetry install
poetry build
pip install dist/bert_artisan_nlp-0.1.0-py3-none-any.whl
```

---

## Lancement des scripts

```bash
poetry run python scripts/1_clean.py        # Nettoyage
poetry run python scripts/3_detect_tools.py # Détection d'outils
```

---