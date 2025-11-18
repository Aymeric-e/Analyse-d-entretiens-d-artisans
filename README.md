**EN COURS DE CONSTRUCTION**

# Projet bert-artisan-nlp

Ceci est un projet réalisé dans le cadre de mes études qui a pour objectif de concevoir et expérimenter une méthode d’analyse automatique des entretiens d’artisans, en s’appuyant sur BERT et  des  techniques  de  NLP  et  NER. L’objectif est d’extraire des thèmes et entités pertinentes pour qualifier la relation  artisan/matière.

---

## A faire : 

## Contexte du projet

Ce projet s’inscrit dans une exploration des relations entre **l’artisan** et **la matière**, en s’appuyant notamment sur les travaux décrits dans la thèse suivante : https://theses.fr/s394689  
L’objectif général est de mieux comprendre les différents **degrés d’intimité** entre l’artisan et la matière, afin de contribuer à la conception de nouveaux outils ou méthodes de **formation**.

Dans ce cadre, plusieurs types d’indices linguistiques ou conceptuels peuvent permettre de caractériser cette relation. Le projet vise donc à identifier, structurer et analyser ces indices dans des corpus d’entretiens avec des artisans.

---

## Axes de travail

### 1. Détection de la présence d’outils
- Construire un **dictionnaire d’outils** mentionnés par les artisans.  
- Possibilité d’organiser ce dictionnaire **par métiers**.

### 2. Détection de la difficulté de verbalisation

**Objectif** : repérer les moments où l’artisan exprime une difficulté à expliquer un geste ou un savoir-faire, ce qui constitue un indicateur important d’intimité matière-artisan.

- Passera probablement par des techniques **NLP**, avec nécessité d’un **processus d’annotation manuelle** préalable.
- En cas de données insuffisantes, envisager des méthodes d’augmentation :
  - traduction aller/retour,
  - reformulation automatique.

**Exemples de formulations à détecter :**
- "C’est difficile à expliquer"
- "Il faut le faire pour le comprendre"
- "C’est mieux si je vous montre"

### 3. Distance physique artisan / matière
- Identifier des indices verbaux décrivant la proximité ou l’éloignement physique entre l’artisan et la matière.

### 4. Temps d’attente
- Repérer les passages où l’artisan évoque des **temps d’attente sans intervention directe**, révélateurs de phases particulières dans la relation à la matière.

### 5. État physique de la matière
- Détecter les termes décrivant les transformations, propriétés ou conditions de la matière (texture, température, réaction, etc.).

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

