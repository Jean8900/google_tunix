# Fine-tuning de Gemma 2B avec GRPO pour le raisonnement mathématique

Ce projet implémente l'entraînement du modèle Gemma 2 2B-IT sur le benchmark GSM8K (problèmes mathématiques de niveau primaire) en utilisant l'algorithme GRPO (Group Relative Policy Optimization).

## Contexte

Les modèles de langage peuvent parfois donner des réponses sans montrer leur raisonnement. L'objectif ici est d'entraîner le modèle à expliciter son processus de réflexion avant de donner la réponse finale, ce qui améliore à la fois la transparence et la précision.

## Qu'est-ce que GRPO ?

GRPO est une variante optimisée de PPO (Proximal Policy Optimization) pour l'apprentissage par renforcement. La différence principale : on n'a pas besoin d'un modèle de valeur séparé, ce qui réduit considérablement l'utilisation mémoire.

**Principe de fonctionnement :**
1. Pour chaque question, on génère plusieurs réponses différentes (4 dans notre cas)
2. Chaque réponse reçoit une récompense selon sa qualité
3. On calcule un "avantage relatif" : les bonnes réponses du groupe sont renforcées, les mauvaises pénalisées
4. Le modèle apprend progressivement à privilégier les stratégies qui marchent

## Architecture

### Modèle de base
- **Gemma 2 2B-IT** de Google (2.6 milliards de paramètres)
- Disponible sur Kaggle après acceptation de la licence

### Fine-tuning avec LoRA
Au lieu de modifier tous les paramètres du modèle (très coûteux), on utilise LoRA :
- Seulement 79 millions de paramètres ajoutés
- Rank = 64, Alpha = 64
- Application sur les couches d'attention et MLP

### Infrastructure
- TPU v5e avec 8 cœurs
- Distribution sur 4 devices (FSDP + tensor parallelism)
- Entraînement en précision fp32

## Format d'entraînement

Le modèle apprend à structurer ses réponses ainsi :

```
<reasoning>
[Explication étape par étape du raisonnement]
</reasoning>
<answer>
[Réponse numérique finale]
</answer>
```

Ce format permet de séparer clairement le processus de pensée de la conclusion.

## Système de récompenses

Le modèle est évalué selon 4 critères :

**1. Format exact** (+3 points)
- Présence correcte des balises `<reasoning>` et `<answer>`
- Structure respectée

**2. Format approximatif** (+0.5 par élément)
- Compte le nombre de balises correctes
- Pénalise si trop de répétitions

**3. Précision de la réponse** (+3 points max)
- Réponse exacte : +3
- Réponse proche (±10%) : +0.5
- Réponse fausse : -1.0

**4. Extraction numérique** (+1.5 points)
- Vérifie qu'on peut extraire un nombre cohérent de la réponse

## Dataset : GSM8K

Collection de 8 500 problèmes mathématiques de niveau primaire, disponible via deux sources :
- TensorFlow Datasets (tfds)
- Kaggle

Exemple typique :
> "Marie a 4 pièces de 10 centimes, 4 pièces de 25 centimes et 7 pièces de 5 centimes. Sa mère lui donne 5 pièces de 25 centimes. Combien d'argent a-t-elle maintenant ?"

## Hyperparamètres clés

```python
LEARNING_RATE = 3e-6          # Taux d'apprentissage faible pour stabilité
BETA = 0.08                   # Contrôle de la divergence KL
EPSILON = 0.2                 # Clipping PPO-style
NUM_GENERATIONS = 4           # Réponses par prompt
TEMPERATURE = 0.9             # Diversité pendant l'entraînement
NUM_BATCHES = 3738           # ~7500 exemples d'entraînement
```

## Métriques d'évaluation

Le notebook évalue le modèle sur trois axes :

1. **Answer Accuracy** : pourcentage de réponses numériquement exactes
2. **Partial Accuracy** : réponses dans une marge de ±10%
3. **Format Accuracy** : respect du format de sortie

L'évaluation se fait avec génération "greedy" (température très basse) pour avoir des résultats reproductibles.

## Pipeline d'entraînement

1. **Chargement** : récupération du modèle pré-entraîné depuis Kaggle
2. **Préparation** : application de LoRA et sharding sur TPU
3. **Tokenization** : conversion des prompts avec le tokenizer de Gemma
4. **Génération** : 4 complétions par prompt pendant l'entraînement
5. **Récompenses** : calcul des scores selon les 4 fonctions
6. **Optimisation** : mise à jour des paramètres LoRA uniquement
7. **Checkpointing** : sauvegarde tous les 500 steps

## Utilisation

Le code est conçu pour tourner sur Kaggle avec un TPU v5e. Vous devrez :
- Accepter la licence Gemma sur Kaggle
- Configurer vos credentials Kaggle
- Optionnellement configurer W&B pour le tracking

## Résultats attendus

Après entraînement, le modèle devrait :
- Mieux structurer ses réponses
- Montrer son raisonnement étape par étape
- Améliorer sa précision sur les problèmes mathématiques
- Généraliser à des problèmes similaires non vus pendant l'entraînement

## Limitations

- Entraîné uniquement sur des problèmes arithmétiques de base
- Format rigide imposé (pas de flexibilité dans la structure)
- Nécessite un TPU pour un entraînement raisonnable
- Réglages des hyperparamètres non exhaustifs (baseline de départ)

## Références

- GRPO paper : [lien vers la publication originale]
- Gemma 2 : https://ai.google.dev/gemma
- GSM8K dataset : https://github.com/openai/grade-school-math
- Tunix library : https://github.com/google/tunix
