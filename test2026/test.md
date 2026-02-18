---

# **Travail Pratique d'Intelligence Artificielle Appliquée à la Finance**
## Apprentissage Supervisé (KNN), Évaluation et Surapprentissage


<p align="right">
<img src="https://raw.githubusercontent.com/hajar2004hamine/Support-AI/main/encg.png" width="120">
</p> 

<p align="left">
<img src="https://raw.githubusercontent.com/hajar2004hamine/Support-AI/main/Hajar.jpg" width="120">
</p>

<br><br>

<p align="left">
<img src="https://github.com/user-attachments/assets/483ba06c-0cc1-4c28-be78-8203847bc3bf" width="120">
</p>

---

| | |
|---|---|
| **Cours** | Intelligence Artificielle - Contrôle, Audit et Conseil |
| **Établissement** | ENCG Settat - 4ème année |
| **Professeur** | A. Larhlimi |
| **Date** | 17 Février 2026 |
| **Elaborée par** | HAMINE Hajar et HASSI Asmae |

---

## Table des Matières

1.  [Introduction](#introduction)
2.  [Partie 1 : Statistiques et Loi Normale en Finance (Analyse de Portefeuille)](#partie-1--statistiques-et-loi-normale-en-finance-analyse-de-portefeuille)
    - [1.1 Statistiques Descriptives](#11--statistiques-descriptives)
    - [1.2 Visualisation des Distributions](#12--visualisation-des-distributions)
    - [1.3 Value at Risk (VaR 95%) et Test de Normalité](#13--value-at-risk-var-95-et-test-de-normalité)
    - [1.4 Ratio de Sharpe et Recommandation Client](#14--ratio-de-sharpe-et-recommandation-client)
3.  [Partie 2 : Théorème de Bayes et Scoring Crédit (Mise à Jour du Risque)](#partie-2--théorème-de-bayes-et-scoring-crédit-mise-à-jour-du-risque)
    - [2.1 Calcul Bayésien Manuel pour un Client Standard](#21--calcul-bayésien-manuel-pour-un-client-standard)
    - [2.2 Mise à Jour Séquentielle](#22--mise-à-jour-séquentielle)
    - [2.3 Fonction Générique de Mise à Jour Bayésienne](#23--fonction-générique-de-mise-à-jour-bayésienne)
    - [2.4 Matrice de Confusion et Lien avec Bayes](#24--matrice-de-confusion-et-lien-avec-bayes)
4.  [Partie 3 : Modélisation Prédictive avec KNN](#partie-3--modélisation-prédictive-avec-knn)
    - [3.1 Génération et Exploration du Dataset](#31--génération-et-exploration-du-dataset)
    - [3.2 Préparation des Données (Preprocessing)](#32--préparation-des-données-preprocessing)
    - [3.3 Optimisation de l'Hyperparamètre K par Validation Croisée](#33--optimisation-de-lhyperparamètre-k-par-validation-croisée)
    - [3.4 Évaluation du Modèle Final sur l'Ensemble de Test](#34--évaluation-du-modèle-final-sur-lensemble-de-test)
    - [3.5 Analyse de la Courbe ROC et du Seuil de Décision](#35--analyse-de-la-courbe-roc-et-du-seuil-de-décision)
    - [3.6 Calcul du Retour sur Investissement (ROI) et Recommandation Finale](#36--calcul-du-retour-sur-investissement-roi-et-recommandation-finale)
5.  [Conclusion Générale](#conclusion-générale)


---

## Introduction

Ce rapport présente l'analyse complète d'un TP d'Intelligence Artificielle appliqué à la finance. Le travail est structuré en trois parties distinctes et complémentaires :

1.  **Analyse de Portefeuille :** Nous commençons par une analyse statistique classique de deux portefeuilles d'actifs (conservateur et agressif) pour calculer des indicateurs de risque fondamentaux comme la Value at Risk (VaR) et le ratio de Sharpe.
2.  **Scoring Crédit par Inférence Bayésienne :** La deuxième partie illustre l'utilisation du théorème de Bayes pour la mise à jour dynamique du risque de défaut d'un client, un concept clé dans les systèmes de scoring.
3.  **Modélisation Prédictive avec KNN :** La partie centrale du TP consiste à construire, optimiser et évaluer un modèle de classification supervisé (K-Nearest Neighbors) pour prédire le défaut de paiement. L'accent est mis sur l'optimisation des hyperparamètres, l'évaluation robuste des performances et la traduction de ces performances en termes d'impact financier concret via le calcul du retour sur investissement (ROI).

L'objectif principal est de démontrer la capacité à transformer des données financières en décisions stratégiques éclairées, en utilisant des outils quantitatifs et d'apprentissage automatique.

---

## Partie 1 : Statistiques et Loi Normale en Finance (Analyse de Portefeuille)

### Contexte Métier
En tant qu'analyste risques, nous devons conseiller un client disposant d'un capital de **500 000 €**. Deux options d'investissement sont à l'étude :
*   **Portefeuille A (Conservateur)** : Actions de grandes capitalisations européennes (Blue-chips).
*   **Portefeuille B (Agressif)** : Actions de petites capitalisations du secteur technologique émergent.

Le client exige une **perte maximale de 50 000 € (10% du capital) sur un horizon annuel, avec un niveau de confiance de 95%**.

---

### 1.1 — Statistiques Descriptives

L'analyse des rendements mensuels historiques des deux portefeuilles a fourni les statistiques clés suivantes :

```
================================================================================
QUESTION 1.1 — STATISTIQUES DESCRIPTIVES
================================================================================

PORTEFEUILLE CONSERVATIVE (A)
	• Rendement mensuel moyen : 0.94%
	• Écart-type mensuel : 0.48%
	• Médiane : 1.00%
	• Rendement annualisé : 11.85%
	• Volatilité annualisée : 1.65%

PORTEFEUILLE AGRESSIF (B)
	• Rendement mensuel moyen : 2.89%
	• Écart-type mensuel : 4.45%
	• Médiane : 4.70%
	• Rendement annualisé : 40.79%
	• Volatilité annualisée : 15.41%
```

**Interprétation des Statistiques :**
*   Le **portefeuille B** (agressif) offre un rendement annualisé moyen bien supérieur (40.79% contre 11.85%), mais cette performance est accompagnée d'une **volatilité nettement plus élevée** (15.41% contre 1.65%). Cela reflète le risque inhérent aux actifs de croissance.
*   Le **portefeuille A** (conservateur) se caractérise par une grande stabilité. Sa médiane de 1.00% est très proche de sa moyenne (0.94%), indiquant une distribution symétrique et peu d'impact des valeurs extrêmes.
*   L'écart-type, en tant que mesure de la volatilité, est le premier indicateur du risque de chaque actif.

---

### 1.2 — Visualisation des Distributions

Les graphiques ci-dessous illustrent les différences de comportement entre les deux portefeuilles.

**Figure 1.2 : Distributions des Rendements Mensuels**
<img width="1489" height="590" alt="image" src="https://github.com/user-attachments/assets/bf30f273-2d3b-4f71-82cc-15d4b54056c0" />

**Description des Visualisations :**
*   **Histogrammes (Gauche) :** La distribution des rendements du portefeuille A (vert) est étroite et centrée autour de 1%, confirmant sa faible volatilité. En revanche, la distribution du portefeuille B (rouge) est beaucoup plus étalée, avec des rendements allant de -5% à +9%, illustrant son caractère plus risqué et la présence de valeurs extrêmes (outliers).
*   **Boxplots (Droite) :** Les boxplots confirment cette analyse. La "boîte" du portefeuille A est très compacte. Pour le portefeuille B, la boîte est plus large et la présence de plusieurs outliers au-delà des "moustaches" indique des événements de marché extrêmes, typiques des actifs agressifs.

---

### 1.3 — Value at Risk (VaR 95%) et Test de Normalité

La VaR paramétrique, sous l'hypothèse de normalité des rendements, a été calculée pour les horizons mensuel et annuel.

**Résultats de la VaR :**
```
--- VaR Mensuelle (95%) ---
Portefeuille A (Conservative): VaR = 0.15% (Perte de 763.37 €)
Portefeuille B (Agressif): VaR = -4.42% (Perte de -22,117.99 €)

--- VaR Annuelle (95%) ---
Portefeuille A (Conservative): VaR = 8.53% (Perte de 42,656.41 €)
Portefeuille B (Agressif): VaR = 9.36% (Perte de 46,795.89 €)
```

**Test de Normalité de Shapiro-Wilk :**
```
--- Test de Normalité (Shapiro-Wilk) ---
Portefeuille A (Conservative): Statistique=0.803, p-value=0.000
Portefeuille B (Agressif): Statistique=0.837, p-value=0.001
```

**Interprétation et Respect de la Contrainte Client :**
*   **VaR et Contrainte Client :** Les pertes maximales annuelles estimées pour les deux portefeuilles (42 656 € pour A et 46 796 € pour B) sont toutes deux inférieures à la limite de 50 000 € fixée par le client. Sur ce seul critère, **les deux portefeuilles sont acceptables**.
*   **Test de Normalité :** Les p-values des tests de Shapiro pour les deux portefeuilles sont inférieures à 0.05. Nous rejetons donc l'hypothèse nulle de normalité. **Les rendements ne suivent pas une loi normale**. Par conséquent, la VaR paramétrique calculée ici doit être interprétée avec une grande prudence. Elle pourrait sous-estimer le risque réel, notamment la probabilité d'événements extrêmes. Une approche non-paramétrique (comme la VaR historique) serait plus robuste dans ce contexte.

---

### 1.4 — Ratio de Sharpe et Recommandation Client

Le ratio de Sharpe mesure le rendement excédentaire par unité de risque prise.
*   **Sharpe A** : 5.35
*   **Sharpe B** : 2.45

**Recommandation :**
```
--- Justification de la recommandation ---
Les deux portefeuilles respectent la contrainte de perte maximale annuelle du client de 50 000 €. Cependant, le Portefeuille A (Sharpe: 5.35) offre un meilleur rendement ajusté au risque que le Portefeuille B (Sharpe: 2.45). De plus, la volatilité annuelle du portefeuille A est nettement inférieure. Il est crucial de noter que les rendements d'aucun des portefeuilles ne suivent une distribution normale (p-value Shapiro < 0.05). La VaR paramétrique doit donc être interprétée avec prudence, et des méthodes non-paramétriques seraient plus robustes. Compte tenu de la contrainte VaR respectée et du Sharpe Ratio supérieur, le Portefeuille A est recommandé pour un investisseur privilégiant un meilleur rendement ajusté au risque et une volatilité plus faible.
```

**Conclusion de la Partie 1 :** Bien que le portefeuille agressif promette un rendement plus élevé, le portefeuille conservateur est plus efficient (Sharpe plus élevé) et respecte la contrainte de risque. Compte tenu du profil de risque présumé du client, le **Portefeuille A est recommandé**.

---

## Partie 2 : Théorème de Bayes et Scoring Crédit (Mise à Jour du Risque)

### Contexte Métier
Nous développons un système de scoring crédit dynamique. L'objectif est de mettre à jour la probabilité de défaut d'un client en fonction de nouveaux événements observés (retard de paiement, découvert), en utilisant le théorème de Bayes.

---

### 2.1 — Calcul Bayésien Manuel pour un Client Standard

Pour un client du segment "Standard" (taux de défaut a priori de 5%) observant un retard de paiement :

**Calculs :**
*   Prior P(D) = 0.05
*   P(Retard|D) = 0.80
*   P(Retard|¬D) = 0.10

1.  **Calcul de la vraisemblance totale de l'événement :**
    `P(Retard) = [P(Retard|D) * P(D)] + [P(Retard|¬D) * P(¬D)]`
    `P(Retard) = (0.80 * 0.05) + (0.10 * 0.95) = 0.04 + 0.095 = 0.135 (13.5%)`

2.  **Application du Théorème de Bayes :**
    `P(D|Retard) = [P(Retard|D) * P(D)] / P(Retard)`
    `P(D|Retard) = 0.04 / 0.135 = 0.2963 (29.63%)`

**Interprétation et Décision Métier :**
*   **Augmentation du risque :** La probabilité de défaut est passée de 5% à 29.63%, soit un **facteur multiplicatif de 5.93**. Le risque a été multiplié par près de 6.
*   **Décision Métier :** Face à une augmentation aussi significative (facteur > 5), la recommandation métier est de **restreindre immédiatement le crédit**, de mettre le client sous surveillance accrue et d'initier une enquête approfondie sur la cause de ce retard.

---

### 2.2 — Mise à Jour Séquentielle

Deux semaines plus tard, ce même client présente un découvert supérieur à 500€. Nous utilisons sa probabilité postérieure (29.63%) comme nouveau prior.
*   Nouveau prior P(D) = 0.2963
*   P(Découvert|D) = 0.65
*   P(Découvert|¬D) = 0.15

**Calculs :**
1.  `P(Découvert) = (0.65 * 0.2963) + (0.15 * (1-0.2963)) = 0.1926 + 0.1056 = 0.2982 (29.82%)`
2.  `P(D|Retard ET Découvert) = (0.65 * 0.2963) / 0.2982 = 0.6461 (64.61%)`

**Figure 2.2 : Évolution de la Probabilité de Défaut**
<img width="984" height="590" alt="image" src="https://github.com/user-attachments/assets/28ff4c09-de8a-459a-8a7e-bf5cd9464787" />


**Interprétation :** L'accumulation d'événements négatifs transforme un client "standard" en un client à très haut risque. La probabilité de défaut, initialement de 5%, grimpe à près de 65% après seulement deux incidents. Cela justifierait des mesures de gestion de risque extrêmement strictes.

---

### 2.3 — Fonction Générique de Mise à Jour Bayésienne

Une fonction Python `bayes_update()` a été implémentée pour automatiser ce processus.

```python
def bayes_update(prior, likelihood_pos, likelihood_neg):
    """
    Calcule la probabilité a posteriori via le théorème de Bayes.

    Args:
        prior (float): Probabilité a priori P(A).
        likelihood_pos (float): Vraisemblance P(Evidence|Positive).
        likelihood_neg (float): Vraisemblance P(Evidence|Negative).

    Returns:
        float: Probabilité a posteriori P(Positive|Evidence).
    """
    p_evidence = (likelihood_pos * prior) + (likelihood_neg * (1 - prior))
    if p_evidence == 0:
        return 0.0
    posterior = (likelihood_pos * prior) / p_evidence
    return posterior
```

**Test sur un client du segment Risque (prior 15%) :**
L'application séquentielle des trois événements (retard, découvert, refus de crédit) montre une escalade du risque :
*   Initial: 15.00%
*   Après 'Retard paiement': 58.54%
*   Après 'Découvert >500€': 85.95%
*   Après 'Demande crédit refusée ailleurs': 97.68%

Un tel client, avec une probabilité de défaut proche de 100%, serait inéligible à tout nouveau crédit sans garanties exceptionnelles.

---

### 2.4 — Matrice de Confusion et Lien avec Bayes

Sur un échantillon de test de 10 000 clients, le modèle "retard de paiement" a produit les résultats suivants :
*   TP (Vrais Positifs) = 400
*   FP (Faux Positifs) = 950

**Calcul de la Précision :**
`Precision = TP / (TP + FP) = 400 / (400 + 950) = 0.2963`

**Cohérence avec Bayes :**
La précision (0.2963) est rigoureusement égale à la probabilité postérieure P(Défaut|Retard) calculée à la question 2.1 (29.63%).
**Explication :** La précision d'un classifieur n'est rien d'autre qu'une estimation empirique de la probabilité bayésienne P(Classe Vraie | Prédiction Positive). Elle représente, parmi tous les clients pour lesquels le modèle a prédit un "retard" (un signal positif), la proportion qui sont réellement en défaut.

---

## Partie 3 : Modélisation Prédictive avec KNN

### Contexte Métier
Nous devons construire un modèle de scoring automatisé pour des prêts personnels. L'objectif métier prioritaire est d'**identifier au moins 80% des futurs défauts (Recall ≥ 80%)**, tout en gardant une précision raisonnable (>60%) pour ne pas inonder les équipes d'analyse de faux positifs. L'impact financier final est mesuré par le ROI.

---

### 3.1 — Génération et Exploration du Dataset

Un jeu de données synthétique de 2000 clients a été généré, avec un déséquilibre de classe (15% de défauts).

**Structure et Statistiques Clés :**
*   Taux de défaut global : **15.00%**
*   Distribution des classes : 1700 (0) vs 300 (1)
*   Les statistiques descriptives (moyennes, écart-types) des features semblent cohérentes avec des données financières plausibles (âge moyen ~45 ans, salaire moyen ~87k€, etc.).

**Analyse des Corrélations avec la Variable Cible 'defaut' :**
```
Corrélation des caractéristiques avec la variable 'defaut' (triée):
score_credit_bureau    0.042660
salaire                0.023595
anciennete_emploi      0.008688
dette_totale           0.004707
age                   -0.002051
nb_credits_actifs     -0.009355
historique_retards    -0.015507
ratio_dette_revenu    -0.056378
```

**Interprétation :** Les corrélations linéaires sont très faibles. Le `score_credit_bureau` et le `ratio_dette_revenu` sont les deux caractéristiques les plus (bien que faiblement) corrélées au défaut. Cela suggère que la relation entre les features et la cible est probablement non-linéaire, ce qui peut être un défi pour certains modèles.

**Figure 3.1 : Heatmap de Corrélation**
<img width="891" height="808" alt="image" src="https://github.com/user-attachments/assets/60e88ac6-458d-485c-84a6-52d0d2c95581" />

---

### 3.2 — Préparation des Données (Preprocessing)

Les données ont été préparées rigoureusement pour l'entraînement :
*   **Séparation :** Features (`X`) et cible (`y`).
*   **Split Stratifié :** Division en ensembles d'entraînement (80%) et de test (20%) en respectant la proportion des classes (85/15) dans chaque ensemble, crucial pour l'évaluation d'un dataset déséquilibré.
*   **Normalisation :** Les features ont été standardisées en utilisant `StandardScaler` (ajusté uniquement sur l'ensemble d'entraînement pour éviter le "data leakage").

**Vérification de la stratification :**
```
Class distribution in y_train:
defaut
0    0.85
1    0.15

Class distribution in y_test:
defaut
0    0.85
1    0.15
```
La stratification a bien fonctionné, les proportions sont identiques.

---

### 3.3 — Optimisation de l'Hyperparamètre K par Validation Croisée

Nous avons évalué la performance du modèle KNN pour différentes valeurs de K (de 1 à 30) en utilisant une validation croisée stratifiée en 5 plis. L'objectif était de trouver le K optimal maximisant l'AUC, tout en surveillant le Recall et la Precision.

**Résultats de l'Optimisation (Extrait) :**

| K | AUC_mean | AUC_std | Recall_mean | Precision_mean |
| :- | :--------- | :-------- | :------------ | :--------------- |
| 1 | 0.4898     | 0.0361    | 0.1333        | 0.1328           |
| 3 | 0.4831     | 0.0543    | 0.0458        | 0.1239           |
| 5 | 0.4605     | 0.0506    | 0.0167        | 0.1082           |
| 7 | 0.4382     | 0.0349    | 0.0042        | 0.0400           |
| 9 | 0.4556     | 0.0424    | 0.0042        | 0.1000           |
| 11| 0.4422     | 0.0432    | 0.0000        | 0.0000           |

**Figure 3.3 : Performance du KNN (AUC, Recall, Precision) en Fonction de K**
<img width="1489" height="690" alt="image" src="https://github.com/user-attachments/assets/4c842d33-6567-43c1-9307-41ab40ca15be" />


**Analyse des Résultats :**
*   **K Optimal :** Le K qui maximise l'AUC moyenne est **K=1**, avec une AUC de seulement **0.49**. Cette valeur est proche de 0.5, ce qui indique que le modèle a un pouvoir discriminant à peine meilleur qu'un tirage aléatoire.
*   **Objectifs Métier Non Atteints :** Aucune valeur de K ne permet d'atteindre les objectifs de Recall (≥80%) et de Precision (≥60%). Les performances sont particulièrement mauvaises pour K>5, où le Recall devient nul, signifiant que le modèle ne détecte **aucun** défaut.
*   **Conclusion sur l'Optimisation :** Le KNN, avec les features disponibles, est incapable de produire un modèle satisfaisant. Les très faibles corrélations entre les features et la cible, combinées au déséquilibre de classe, expliquent cet échec.

---

### 3.4 — Évaluation du Modèle Final sur l'Ensemble de Test

Le modèle final a été entraîné avec K=1 sur tout l'ensemble d'entraînement, puis évalué sur l'ensemble de test.

**Confusion Matrix Values :**
*   True Positives (TP): 11
*   False Positives (FP): 48
*   False Negatives (FN): 49
*   True Negatives (TN): 292

**Classification Metrics :**
*   **Accuracy:** 0.7575
*   **Precision (for class 1):** 0.1864
*   **Recall (for class 1):** 0.1833
*   **F1-Score:** 0.1849
*   **Specificity:** 0.8588

**Figure 3.4 : Matrice de Confusion du Modèle Final**
<img width="649" height="547" alt="image" src="https://github.com/user-attachments/assets/987530fb-a664-4107-acfe-fe509c149f2d" />

**Interprétation des Performances :**
Les performances sur le jeu de test confirment les résultats de la validation croisée. Le modèle a un **Recall extrêmement faible (18%)** : il ne parvient à identifier correctement que 11 des 60 clients qui feront défaut. Les 49 autres (FN) représentent des pertes financières directes. De plus, sa **Précision est très basse (19%)** : parmi les 59 clients identifiés comme "à risque", seuls 11 le sont vraiment, les 48 autres (FP) entraîneront des coûts d'analyse et d'opportunité inutiles. Le modèle est donc **inutilisable en l'état**.

---

### 3.5 — Analyse de la Courbe ROC et du Seuil de Décision

La courbe ROC confirme le faible pouvoir discriminant du modèle.

**Figure 3.5 : Courbe ROC et Seuil Optimal de Youden**
<img width="857" height="701" alt="image" src="https://github.com/user-attachments/assets/9a962920-eb61-4be5-97a0-95a197e7abe2" />
ng)

**Analyse :**
*   **AUC :** L'aire sous la courbe est de 0.52, confirmant que le modèle est à peine meilleur qu'un classifieur aléatoire (diagonale pointillée).
*   **Seuil Optimal de Youden :** L'indice de Youden (J = TPR - FPR) est maximisé à un seuil de **1.0**. Cela signifie que pour ce modèle, la meilleure séparation entre les classes est obtenue en n'attribuant la classe positive qu'aux points ayant une probabilité prédite de 1.0 (le plus proche voisin unique). Cela reflète l'incertitude extrême du modèle.
*   **Évaluation de seuils spécifiques :**
    *   Seuil 0.3 → **Recall:** 0.1833, **Precision:** 0.1864
    *   Seuil 0.5 → **Recall:** 0.1833, **Precision:** 0.1864
    *   Seuil 0.7 → **Recall:** 0.1833, **Precision:** 0.1864
    *   **Les résultats sont identiques pour les trois seuils**, car le modèle ne produit presque que des prédictions de probabilité à 0 ou 1, un comportement typique de KNN avec K=1. Il n'y a pas de "réglage fin" possible du seuil pour ce modèle.

---

### 3.6 — Calcul du Retour sur Investissement (ROI) et Recommandation Finale

En appliquant les coûts métier aux matrices de confusion des différents seuils (tous identiques), nous obtenons un ROI catastrophique.

**Calcul du ROI pour le seuil 0.5 :**
*   Gains totaux (TP) : 11 * 15,000 = 165,000 €
*   Coûts totaux (FP) : 48 * (500 + 1,200) = 81,600 €
*   Pertes totales (FN) : 49 * 15,000 = 735,000 €
*   **Net ROI pour le seuil 0.5: -651,600.00 €**
    ---

## Conclusion Générale

Ce TP a permis de parcourir l'ensemble du cycle de vie d'un projet d'IA en finance, de l'analyse statistique descriptive à l'évaluation financière d'un modèle prédictif.

1.  **L'analyse de portefeuille** a démontré l'importance de ne pas se fier uniquement au rendement, mais de considérer le risque (volatilité) et l'efficience (Sharpe) pour une décision d'investissement éclairée.
2.  **L'approche bayésienne** a illustré avec force comment intégrer de l'information de manière dynamique pour le suivi du risque client, un pilier des systèmes de scoring modernes.
3.  **La modélisation KNN** a été un cas d'école d'échec de modélisation. Malgré une préparation rigoureuse et une optimisation méthodique, le modèle s'est révélé incapable d'apprendre une relation significative entre les features et la cible. Cela a mis en lumière des défis cruciaux en data science :
    *   **L'importance de la qualité des features :** Des corrélations trop faibles condamnent le modèle.
    *   **L'impact du déséquilibre de classe :** Un modèle peut être très "précis" en prédisant toujours la classe majoritaire, mais totalement inutile pour le problème métier.
    *   **L'évaluation par le ROI :** Elle ancre la discussion dans la réalité économique et justifie l'abandon d'un projet non viable, ce qui est une décision aussi importante que de recommander un modèle performant.

Ce travail souligne que la maîtrise des outils techniques (Python, Scikit-learn) doit toujours être accompagnée d'une **compréhension profonde du métier** et de la **capacité à interpréter les résultats** pour guider la prise de décision.

---
