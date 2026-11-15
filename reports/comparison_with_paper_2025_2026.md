# Comparaison Détaillée : Nos Résultats vs Article INRIA 2025/2026 (Ju, Collas, Bouchard, Thirion)

**Article de Référence :**  
*“Benchmarking External Generalization of SPD Matrix Learning for Resting-State fMRI Connectome Prediction”*  
**Auteurs :** Ce Ju, Antoine Collas, Florent Bouchard, Bertrand Thirion  
**Institution :** INRIA Parietal, NeuroSpin, CEA, Université Paris-Saclay (2025/2026)  
**Dépôt GitHub Officiel :** [GeometricBCI/rsfmri-spd-connectome-external-generalization-benchmark](https://github.com/GeometricBCI/rsfmri-spd-connectome-external-generalization-benchmark)

---

## 1. Tableau Comparatif Synthétique

| Critère d'Évaluation | Article INRIA 2025/2026 (Ju et al.) | Notre Étude (Stage ENSI 2026) | Accord & Complémentarité |
| :--- | :--- | :--- | :--- |
| **Données Cérébrales** | Connectomes rs-fMRI (COBRE, ADNI, ABIDE, OASIS-3, Cam-CAN) | Connectomes rs-fMRI COBRE ($N=86$ sujets, matrices $28 \times 28$ SPD) | **Identique :** Même cohorte clinique COBRE (Schizophrénie vs Témoins) |
| **Objectif Principal** | Généralisation externe (LODO) et régression de biomarqueurs (Age) | Réduction de dimension ($D=406 \to d=5$) $\times$ Choix du classifieur | **Complémentaire :** Étude réduction $\times$ classification demandée dans le sujet |
| **Représentation Euclidienne** | `CorrVec + Ridge` ($R^2 = 0.785$, avec harm. $0.842$) | `ACP Naïve + Classifieurs` (Acc: $62.75\%$ à $65.16\%$, Distorsion: $0.344$) | **Accord :** L'aplatissement euclidien ignore la courbure et plafonne |
| **Espace Tangent (t-PCA)** | `TS Ridge` ($R^2 = 0.819$, avec harm. $0.871$) &rarr; **Gain net riemannien** | `PCA Tangente + SVM` ($66.34\%$, Distorsion minimale: $0.324$) | **Accord :** La projection tangente surpasse l'euclidien si le classifieur est adapté |
| **Limitation du Barycentre Unique** | Échec sous forte hétérogénéité / décalage de distribution (LODO) | Chute de performance avec $k$-NN ($56.93\%$) due aux distorsions locales | **Accord Majeur :** Un barycentre global unique étire les amas périphériques |
| **Solution Multi-Amas / Non-Linéaire** | Réseau profond `SPDNet` ($R^2 = 0.876$) & Harmonisation riemannienne | `Multi-Repères Adaptatifs` ($K=2$ barycentres locaux) + SVM Linéaire (**$73.40\%$**) | **Validation :** Multi-Repères égale la baseline brute ($73.46\%$) avec $98.8\%$ de compression |
| **Démystification du Classifieur** | Non abordé (classifieur unique par pipeline : Ridge ou MLP) | **Résolu :** Démontre que le paradoxe antérieur venait du classifieur ($k$-NN vs SVM) | **Apport Clé ENSI :** Répond exactement à l'objectif de la fiche de stage |

---

## 2. Analyse Approfondie des Convergences et Avancées

### A. Convergence : La Supériorité de la Géométrie Riemannienne
Dans l'article de Ju et al. (INRIA), la projection en espace tangent (`TS Ridge`) surpasse systématiquement la vectorisation euclidienne naïve (`CorrVec Ridge`), passant d'un $R^2$ de $0.785$ à $0.819$ (sans harmonisation) et de $0.842$ à $0.871$ (avec harmonisation).  
Dans nos expériences sur les mêmes connectomes COBRE :
* L'ACP euclidienne plafonne à **$62.75\%$** en SVM.
* La PCA Tangente Riemannienne atteint **$66.34\%$** en SVM RBF (+3.59% de gain direct).
* La distorsion géodésique de la PCA Tangente est la plus basse (**0.3240** vs 0.3440 pour l'ACP).

### B. Convergence : La Limite du Barycentre Unique de Fréchet
L'article d'INRIA souligne une vulnérabilité critique : sous décalage de distribution (Leave-One-Dataset-Out), le modèle à barycentre tangent unique perd en généralisation (l'erreur MAE monte à 16-19 ans sur COBRE), car une seule référence $\Gamma^*$ ne peut pas linéariser fidèlement une variété multimodale ou hétérogène.  
Nos tests confirment ce phénomène géométrique :
* Sur connectomes réels, la projection sur un barycentre global unique étire certains voisinages locaux, ce qui fait chuter le $k$-NN à **$56.93\%$**.
* Notre méthode de **Multi-Repères Adaptatifs** (partitionnement en $K=2$ barycentres locaux) résout cette limitation et propulse l'accuracy à **$73.40 \pm 8.90\%$** (+10.65% par rapport à l'ACP euclidienne).

### C. Notre Apport Propre : La Résolution du Paradoxe « Réduction $\times$ Classifieur »
La question centrale posée par le sujet de stage ENSI était :  
> *« Une étude précédente avec un classifieur unique ($k$-NN) avait conclu à l'absence de gain de la PCA Tangente. Dans quelle mesure le choix du classifieur compense-t-il cette limite ? »*

L'article INRIA benchmarkait la régression avec Ridge et SPDNet, mais n'analysait pas l'interaction entre réduction et typologie de classifieur. Notre travail apporte la réponse définitive :
1. **Le paradoxe est reproduit :** Le $k$-NN échoue sur la PCA Tangente ($56.93\%$ vs $65.16\%$ pour l'ACP) car il partitionne l'espace selon des cellules de Voronoï sensibles aux déformations tangentielles locales.
2. **Le classifieur à marge maximale (SVM) rétablit la supériorité riemannienne :** En maximisant la marge globale, le SVM exploite l'isométrie géodésique et fait rebondir la performance à **$66.34\%$** (PCA Tangente) et **$73.40\%$** (Multi-Repères).
