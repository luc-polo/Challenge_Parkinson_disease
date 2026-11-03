# Challenge Parkinson — prédiction du score moteur OFF

Ce projet vise à prédire le score moteur OFF corrigé pour chaque visite d'un patient dans le cadre du [challenge de l'Institut du Cerveau](https://challengedata.ens.fr/challenges/159).

[Guide du dépôt et des notebooks](README_REPO.md)

## Explication de la démarche

### Premier benchmark

Un premier prétraitement a été réalisé : séparation entraînement/validation par patient, encodage des variables catégorielles et imputation par la médiane avec indicateurs de valeurs manquantes. Plusieurs modèles de régression ont ensuite été comparés sur la MSE de validation, sans optimisation des hyperparamètres, afin d'établir un benchmark pour les prochaines améliorations et de produire une première soumission.
Dans ce pipeline de benchmark, on encode les variables catégorielles avec un encodage one-hot et on impute les données manquantes par la médiane en ajoutant une colonne indicatrice de valeur manquante. On s'assure également de regrouper les visites d'un même patient dans le même ensemble lors de chaque séparation de validation croisée, puis on retire la colonne `patient_id` des variables explicatives. Le meilleur modèle testé est un modèle de gradient boosting :

Résultats du premier benchmark :

- LR : MSE 72.681 ± 1.229
- XGBoost : MSE 56.585 ± 0.748
- SVM : MSE 59.266 ± 1.040
- HistGB : MSE 55.039 ± 0.728
- RandomForest : MSE 58.789 ± 0.988

### Prétraitement

Dans cette partie, nous ne citons que les résultats de HistGB, car il est resté le meilleur parmi les modèles testés au fil des expérimentations.

#### Imputation des données manquantes

Nous avons travaillé principalement avec des modèles d'imputation multivariée par régression avec gradient boosting. Nous avons d'abord essayé de prédire les valeurs manquantes de trois variables, puis de quatre, à l'aide de ces modèles. Les autres variables numériques sont imputées par la médiane, tandis que les valeurs manquantes de `gene` sont représentées par la catégorie one-hot `gene_nan`. Pour chaque variable imputée, nous avons également conservé la colonne indicatrice de valeur manquante. Voici les resultats:

- `multivariate_3` — HistGB : MSE 54.394 ± 0.712
- `multivariate_4` — HistGB : MSE 54.234 ± 0.721

L'amélioration est maigre (réduction de la MSE de 1,46 % par rapport au premier benchmark).

Nous avons ensuite ajouté des variables construites à partir des visites antérieures et ultérieures disponibles du patient, afin d'aider les modèles d'imputation multivariée par régression. Exemples :
`on/off_median_patient`, `time_since_first_visit`, `on/off_slope_patient`, `ledd_median_patient`, etc.

- `multivariate_4_history` — HistGB : MSE 45.286 ± 0.909

Le gain a ici été assez phénoménal (réduction de la MSE de 16,5 %). Il provient tout particulièrement de l'imputation de la variable `off` avec l'historique, qui fait à elle seule baisser la MSE de 13,6 %. Ces deux gains sont mesurés par rapport à `multivariate_4`, sans historique.

Améliorations possibles :

- Entraîner les modèles d'imputation multivariée successivement, en utilisant les prédictions des précédents comme entrées.
- Essayer d'imputer les valeurs de `time_since_intake_on/off`.
- Essayer d'ajouter des variables plus pertinentes pour les modèles d'imputation multivariée par régression.


#### Features engineering

J'ai tenté de créer différentes features pour améliorer les performances du modèles. Toutes le features précédemment mentionnées dans l'imputage étaient spécifique pour entrainer les imputeurs à regression, elles n'ont pas été directement coservées. Ainsi dans cettepartie j'ai repris le features engineering de 0.

On peut distinguer deux groupes de features différents:
-Per visit
-Cross visits: features qui calcule une statistique qui utilise des valeurs tirées de visites futures/passées du même patient

Voici les features qui ont été testées:

- `time_since_first_visit` : temps écoulé depuis la première visite du patient.
- `n_past_visits`, `n_future_visits` : nombres de visites strictement antérieures et ultérieures à la visite actuelle.
- `on_slope_patient`, `off_slope_patient`, `ledd_slope_patient` : pentes annuelles des droites ajustées aux valeurs du patient en fonction de son âge.
- `off_on_median_patient` : médiane de l’écart `off - on` du patient.
- `on_trend_at_visit`, `off_trend_at_visit` : valeurs estimées à l’âge de la visite actuelle par les droites de tendance du patient ; `off_trend_other_visits` : tendance estimée à partir des OFF observés aux autres visites.
- `on_previous`, `on_next`, `off_previous`, `off_next` : valeurs ON/OFF des visites précédente et suivante disponibles.
- `on_previous_gap`, `on_next_gap`, `off_previous_gap`, `off_next_gap` : écarts de temps entre la visite actuelle et les visites correspondant à ces valeurs voisines.
- `on_n_observed`, `off_n_observed`, `ledd_n_observed` : nombres de valeurs non manquantes après imputation par patient ; `off_observed_fraction_patient` : proportion de visites avec un OFF réellement observé.
- `on_std_patient`, `off_std_patient` : écarts-types des valeurs ON et OFF du patient.

##### Ce que j'ai essayé mais qui n'a pas aidé le modèle:
- `ledd_time_ratio_on`, `ledd_time_ratio_off` : rapports `ledd / (1 + time_since_intake_on)` et `ledd / (1 + time_since_intake_off)`.
- `ledd_x_time_on`, `ledd_x_time_off` : produits de la LEDD par le délai depuis la prise du traitement pour les mesures ON et OFF.
- `disease_duration` : durée de la maladie, calculée par `age - age_at_diagnosis`.
- `on_median_patient`, `off_median_patient`, `ledd_median_patient` : médianes des valeurs du patient sur ses différentes visites.

- Effet médicamenteux :

J’ai fait un premier essai en définissant une courbe triangulaire commune $h_\theta(t)$ pour approximer l’effet du traitement en fonction du temps écoulé depuis la dernière prise :

$$
h_\theta(t)=\max\left(0,\min\left(\frac{t}{t_{\mathrm{pic}}},\frac{t_{\mathrm{fin}}-t}{t_{\mathrm{fin}}-t_{\mathrm{pic}}}\right)\right),
\qquad \theta=(t_{\mathrm{pic}},t_{\mathrm{fin}}),\quad 0<t_{\mathrm{pic}}<t_{\mathrm{fin}}.
$$

La courbe est normalisée à 1 au pic ; le premier essai utilisait un effet $a\,h_\theta(t)$ avec une amplitude $a$ commune à tous les patients.

J’ai ensuite modélisé cet effet par une amplitude individuelle $A_i$. Leur ajustement conjoint, sur chaque fold d’entraînement, minimise :

$$
\min_{\theta,\{A_i\ge0\}}\sum_{i,j}\left[(OFF-ON)_{ij}-A_i\bigl(h_\theta(t_{ON,ij})-h_\theta(t_{OFF,ij})\bigr)\right]^2.
$$

On effectue une recherche sur grille (*grid search*) sur les paramètres $\theta=(t_{\mathrm{pic}},t_{\mathrm{fin}})$ de la courbe $h_\theta$. Pour chaque combinaison, on calcule analytiquement l’amplitude optimale $A_i$ de chaque patient, puis la somme des erreurs au carré sur l’ensemble des paires éligibles. On retient finalement la combinaison $\theta$ qui minimise cette erreur.

Pour l'entrainement on utilise uniquement les visites où ON/OFF et leurs deux délais observés sont présents (20,1 % des visites).

Après entrainement,pour calculer les features finales ($A_i$, $A_i h_\theta(t_{ON/OFF})$) à ajouter au dataset, environ 40 % des patients nécessitent une amplitude commune de remplacement. Les features ajoutées avec un indicateur de remplacement n’améliorent pas HistGB (`all_groups`, avant optimisation) : MSE **14,7793 → 14,7848**.


##### Améliorations possibles :

Expected medication effect when the intake delay is missing

Intake delays are missing in 46.7% of training rows for ON and 78.8% for OFF. The current pipeline fills these with medians.

Rather than imputing one delay, estimate a plausible delay distribution from available patient history and pooled cohort data, then calculate:
- expected_drug_effect_on/off
- drug_effect_uncertainty_on/off
Average the effect over plausible delays: for a nonlinear response curve, the effect at the average delay can differ substantially from the average effect. This could preserve useful uncertainty that a single imputed time loses.

### Modeling
We keep the Histgb which performs the best. Here is a small ranking of all the models with optimized parameters:

#### Encoder
Une bonne inspiration ici est de combiner un transformer (qui réalise du très bon Structure Learning surr des données sequentielles) et le Histgb que nous utilisons maintenant.

Nous créons donc un encodeur à l'architecture simple. Je fais le choix d'un encodeur bidirectionnel car on peut dans le train comme dans test set accéder aux visites futures comme passées des patients. Cet encodeur prend end entrée les séquences des visites de patients en applicant du padding pour les séquences trop courtes et auxquelles ont a ajouté le temps depuis la première visite pour encoder la dimenion temporelle.
Il est entrainé à prédire les score off pour chaque visite. Voir architecture ci-dessous. Une fois ce modèle entrainé sur l'ensemble du dataset, on pass-forward l'ensemble des samples de notre trainset, extrait les embedddings de la sortie du dernier bloc d'attention pour les ajouter aux features précédemment travaillées dans features enginneering du histgb qui lui réalise la prédiciton finale. 

##### Les améliorations notables dans cette architecture:
-Optimisation des paramètres de l'encoder par optimisation bayesienne avec cross validation pour estimer la performance de chaque combinaison de paramètre + hold-out set pour choisir pour chaque test la meilleure valeure du nombre d'époch. -9% de MSE
-Use a schduler: Cosin annealing with warm restart -7% MSE


##### Ce que nous avons essayé et qui n'a pas amélioré la performance globale du modèle entier:

-Utiliser un token CLS (pas très utile mais curiosité) +1% de MSE
-Utiliser l'ensemble ou une partie des features crées durant le features engineering à l'entrée de l'encoder +1.8% MSE
-Encoder le temps écoulé depuis la première visite par un vecteur sinusoïdal de dimension 64 (sinus et cosinus à différentes fréquences), ajouté aux features des visites après leur projection dans l'espace de l'encodeur. MSE quasiment inchangée
-Utiliser les predicitons de score off prédites par l'ecodeur en plus des embeddings dans les features de histgb. Cependant cela nécessite de diviser le trainset en deux pour ne pas entrainer les modèles sur le même dataset. +5% de MSE
-Regression Ridge à la place de histgb +5% de MSE

Ce que nous pourions essayer différemment:

### Post-Processing

Les trajectoires cibles étant non décroissantes, nous avons appliqué un post-traitement aux prédictions finales, séparément pour chaque patient. Les modèles restent inchangés. Seules leurs prédictions sont ajustées selon l’âge des visites.

Deux méthodes ont été testées.

- Régression isotone, pour supprimer les baisses. -0.96% de MSE

- Lissage monotone, pour réduire les variations de pente en tenant compte des intervalles entre visites.
Le paramètre de lissage λ est optimisé par gridsearch avec cross validation à 5 folds.
 -2% de MSE


