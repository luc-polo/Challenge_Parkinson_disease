# Challenge Parkinson — prédiction du score moteur OFF

Ce projet vise à prédire le score moteur OFF corrigé pour chaque visite d'un patient dans le cadre du [challenge de l'Institut du Cerveau](https://challengedata.ens.fr/challenges/159).

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
- `on_median_patient`, `off_median_patient`, `ledd_median_patient` : médianes des valeurs du patient sur ses différentes visites.
- `on_slope_patient`, `off_slope_patient`, `ledd_slope_patient` : pentes annuelles des droites ajustées aux valeurs du patient en fonction de son âge.
- `off_on_median_patient` : médiane de l’écart `off - on` du patient.
- `disease_duration` : durée de la maladie, calculée par `age - age_at_diagnosis`.
- `ledd_x_time_on`, `ledd_x_time_off` : produits de la LEDD par le délai depuis la prise du traitement pour les mesures ON et OFF.
- `ledd_time_ratio_on`, `ledd_time_ratio_off` : rapports `ledd / (1 + time_since_intake_on)` et `ledd / (1 + time_since_intake_off)`.
- `on_trend_at_visit`, `off_trend_at_visit` : valeurs estimées à l’âge de la visite actuelle par les droites de tendance du patient.
- `on_previous`, `on_next`, `off_previous`, `off_next` : valeurs ON/OFF des visites précédente et suivante disponibles.
- `on_previous_gap`, `on_next_gap`, `off_previous_gap`, `off_next_gap` : écarts de temps entre la visite actuelle et les visites correspondant à ces valeurs voisines.
- `on_n_observed`, `off_n_observed`, `ledd_n_observed` : nombres de valeurs non manquantes par patient.
- `on_std_patient`, `off_std_patient` : écarts-types des valeurs ON et OFF du patient.

Ce que j'ai essayé mais qui n'a pas aidé le modèle:

J’ai essayé d’ajouter des variables de type `ledd*/time`, mais les performances du modèle se sont plutôt dégradées qu’améliorées.

J'ai tenté d'ajouter une features qui represente la concentration de ldd dans le sang au moment de la mesure du score off à prédire. J'ai réalisé plusieurs tentatives différentes: