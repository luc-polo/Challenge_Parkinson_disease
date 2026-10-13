# Challenge Parkinson — prédiction du score moteur OFF

Ce projet vise à prédire le score moteur OFF corrigé pour chaque visite d'un patient dans le cadre du [challenge de l'Institut du Cerveau](https://challengedata.ens.fr/challenges/159). 

## Explication de la démarche

### Premier Benchmark
Un premier prétraitement a été réalisé : séparation entraînement/validation par patient, encodage des variables catégorielles et imputation par la médiane avec indicateurs de valeurs manquantes. Plusieurs modèles de régression ont ensuite été comparés sur la RMSE de validation, sans optimisation des hyperparamètres, afin d'établir un benchmark pour les prochaines améliorations et de produire une première soumission. Il en ressort que le meilleur modèles est un modèle de Gradient Boosting:

First Benchmark results:
LR: MSE 72.681 ± 1.229
XGBoost: MSE 56.585 ± 0.748
SVM: MSE 59.266 ± 1.040
HistGB: MSE 55.039 ± 0.728
RandomForest: MSE 58.789 ± 0.988


La suite consistera à affiner le traitement des données, notamment avec des méthodes d'imputation plus adaptées aux variables et à l'historique des patients.
