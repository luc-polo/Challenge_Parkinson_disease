# Challenge Parkinson — prédiction du score moteur OFF

Ce projet vise à prédire le score moteur OFF corrigé pour chaque visite d'un patient dans le cadre du [challenge de l'Institut du Cerveau](https://challengedata.ens.fr/challenges/159). 

## Explication de la démarche

Un premier prétraitement a été réalisé : séparation entraînement/validation par patient, encodage des variables catégorielles et imputation par la médiane avec indicateurs de valeurs manquantes. Plusieurs modèles de régression ont ensuite été comparés sur la RMSE de validation, sans optimisation des hyperparamètres, afin d'établir un benchmark pour les prochaines améliorations et de produire une première soumission. La suite consistera à affiner le traitement des données, notamment avec des méthodes d'imputation plus adaptées aux variables et à l'historique des patients.
