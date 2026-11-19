# Modèle d'annulation v2 — Résultats

## Démarche

1. Charger les 78 703 réservations depuis PostgreSQL, avec une requête SQL.
2. Garder les 4 derniers mois comme jeu de test, utilisé une seule fois à la fin.
3. Retirer les colonnes qui donnent la réponse.
4. Rejouer la v1 sur les mêmes données, comme référence.
5. Construire le pipeline v2 et comparer trois modèles.
6. Régler le meilleur modèle, puis le seuil de décision.
7. Comparer la v1 et la v2 sur le jeu de test.

Modèle retenu : forêt aléatoire (200 arbres, profondeur illimitée, `min_samples_leaf=5`), 3 variables nouvelles, seuil 0,35.

Fichiers : `notebooks/01_exploration.ipynb` (tâches 1 à 3), `notebooks/02_modelisation.ipynb` (tâches 4 à 8), `train.py` (script final).

## Données

| | Période d'arrivée | Réservations | Taux d'annulation |
|---|---|---|---|
| Entraînement | 2015-07-01 → 2016-08-31 | 58 792 | 35,6 % |
| Test | 2016-09-01 → 2016-12-31 | 19 911 | 38,0 % |

Le découpage suit le temps, car le modèle devra prédire des réservations futures.

## Fuite de données (tâche 4)

Un premier modèle sur toutes les colonnes obtenait 100 % de bonnes réponses : c'est impossible. Les variables qui comptaient le plus venaient de `reservation_status`, le statut final de la réservation (annulée, venue, absente). Cette colonne contient la réponse.

Pour chaque colonne, j'ai vérifié si elle est connue au moment de la réservation. Cinq colonnes ne le sont pas et ont été retirées :

| Colonne | Pourquoi elle est retirée |
|---|---|
| `reservation_status` | statut final : c'est la réponse |
| `reservation_status_date` | date de ce statut final |
| `assigned_room_type` | chambre attribuée à l'arrivée du client |
| `booking_changes` | nombre de modifications, mis à jour après la réservation |
| `required_car_parking_spaces` | aucune réservation avec parking n'est annulée : l'information semble saisie à l'arrivée |

Liste dans `model/features.py` (`COLONNES_FUITE`).

## Variables nouvelles

Trois variables sont calculées dans le pipeline (`ajouter_variables` dans `model/features.py`), à partir de colonnes connues à la réservation.

| Variable | Calcul | Sens métier |
|---|---|---|
| `has_previous_cancel` | le client a déjà annulé | il annule à 96 %, contre 29 % pour les autres |
| `has_special_request` | le client a fait une demande particulière | il s'engage dans son séjour : 17 % d'annulation contre 46 % |
| `total_price` | prix par nuit × nombre de nuits | montant engagé : un séjour cher est plus souvent comparé et annulé |

Sept autres variables avaient été testées, puis désactivées (en commentaire dans le code) : durée du séjour, nombre de voyageurs, présence d'enfants, prix par personne, réservation d'entreprise, mois et jour d'arrivée. Avec seulement les trois variables ci-dessus, le modèle est meilleur et plus stable :

| | 10 variables | 3 variables |
|---|---|---|
| F1 validation (forêt, par défaut) | 0,721 ± 0,122 | **0,747 ± 0,079** |
| F1 sur le test (modèle final) | 0,726 | **0,733** |

Les variables retirées ajoutaient surtout du bruit. Le mois et le jour d'arrivée, par exemple, apprenaient des particularités de certaines périodes qui ne se répètent pas d'une année à l'autre.

## Validation et réglage

**Validation croisée temporelle.** Chaque fold apprend sur le passé et valide sur la période suivante, comme pour le jeu de test. Une validation aléatoire est trop optimiste : la v1 y obtient un F1 de 0,850, contre 0,701 en validation temporelle.

**Comparaison des modèles** (F1, validation temporelle) :

| Modèle | Validation | Entraînement |
|---|---|---|
| v1 (gradient boosting) | 0,701 | 0,924 |
| v2 régression logistique | 0,692 | 0,797 |
| v2 forêt aléatoire | **0,747** | 0,987 |
| v2 gradient boosting | 0,746 | 0,902 |

La forêt aléatoire est la meilleure, de très peu devant le gradient boosting. Elle surapprend : 0,987 en entraînement contre 0,747 en validation.

**Réglage** (`GridSearchCV`, trace dans `notebooks/traces/grille_foret_f1.csv`) : 9 combinaisons de `max_depth` (profondeur des arbres) et `min_samples_leaf` (taille minimum des feuilles).

| Réglage | Entraînement | Validation | Écart |
|---|---|---|---|
| par défaut / meilleur score (profondeur illimitée, feuilles de 1) | 0,987 | 0,747 | 0,240 |
| profondeur 20, feuilles de 1 | 0,936 | 0,745 | 0,191 |
| **retenu (profondeur illimitée, feuilles de 5)** | 0,885 | 0,738 | **0,147** |
| trop simple (profondeur 10, feuilles de 1) | 0,784 | 0,634 | 0,150 |

Le réglage retenu a presque le même score que le meilleur (0,738 contre 0,747), mais surapprend nettement moins. Règle appliquée : parmi les scores à moins de 0,01 du meilleur, prendre celui qui a le plus petit écart.

## Choix de la métrique

| Erreur | Ce qui se passe | Coût pour l'hôtel |
|---|---|---|
| Annulation ratée | le modèle dit « le client viendra », mais il annule | une chambre vide, une nuit non vendue |
| Fausse alerte | le modèle dit « le client va annuler », mais il vient | élevé si l'hôtel a revendu la chambre (client à reloger), faible s'il l'a seulement relancé |

Métrique principale : **F1**, qui équilibre les deux erreurs. On surveille le rappel (part des annulations détectées), point faible de la v1.

**Seuil.** Le modèle donne une probabilité d'annulation. Par défaut, il annonce une annulation à partir de 0,5. Le seuil a été choisi sur les 4 derniers mois de l'entraînement, jamais sur le test : **0,35** donne le meilleur F1.

## Verdict : v1 contre v2, sur le jeu de test

| | v1 | v2, seuil 0,5 | v2, seuil 0,35 (retenue) |
|---|---|---|---|
| **F1** | 0,575 | 0,680 | **0,733** |
| Précision (alertes justes) | 0,924 | 0,912 | 0,871 |
| Rappel (annulations détectées) | 0,418 | 0,542 | **0,633** |
| ROC AUC (qualité du classement) | 0,867 | 0,906 | 0,906 |
| Temps d'entraînement | 5,6 s | 8,9 s | 8,7 s |
| Taille du modèle | 0,7 Mo | 34,8 Mo | 34,8 Mo |

Matrices de confusion (lignes : réalité ; colonnes : prédiction) :

| v1 | annulée | maintenue |
|---|---|---|
| annulée | 3 162 | 4 408 |
| maintenue | 262 | 12 079 |

| v2, seuil 0,35 | annulée | maintenue |
|---|---|---|
| annulée | 4 789 | 2 781 |
| maintenue | 711 | 11 630 |

**Ce qu'il faut retenir :**
- La v2 détecte 4 789 annulations, contre 3 162 pour la v1.
- En contrepartie, elle fait plus de fausses alertes : 711 contre 262.
- La v2 bat la v1 même au seuil par défaut (F1 0,680 contre 0,575). Le réglage du seuil ajoute encore du rappel.
- La v2 classe mieux les réservations (ROC AUC 0,906 contre 0,867).
- La v2 n'utilise que des informations connues à la réservation. La v1 utilise le parking, probablement connu seulement à l'arrivée.
- La v2 est environ 50 fois plus lourde.

## Recommandation

Mettre la v2 en service à la place de la v1. Elle détecte nettement plus d'annulations, avec des données disponibles au moment de la réservation.

Points de vigilance :
- **Fausses alertes** : si l'hôtel revend systématiquement la chambre sur alerte, relever le seuil.
- **Ré-entraînement** : ré-entraîner sur tout l'historique avant la mise en service.
- **Parking** : confirmer avec l'hôtel le moment où cette information est saisie.
- **Taille** : réduire le nombre d'arbres si les 35 Mo posent problème.
- **Suivi** : surveiller le rappel une fois en production.
