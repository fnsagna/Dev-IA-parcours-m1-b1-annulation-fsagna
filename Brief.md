### Contexte du projet

Lundi 9h, en centre de formation. Karim a réservé une salle pour la journée :

\=> Ce que tu produis : une base PostgreSQL (fournie) → des notebooks d'exploration → un script train.py → model/cancellation_model_v2.joblib, sa fiche model_info_v2.json et RESULTATS.md.

🎯 Objectifs pédagogiques

À la fin de ce brief, tu seras capable de :

-   Interroger une base de données en SQL et charger le résultat dans pandas.
-   Découper les données de façon cohérente avec l'usage réel du modèle.
-   Construire un pipeline scikit-learn qui prépare les données et crée de nouvelles variables.
-   Valider un modèle par validation croisée et régler ses hyperparamètres.
-   Détecter le surapprentissage et une fuite de données.
-   Comparer deux modèles avec des métriques adaptées au métier, et documenter les résultats.

🧰 Tâches

1.  Démarrer : docker, dépendances, puis ouvre le notebook : la première cellule affiche le nombre de réservations.
2.  Explorer en SQL (mini-cours 08). Par des requêtes SQL : réservations et taux d'annulation par hôtel (avec son nom, table hotels), selon le mois d'arrivée, selon le délai de réservation, par segment et par type d'acompte. Note ce qui te surprend.
3.  Découper (mini-cours 10, section 1). Charge les réservations en une requête avec la jointure. Le modèle prédira des réservations futures : mets de côté les 4 derniers mois d'arrivée comme jeu de test, et n'y touche plus avant la tâche 8.
4.  Un premier modèle. Entraîne un modèle simple (par exemple une forêt aléatoire) sur toutes les colonnes, encodées sommairement avec pd.get\_dummies, dates exclues. Regarde le score, puis les variables qui comptent le plus. Si quelque chose te paraît anormal, enquête : à quel moment chaque information est-elle connue dans la vie d'une réservation ?
5.  Reproduire la v1 (mini-cours 10, exercice guidé). La v1 a été entraînée sur des données qui recouvrent ton test : la comparer telle quelle serait injuste. Rejoue sa recette (build_pipeline() de model/train_baseline.py) sur ton jeu d'entraînement, et mesure-la par validation croisée. C'est ta référence à battre.
6.  Ton pipeline (mini-cours 09). Un Pipeline qui prend les données brutes en entrée, comme celui de la v1 : préparation (encodage, valeurs manquantes, mise à l'échelle si besoin), nouvelles variables issues de ton exploration (durée du séjour, présence d'enfants…), puis un modèle. Compare plusieurs familles de modèles.
7.  Valider et régler (mini-cours 10). Validation croisée sur l'entraînement, réglage des principaux hyperparamètres avec une trace des essais, comparaison du score d'entraînement et du score de validation. Un grand écart ? Ajuste, et explique.
8.  Le verdict : Choisis ta métrique en te demandant ce que coûte à l'hôtel une annulation prédite à tort, et une annulation ratée. Puis évalue une seule fois la v1 rejouée et ta v2 sur le test : métriques, matrices de confusion, temps d'entraînement, taille du modèle.
9.  Script et documentation : Reporte ta démarche finale dans train.py : il produit le modèle et sa fiche (version, colonnes, hyperparamètres, métriques). Rédige RESULTATS.md : démarche, fuite de données trouvée, tableau v1 / v2, etc....

### Modalités pédagogiques

Format : 6 h en synchrone.

Outils : Docker (PostgreSQL), Python, Jupyter, pandas, SQLAlchemy, scikit-learn, Git.

Ressources : le squelette du repo et les mini-cours 08 (SQL), 09 (pipelines) et 10 (validation).

### Modalités d'évaluation

-   1\. Restitution orale en fin de journée (5 min par personne) :
-   \- ce que tu as trouvé en tâche 4, et comment tu l'as trouvé ;
-   \- tes deux ou trois meilleures variables nouvelles, et pourquoi elles ont un sens métier ;
-   \- le tableau v1 / v2, et la métrique que tu as choisie.
-   2\. Évaluation sur le repo, .
-   \- Compétence visée — C5, niveau 1 (imiter): entraîner le modèle d'IA de façon supervisée pour valider la pertinence de la solution au regard du cas d'usage métier.
-   \- Compétences transversales observées, hors certification : CT4 (recherche méthodique de solutions) et CT5 (documentation et partage).

### Critères de performance

-   1 : Les données sont extraites par une requête SQL, et le jeu de test correspond aux 4 derniers mois de l'historique, sans être utilisé avant l'évaluation finale.
-   2 : Les colonnes qui trahissent la réponse sont identifiées, retirées, et l'explication figure dans RESULTATS.md.
-   3: Le pipeline prend les données brutes en entrée et contient au moins deux variables nouvelles justifiées par le métier.
-   4 : Le modèle est évalué par validation croisée, et au moins un hyperparamètre est réglé de façon tracée.
-   5 : L'écart entre le score d'entraînement et le score de validation est mesuré et commenté.
-   6 : La v1 rejouée et la v2 sont comparées sur le même jeu de test, avec une métrique dont le choix est justifié par le métier.
-   7 : train.py s'exécute de bout en bout et produit le modèle et sa fiche model\_info\_v2.json.

## Livrables

3 à rendre

1.  Le lien du repo GitHub \`M1-B1-annulation-v2-<prénom>\`
2.  le ou les notebooks d'exploration
3.  RESULTATS.md