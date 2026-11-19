"""Entraînement du modèle d'annulation v2 — Escale Atlantique (mission FastIA).

Tâche 9 : démarche finale mise au point dans les notebooks 01_exploration et 02_modelisation.

    python train.py

1. charge les réservations depuis PostgreSQL (une requête SQL avec jointure) ;
2. met de côté les 4 derniers mois d'arrivée comme jeu de test ;
3. choisit le seuil de décision sur les 4 derniers mois de l'entraînement (F1) ;
4. entraîne le pipeline v2 (variables nouvelles + préparation + forêt aléatoire réglée + seuil) ;
5. l'évalue une seule fois sur le test ;
6. sauvegarde model/cancellation_model_v2.joblib et sa fiche model/model_info_v2.json.
"""

import json
import os
import time
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import FixedThresholdClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine, text

from model import features

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TEST_MONTHS = 4
TARGET = "is_canceled"

# Hyperparamètres retenus après la recherche tracée du notebook 02_modelisation (tâche 7) :
# forêt aléatoire, validation croisée temporelle (TimeSeriesSplit), métrique F1.
BEST_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,         # profondeur illimitée : la taille des feuilles limite le surapprentissage
    "max_features": "sqrt",    # chaque nœud ne regarde qu'un sous-ensemble de variables
    "min_samples_leaf": 5,     # score proche du meilleur, écart train/validation réduit
}

load_dotenv(ROOT / ".env")

# Une seule requête : la jointure récupère le code de l'hôtel ('City Hotel' / 'Resort Hotel'),
# attendu sous le nom `hotel` par la recette v1 comme par la v2.
QUERY = """
SELECT h.code AS hotel,
       r.*
FROM reservations AS r
JOIN hotels AS h ON h.hotel_id = r.hotel_id
ORDER BY r.arrival_date, r.reservation_id
"""


def load_data(engine) -> pd.DataFrame:
    """Charge toutes les réservations (avec le code de l'hôtel), triées par date d'arrivée."""
    return pd.read_sql(text(QUERY), engine, parse_dates=["arrival_date", "reservation_status_date"])


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Découpage TEMPOREL : les TEST_MONTHS derniers mois d'arrivée forment le test.

    Le modèle prédira des réservations futures : on imite cet usage en entraînant sur le
    passé et en testant sur la période la plus récente (mini-cours 10, section 1).
    """
    date_limite = df["arrival_date"].max() - pd.DateOffset(months=TEST_MONTHS)
    train = df[df["arrival_date"] <= date_limite].reset_index(drop=True)
    test = df[df["arrival_date"] > date_limite].reset_index(drop=True)
    return train, test


def build_pipeline(params: dict | None = None, threshold: float = 0.5):
    """Pipeline v2 : variables nouvelles -> préparation -> forêt aléatoire.

    params    : hyperparamètres de la forêt (BEST_PARAMS par défaut).
    threshold : seuil de décision ; à 0,5 on renvoie le pipeline tel quel, sinon il est
                enveloppé dans un FixedThresholdClassifier (annulation prédite si proba >= seuil).
    """
    params = BEST_PARAMS if params is None else params
    classifier = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    pipeline = features.build_pipeline(classifier)
    if threshold == 0.5:
        return pipeline
    return FixedThresholdClassifier(pipeline, threshold=threshold, response_method="predict_proba")


def choose_threshold(train: pd.DataFrame, params: dict | None = None) -> tuple[float, pd.DataFrame]:
    """Choisit le seuil qui maximise le F1 sur les TEST_MONTHS derniers mois de l'entraînement.

    Le modèle est entraîné sur le reste de l'entraînement ; le jeu de test n'est pas utilisé.
    Renvoie le seuil retenu et la trace des seuils essayés.
    """
    debut, fin = split_train_test(train)
    model = build_pipeline(params)
    model.fit(debut[features.RAW_FEATURES], debut[TARGET])
    proba = model.predict_proba(fin[features.RAW_FEATURES])[:, 1]
    y = fin[TARGET]
    lignes = []
    for seuil in np.round(np.arange(0.20, 0.81, 0.05), 2):
        y_pred = (proba >= seuil).astype(int)
        lignes.append({
            "seuil": seuil,
            "f1": f1_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred),
        })
    trace = pd.DataFrame(lignes)
    return float(trace.loc[trace["f1"].idxmax(), "seuil"]), trace


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """Métriques de classification sur (X, y). Métrique principale : F1 (voir RESULTATS.md)."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    return {
        "f1": round(f1_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred), 4),
        "recall": round(recall_score(y, y_pred), 4),
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "roc_auc": round(float(roc_auc_score(y, y_proba)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main() -> None:
    engine = create_engine(os.environ["DATABASE_URL"])
    df = load_data(engine)
    train, test = split_train_test(df)
    print(f"Entraînement : {len(train)} réservations | test : {len(test)} réservations")

    # Seules les colonnes brutes connues à la réservation entrent dans le modèle
    # (les colonnes de fuite de features.COLONNES_FUITE en sont exclues).
    X_train, y_train = train[features.RAW_FEATURES], train[TARGET]
    X_test, y_test = test[features.RAW_FEATURES], test[TARGET]

    threshold, _ = choose_threshold(train)
    print(f"Seuil de décision retenu : {threshold}")

    model = build_pipeline(threshold=threshold)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = round(time.perf_counter() - start, 1)

    metrics = evaluate(model, X_test, y_test)
    print("Métriques sur le jeu de test :", metrics)

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "cancellation_model_v2.joblib"
    joblib.dump(model, model_path)
    info = {
        "model_name": "cancellation_model",
        "version": "2.0.0",
        "trained_on": date.today().isoformat(),
        "scikit_learn_version": sklearn.__version__,
        "features": list(X_train.columns),
        "engineered_features": features.NEW_NUMERICAL,
        "excluded_leaky_columns": features.COLONNES_FUITE,
        "target": TARGET,
        "main_metric": "f1",
        "decision_threshold": threshold,
        "train_period": [str(train["arrival_date"].min().date()), str(train["arrival_date"].max().date())],
        "test_period": [str(test["arrival_date"].min().date()), str(test["arrival_date"].max().date())],
        "train_size": len(train),
        "test_size": len(test),
        "hyperparameters": {
            k.removeprefix("estimator__"): v for k, v in model.get_params().items()
            if "classifier__" in k and isinstance(v, (int, float, str, bool, type(None)))
        },
        "test_metrics": metrics,
        "training_seconds": training_seconds,
        "model_size_mb": round(model_path.stat().st_size / 1e6, 2),
    }
    (MODEL_DIR / "model_info_v2.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Modèle et fiche sauvegardés dans", MODEL_DIR)


if __name__ == "__main__":
    main()
