"""Variables et pipeline du modèle d'annulation v2.

Ce module est importé par les notebooks ET par train.py. Il doit aussi être importable
là où le modèle sera chargé (API) : joblib enregistre une *référence* vers
`ajouter_variables`, pas son code (cf. mini-cours 09, section 4).

Règle suivie pour chaque colonne : elle doit être CONNUE AU MOMENT OÙ LA RÉSERVATION
EST PRISE, c'est-à-dire au moment où le modèle fera sa prédiction.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "is_canceled"

# Colonnes qui trahissent la réponse ou ne sont connues qu'après coup (tâche 4).
# Elles ne sont jamais données au modèle v2.
COLONNES_FUITE = [
    "reservation_status",           # Canceled / Check-Out / No-Show : c'est la réponse elle-même
    "reservation_status_date",      # date d'annulation ou de départ : connue après l'événement
    "assigned_room_type",           # chambre attribuée à l'arrivée du client
    "booking_changes",              # nombre FINAL de modifications, s'accumule jusqu'au départ
    "required_car_parking_spaces",  # 0 annulation sur ~4 000 cas : saisi à l'arrivée (suspect)
]

# --- Colonnes brutes utilisées (toutes connues à la réservation) ---------------
RAW_CATEGORICAL = [
    "hotel",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
    "agent",        # identifiant numérique, mais c'est une catégorie (agence n°9 ≠ 9 × agence n°1)
]
RAW_NUMERICAL = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "days_in_waiting_list",
    "adr",
    "total_of_special_requests",
]
# arrival_date et company ne vont pas directement au modèle : on en tire des variables.
RAW_FEATURES = RAW_CATEGORICAL + RAW_NUMERICAL + ["arrival_date", "company"]

# --- Variables nouvelles, créées par ajouter_variables() ------------------------
NEW_NUMERICAL = [
    "total_nights",           # durée du séjour
    "total_guests",           # taille du groupe
    "has_children",           # enfants ou bébés présents
    "adr_per_guest",          # prix par nuit et par personne
    "total_price",            # montant total du séjour (adr × nuits)
    "has_special_request",    # au moins une demande particulière
    "has_previous_cancel",    # le client a déjà annulé au moins une fois
    "has_company",            # réservation payée par une entreprise
    "arrival_month",          # saisonnalité (1-12)
    "arrival_weekday",        # jour de la semaine d'arrivée (0 = lundi)
]
CATEGORICAL_FEATURES = RAW_CATEGORICAL
NUMERICAL_FEATURES = RAW_NUMERICAL + NEW_NUMERICAL


def ajouter_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Reçoit le DataFrame BRUT et renvoie une copie enrichie des variables nouvelles."""
    df = df.copy()  # ne jamais modifier l'entrée
    children = df["children"].fillna(0)
    nights = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    guests = df["adults"] + children + df["babies"]

    df["total_nights"] = nights
    df["total_guests"] = guests
    df["has_children"] = ((children + df["babies"]) > 0).astype(int)
    # adults + children : les bébés ne paient pas ; on évite la division par zéro
    df["adr_per_guest"] = df["adr"] / (df["adults"] + children).replace(0, np.nan)
    df["total_price"] = df["adr"] * nights
    df["has_special_request"] = (df["total_of_special_requests"] > 0).astype(int)
    df["has_previous_cancel"] = (df["previous_cancellations"] > 0).astype(int)
    df["has_company"] = df["company"].notna().astype(int)

    arrival = pd.to_datetime(df["arrival_date"])
    df["arrival_month"] = arrival.dt.month
    df["arrival_weekday"] = arrival.dt.dayofweek

    # agent : identifiant -> texte, pour qu'il soit traité comme une catégorie
    df["agent"] = df["agent"].astype("Int64").astype(str).replace("<NA>", "aucun")
    return df


def build_preprocessing(scale: bool = False) -> ColumnTransformer:
    """Préparation : imputation + encodage (+ mise à l'échelle pour les modèles linéaires)."""
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                    ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50)),
                ]),
                CATEGORICAL_FEATURES,
            ),
            ("num", Pipeline(num_steps), NUMERICAL_FEATURES),
        ],
        sparse_threshold=0,  # sortie dense (HistGradientBoosting n'accepte pas le sparse)
    )


def build_pipeline(classifier=None, scale: bool = False) -> Pipeline:
    """Pipeline complet v2 : données brutes -> variables -> préparation -> modèle."""
    if classifier is None:
        classifier = HistGradientBoostingClassifier(random_state=42)
    return Pipeline([
        ("variables", FunctionTransformer(ajouter_variables)),
        ("preprocessing", build_preprocessing(scale=scale)),
        ("classifier", classifier),
    ])
