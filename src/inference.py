import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data import build_feature_matrix


def predict_titer(model: Pipeline, raw_df: pd.DataFrame) -> float:
    """Predict the final titer for a single experiment's long-form record.

    Collapses 'raw_df' into the wide feature matrix via
    :func:`src.data.build_feature_matrix`, reindexes it onto the model's
    'feature_names_in_', runs 'model.predict', and exponentiates the
    log-titer prediction back to the natural scale.

    Args:
        model: Fitted pipeline whose 'feature_names_in_' enumerates the
            features it was trained on. Predictions are on log 'Y:Titer'.
        raw_df: Long-form dataframe for a single experiment with the same
            'Z:'/'W:'/'X:' columns the model expects.

    Returns:
        Predicted titer in the natural scale.

    Raises:
        ValueError: If 'raw_df' is missing features the model requires.
    """
    features = build_feature_matrix(raw_df)
    feature_names = list(model.feature_names_in_)
    missing = [c for c in feature_names if c not in features.columns]
    if missing:
        raise ValueError(f"Payload is missing inputs required by the model: {missing}")

    X = features.reindex(columns=feature_names)
    log_titer = float(model.predict(X)[0])
    return float(np.exp(log_titer))
