import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import MODEL_DIR
from src.middleware.logger import get_logger

logger = get_logger(__name__)

def fit_transform_features(X: pd.DataFrame):
    """Fit + transform (treino)."""

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    scaler = StandardScaler()

    X_cat = encoder.fit_transform(X[categorical_cols])
    X_num = scaler.fit_transform(X[numerical_cols])

    # salvar artefatos
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, MODEL_DIR / "encoder.joblib")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")

    X_final = _combine_features(X, X_cat, X_num, encoder, categorical_cols, numerical_cols)

    feature_names = X_final.columns.tolist()
    joblib.dump(feature_names, MODEL_DIR / "feature_names.joblib")

    return X_final


def transform_features(X: pd.DataFrame):
    """Transform apenas (inferência/teste)."""

    encoder = joblib.load(MODEL_DIR / "encoder.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    expected_columns = joblib.load(MODEL_DIR / "feature_names.joblib")

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

    X_cat = encoder.transform(X[categorical_cols])
    X_num = scaler.transform(X[numerical_cols])

    X_final = _combine_features(X, X_cat, X_num, encoder, categorical_cols, numerical_cols)

    X_final = X_final.reindex(columns=expected_columns, fill_value=0)

    return X_final


def _combine_features(X, X_cat, X_num, encoder, cat_cols, num_cols):
    """Função interna para juntar features."""

    X_cat_df = pd.DataFrame(
        X_cat,
        columns=encoder.get_feature_names_out(cat_cols),
        index=X.index
    )

    X_num_df = pd.DataFrame(
        X_num,
        columns=num_cols,
        index=X.index
    )

    return pd.concat([X_num_df, X_cat_df], axis=1)
