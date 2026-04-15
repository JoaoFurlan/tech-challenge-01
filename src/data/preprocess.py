import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.logger import get_logger
from src.config import RANDOM_STATE, MODEL_DIR, RAW_DATA_PATH

logger = get_logger(__name__)

def load_data(path: str) -> pd.DataFrame:
    """Carrega o dataset bruto."""
    logger.info(f"Carregando dados de: {path}")
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executa a limpeza inicial:
    - Converte TotalCharges para numérico
    - Trata valores nulos
    - Remove CustomerID
    """
    df = df.copy()

    # 1. Converter TotalCharges (que vem como string/object) para float
    # O 'errors=coerce' transforma espaços vazios em NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # 2. Tratar valores nulos
    null_count = df['TotalCharges'].isnull().sum()
    if null_count > 0:
        logger.info(f"Preenchendo {null_count} valores nulos em TotalCharges com 0")
        df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # 3. Remover customerID
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # 4. Codificar variável alvo (Churn) para numérico
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    logger.info("Limpeza de dados concluída.")
    return df


def split_data(df: pd.DataFrame, target_column: str = 'Churn', test_size: float = 0.2):
    """ Divide os dados em treino e teste."""
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    logger.info(f"Dados divididos: Treino={X_train.shape}, Teste={X_test.shape}")
    return X_train, X_test, y_train, y_test



def encode_and_scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """ Aplica o OneHotEncoder (categóricas) e StandardScaler (numéricas)"""
    logger.info("Aplicando encoding e scaling...")

    # Separa colunas categórias e numéricas
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()

    # 1. Processar Categóricas
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_cat = encoder.fit_transform(X_train[categorical_cols])
    X_test_cat = encoder.transform(X_test[categorical_cols])
    cat_cols_names = encoder.get_feature_names_out(categorical_cols)

    # 2. Processar Numéricas
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[numerical_cols])
    X_test_num = scaler.transform(X_test[numerical_cols])

    # 3. Juntar tudo em DataFrames
    X_train_cat_df = pd.DataFrame(X_train_cat, columns=cat_cols_names, index=X_train.index)
    X_test_cat_df = pd.DataFrame(X_test_cat, columns=cat_cols_names, index=X_test.index)
    
    X_train_num_df = pd.DataFrame(X_train_num, columns=numerical_cols, index=X_train.index)
    X_test_num_df = pd.DataFrame(X_test_num, columns=numerical_cols, index=X_test.index)

    X_train_final = pd.concat([X_train_num_df, X_train_cat_df], axis=1)
    X_test_final = pd.concat([X_test_num_df, X_test_cat_df], axis=1)

    # 4. Salvar os artefatos para a API
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, MODEL_DIR / 'one_hot_encoder.joblib')
    joblib.dump(scaler, MODEL_DIR / 'scaler.joblib') # Salva o scaler também!
    
    return X_train_final, X_test_final



def prepare_data_pipeline(df: pd.DataFrame):
    """Pipeline completo: Limpa, divide e codifica os dados."""
    df_cleaned = clean_data(df)

    # Divide os dados antes do encoding para evitar data leakage
    logger.info("Dividindo dados em treino e teste...")
    X_train, X_test, y_train, y_test = split_data(df_cleaned)

    # Aplica o encoding
    X_train_ready, X_test_ready = encode_and_scale_data(X_train, X_test)

    logger.info(f"Pipeline concluído. Formato do Treino: {X_train_ready.shape}")
    return X_train_ready, X_test_ready, y_train, y_test