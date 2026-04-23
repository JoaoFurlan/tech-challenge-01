import pandas as pd
import pandera.pandas as pa

from src.middleware.logger import get_logger

logger = get_logger(__name__)


# Definição do Schema Completo para validação dos dados de treino
CHURN_SCHEMA = pa.DataFrameSchema({
    "customerID": pa.Column(str),
    "gender": pa.Column(str, checks=pa.Check.isin(["Female", "Male"])),
    "SeniorCitizen": pa.Column(int, checks=pa.Check.isin([0, 1])),
    "Partner": pa.Column(str, checks=pa.Check.isin(["Yes", "No"])),
    "Dependents": pa.Column(str, checks=pa.Check.isin(["Yes", "No"])),
    "tenure": pa.Column(int, checks=pa.Check.ge(0)),
    "PhoneService": pa.Column(str),
    "MultipleLines": pa.Column(str),
    "InternetService": pa.Column(str),
    "OnlineSecurity": pa.Column(str),
    "OnlineBackup": pa.Column(str),
    "DeviceProtection": pa.Column(str),
    "TechSupport": pa.Column(str),
    "StreamingTV": pa.Column(str),
    "StreamingMovies": pa.Column(str),
    "Contract": pa.Column(str),
    "PaperlessBilling": pa.Column(str),
    "PaymentMethod": pa.Column(str),
    "MonthlyCharges": pa.Column(float),
    "TotalCharges": pa.Column(object), # Mantido como object porque o CSV bruto tem espaços vazios
    "Churn": pa.Column(str, checks=pa.Check.isin(["Yes", "No"]), required=False) # 'required=False' caso use para predição real
})



def load_data(path: str) -> pd.DataFrame:
    """
    Carrega o dataset e valida o schema completo.
    """
    logger.info(f"Carregando e validando dados de: {path}")
    try:
        df = pd.read_csv(path)

        validated_df = CHURN_SCHEMA.validate(df)

        logger.info("Validação de schema (Pandera) concluída com sucesso.")
        return validated_df

    except pa.errors.SchemaError as e:
        logger.error(f"Falha na integridade dos dados: {e}")
        raise
    except Exception as e:
        logger.error(f"Erro ao carregar o arquivo: {e}")
    raise    
    
