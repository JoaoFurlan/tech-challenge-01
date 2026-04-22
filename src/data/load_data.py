import pandas as pd

from src.middleware.logger import get_logger

logger = get_logger(__name__)

def load_data(path: str) -> pd.DataFrame:
    """Carrega o dataset bruto."""
    logger.info(f"Carregando dados de: {path}")
    return pd.read_csv(path)
