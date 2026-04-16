import logging
import sys

def get_logger(name: str):
    logger = logging.getLogger(name)

    # Silencia warnings do MLflow e outras bibliotecas no terminal
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        # Propagate=False evita que o log suba para o root e duplique a mensagem
        logger.propagate = False

        # Formato: Data - Nome do Arquivo - Nível - Mensagem
        formatter = logging.Formatter(
            "%(asctime)s - [%(name)s] - %(levelname)-8s - %(message)s",
            datefmt="%H:%M:%S"
        )

        # Log no terminal
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
