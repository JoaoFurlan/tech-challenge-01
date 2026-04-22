import logging
import sys
import os

# Definição de cores ANSI
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
CYAN = "\033[36m"

def get_logger(name: str):
    logger = logging.getLogger(name)

    # Silencia warnings do MLflow e outras bibliotecas no terminal
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        # Propagate=False evita que o log suba para o root e duplique a mensagem
        logger.propagate = False

        # 1. FORMATADORES
        # Formato com cores para o terminal (melhor para leitura)
        shell_formatter = logging.Formatter(
            f"{CYAN}%(asctime)s{RESET} - [%(name)s] - {BOLD}%(levelname)-8s{RESET} - %(message)s", datefmt="%H:%M:%S"
        )

        # Formato limpo para o arquivo (sem códigos de cor ANSI)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )


        # 2. HANDLER TERMINAL
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(shell_formatter)
        logger.addHandler(stream_handler)


        # 3. HANDLER ARQUIVO
        log_dir = os.path.join("reports", "logs")
        os.makedirs(log_dir, exist_ok=True)


        file_handler = logging.FileHandler(
                                    os.path.join(log_dir, "api.log"),
                                    mode="a", 
                                    encoding="utf-8"
                                    )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
