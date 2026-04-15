import argparse

from src.pipelines.training_pipeline import run_training_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """
    Entry point do projeto.
    
    Permite executar o pipeline de treino via linha de comando.
    """
    parser = argparse.ArgumentParser(description="Pipeline de Treinamento - Churn Prediction")

    parser.add_argument(
        "--train",
        action="store_true",
        help="Executa o pipeline completo de treinamento"
    )

    args = parser.parse_args()

    if args.train:
        logger.info("Iniciando pipeline de treinamento...")
        run_training_pipeline()
        logger.info("Pipeline finalizado com sucesso.")
    else:
        logger.warning("Nenhuma ação especificada. Use --train para treinar o modelo.")


if __name__ == "__main__":
    main()