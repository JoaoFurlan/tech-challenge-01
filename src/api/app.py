from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException, Request
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.schemas import CustomerInput, PredictionOutput
from src.config import CHURN_THRESHOLD, MODEL_DIR
from src.middleware.latency import log_latency_middleware
from src.middleware.logger import get_logger
from src.models.predict import load_model_in_memory, predict_new_customer

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # O que acontece ANTES da API começar a rodar (Startup)
    logger.info("Iniciando API: Carregando modelo em memória (Singleton)...")
    try:
        expected_columns = joblib.load(MODEL_DIR / "feature_names.joblib")
        input_dim = len(expected_columns)
        load_model_in_memory(input_dim)
        logger.info(f"Modelo carregado com sucesso. Dimensão: {input_dim}")
    except Exception as e:
        logger.error(f"Erro crítico ao carregar modelo: {e}")

    yield  # Aqui a API "roda"

    # O que acontece quando a API é DESLIGADA (Shutdown)
    logger.info("Encerrando API...")

app = FastAPI(
    title="Churn Prediction API",
    description="API para prever o cancelamento de clientes (Churn) de Telecom",
    version="1.0.0",
    lifespan=lifespan
)


Instrumentator().instrument(app).expose(app)

# Registrar o middleware
@app.middleware("http")
async def add_latency_middleware(request: Request, call_next):
    return await log_latency_middleware(request, call_next)



@app.get("/health", tags=["Health"])
def health_check():
    """Endpoint para verificar se a API está online."""
    logger.info("Health check acessado.")
    return {"status": "ok", "message": "API is running"}



@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict_churn(customer: CustomerInput):
    """Recebe dados de um cliente e retorna a probabilidade de Churn."""
    logger.info("Recebendo requisição de predição para novo cliente.")

    try:
        # Converter o modelo Pydantic para um dicionário normal
        customer_dict = customer.model_dump()

        # Chamar a sua função de inferência já existente
        probabilidade = predict_new_customer(customer_dict)

        # Definir a classe (0 ou 1) baseado no threshold
        limite_decisao = CHURN_THRESHOLD
        predicao = 1 if probabilidade >= limite_decisao else 0
        mensagem = "Provável Churn" if predicao == 1 else "Fiel"

        logger.info(f"Predição realizada: {mensagem} (Prob: {probabilidade:.2%})")

        return PredictionOutput(
            churn_probability=probabilidade,
            churn_prediction=predicao,
            message=mensagem
        )

    except Exception as e:
        logger.error(f"Erro ao realizar predição: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno durante a predição.")
