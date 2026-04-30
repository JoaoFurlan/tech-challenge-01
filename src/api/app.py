from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException, Request
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.schemas import CustomerInput, PredictionOutput
from src.config import CHURN_THRESHOLD, MODEL_DIR, RAW_DATA_PATH
from src.data.load_data import load_data
from src.middleware.latency import log_latency_middleware
from src.middleware.logger import get_logger
from src.models.predict import load_model_in_memory, predict_new_customer

_TEST_DATA = None
logger = get_logger(__name__)

# Metadados para as tags do Swagger
tags_metadata = [
    {
        "name": "Health",
        "description": "Endpoints de diagnóstico para verificar a saúde"
        " e disponibilidade operacional da API.",
    },
    {
        "name": "Prediction",
        "description": "Endpoints principais de inferência. Utiliza a rede"
        " neural treinada para prever a probabilidade de churn.",
    },
    {
        "name": "Utils",
        "description": "Ferramentas auxiliares para extração de amostras"
        " reais do dataset e validação rápida do fluxo de dados.",
    },
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _TEST_DATA
    logger.info("Iniciando API: Carregando modelo em memória (Singleton)...")
    try:
        # Carregamento do modelo e metadados de features
        expected_columns = joblib.load(MODEL_DIR / "feature_names.joblib")
        input_dim = len(expected_columns)
        load_model_in_memory(input_dim)
        logger.info(f"Modelo carregado com sucesso. Dimensão: {input_dim}")

        # Carregamento do dataset para o endpoint de random_customer
        _TEST_DATA = load_data(str(RAW_DATA_PATH))
        if _TEST_DATA is not None:
            logger.info(f"Dataset carregado para shuffle: {_TEST_DATA.shape[0]} linhas.")
        else:
            logger.error("Dataset carregado retornou None.")
    except Exception as e:
        logger.error(f"Erro CRÍTICO ao carregar modelo: {e}")

    yield
    logger.info("Encerrando API...")

app = FastAPI(
    title="Churn Prediction API - Telco Customer Churn",
    description="""
## API para Previsão de Churn de Clientes.
Esta API utiliza um modelo de Rede Neural (MLP) treinado em PyTorch para identificar
clientes com alto risco de cancelamento.

### Funcionalidades:
* **Previsão**: Avalia o risco de churn com base em perfil demográfico e financeiro.
* **Monitoramento**: Expõe métricas para Prometheus e logs estruturados.
* **Apoio ao Teste**: Permite sortear clientes reais da base para validar o modelo.

**Regra de Negócio**: O limite de decisão (Threshold) atual é de **0.3**, priorizando o Recall.
    """,
    version="1.0.0",
    openapi_tags=tags_metadata,
    lifespan=lifespan
)

# Configuração do Instrumentator (Prometheus)
Instrumentator().instrument(app).expose(app)

# Registro de Middlewares
@app.middleware("http")
async def add_latency_middleware(request: Request, call_next):
    return await log_latency_middleware(request, call_next)

@app.get("/health", tags=["Health"])
def health_check():
    """
    ### Verificar Disponibilidade
    Endpoint simples para conferir se o serviço está operando normalmente.
    - **Retorno**: JSON com status 'ok'.
    """
    logger.info("Health check acessado.")
    return {"status": "ok", "message": "API is running"}

@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict_churn(customer: CustomerInput):
    """
    ### Realizar Predição de Churn
    Envia os dados de um cliente para a rede neural e recebe a classificação.

    **Parâmetros de Entrada**:
    - Perfil Demográfico (gender, SeniorCitizen, etc.)
    - Perfil de Serviço (InternetService, TechSupport, etc.)
    - Perfil Financeiro (MonthlyCharges, TotalCharges, etc.)

    **Retorno**:
    - `churn_probability`: Probabilidade calculada pela rede neural (0 a 1).
    - `churn_prediction`: 1 para Churn (se Prob >= 0.3), 0 para Fiel.
    - `message`: Descrição legível do resultado.
    """
    logger.info("Recebendo requisição de predição.")
    try:
        customer_dict = customer.model_dump()
        probabilidade = predict_new_customer(customer_dict)

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

@app.get("/random_customer", response_model=CustomerInput, tags=["Utils"])
def get_random_customer():
    """
    ### Obter Cliente Aleatório
    Retorna os dados de um cliente real extraído do dataset original (`Telco-Customer-Churn.csv`).

    **Caso de Uso**:
    Ideal para testar o endpoint `/predict` rapidamente copiando o JSON retornado aqui e
    colando no corpo da requisição de predição.
    """
    global _TEST_DATA
    if _TEST_DATA is None:
        logger.error("Tentativa de shuffle mas _TEST_DATA está None.")
        raise HTTPException(status_code=500, detail="Base de dados não carregada.")

    try:
        sample = _TEST_DATA.sample(n=1).to_dict(orient="records")[0]

        # Tratamento rápido para inconsistências comuns do CSV bruto (espaços vazios)
        if sample.get("TotalCharges") == " ":
            sample["TotalCharges"] = "0"

        return sample
    except Exception as e:
        logger.error(f"Erro ao sortear cliente: {e}")
        raise HTTPException(status_code=500, detail="Erro interno ao processar sorteio.")
