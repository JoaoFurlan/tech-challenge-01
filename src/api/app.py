from fastapi import FastAPI, HTTPException
from src.api.schemas import CustomerInput, PredictionOutput
from src.models.predict import predict_new_customer
from src.middleware.logger import get_logger
from src.config import CHURN_THRESHOLD

logger = get_logger(__name__)

app = FastAPI(
    title="Churn Prediction API",
    description="API para prever o cancelamento de clientes (Churn) de Telecom",
    version="1.0.0"
)

@app.get("/health", tags=["Health"])
def health_check():
    """Endpoint para verificar se a API está online."""
    logger.info("Health check acessado.")
    return {"status": "ok", "message": "API is running"}

@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict_churn(customer: CustomerInput):
    """Recebe dados de um cliente e retorna a probabilidade de Churn."""
    logger.info(f"Recebendo requisição de predição para novo cliente.")
    
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