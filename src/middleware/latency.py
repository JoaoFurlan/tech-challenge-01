import time

from fastapi import Request

from src.middleware.logger import get_logger

logger = get_logger(__name__)

async def log_latency_middleware(request: Request, call_next):
    start_time = time.time()

    # Processa a requisição
    response = await call_next(request)

    process_time = time.time() - start_time
    # Loga o tempo de latência formatado
    logger.info(f"Path: {request.url.path} - Latency: {process_time:.4f}s")

    # Adiciona o tempo no cabeçalho da resposta
    response.headers["X-Process-Time"] = str(process_time)

    return response
