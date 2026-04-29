FROM python:3.11-slim

WORKDIR /app

# 1. Instala dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Otimização de Cache: Instala o PyTorch CPU primeiro
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 3. Copia TODOS os arquivos necessários para a instalação do pacote
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY data/ ./data/

# 4. Instala o seu projeto e as outras dependências (pandas, mlflow, etc)
RUN pip install --no-cache-dir .

# 5. Copia o restante dos arquivos (testes, configs, modelos salvos, etc)
COPY . .


CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]