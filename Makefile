.PHONY: help install lint test train run

# Comando padrão ao digitar apenas 'make'
.DEFAULT_GOAL := help

help:
	@echo "Comandos disponíveis:"
	@echo "  make install   - Instala as dependências e o pacote em modo editável"
	@echo "  make lint      - Roda o ruff para formatar e limpar o código"
	@echo "  make test      - Executa a suíte de testes com pytest"
	@echo "  make train     - Executa o pipeline de treinamento (main.py)"
	@echo "  make run       - Inicia a API FastAPI localmente"

install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"

lint:
	python -m ruff check . --fix
	python -m ruff format .

test:
	python -m pytest tests/

train:
	python main.py --train

run:
	python -m uvicorn src.api.app:app --reload