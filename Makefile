.PHONY

help:
		@echo "Comandos disponíveis:"
		@echo " make install   -   Instala as dependências"
		@echo " make lint      -   Roda o linter (limpeza de código)"
		@echo " make test      -   Roda os testes (stand-by)"
		@echo " make train     -   Executa o pipeline de treinamento"

install: 
		pip install --upgrade pip
		pip install -e ".[dev]"

lint:	
		ruff check . --fix

test:
		pytest tests/