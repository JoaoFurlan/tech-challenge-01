Tech Challenge 01 — Previsão de Churn
=====================================

Este projeto consiste em uma solução end-to-end para previsão de churn em uma operadora de telecomunicações. O objetivo é identificar clientes com alto risco de cancelamento utilizando uma rede neural MLP (Multi-Layer Perceptron) desenvolvida em PyTorch, integrada com rastreamento de experimentos, conteinerização e monitoramento.

Estrutura do Projeto
--------------------
```
tech-challenge-01
├── data/               # Datasets brutos e processados
├── docs/               # Documentação adicional e Model Card
├── models/             # Artefatos e modelos treinados (.pt, .joblib)
├── notebooks/          # Análise exploratória (EDA) e experimentos
├── src/                # Código-fonte modularizado
│   ├── api/            # Endpoints FastAPI
│   ├── data/           # Scripts de carga e limpeza
│   ├── features/       # Engenharia de atributos
│   ├── middleware/     # Utilitários de logging (Middleware)
│   ├── models/         # Arquitetura e avaliação da rede neural
│   ├── pipelines/      # Orquestração do treino
│   └── utils/          # Helpers e ferramentas de treino
├── tests/              # Testes unitários e de integração
├── Dockerfile          # Configuração da imagem Docker
├── docker-compose.yml  # Orquestração da API e monitoramento
├── Makefile            # Atalhos para comandos comuns
└── pyproject.toml      # Configuração de dependências e ferramentas
```



Instalação e Setup
------------------

1.  `git clone https://github.com/JoaoFurlan/tech-challenge-01`
    
2.  `make install` Ou manualmente: `pip install -e ".\[dev\]"`
    

Execução do Projeto
-------------------

### Treinamento do Modelo
O pipeline realiza a carga, pré-processamento, treino com Early Stopping e registro no MLflow.
`make train`

### API de Inferência (FastAPI)
Para rodar o serviço localmente: `uvicorn src.api.app:app --reload   `

Acesse o Swagger em: http://localhost:8000/docs.
### Testes e Qualidade

Para rodar os testes automatizados e o linter:
`make test  make lint`

Uso com Docker
--------------

O projeto está configurado para subir a API e a infraestrutura de monitoramento (Prometheus e Grafana).

1.  `docker-compose up --build`
    
2.  **Acessos:**
    
    *   **API:** http://localhost:8000
        
    *   **Prometheus:** http://localhost:9090
        
    *   **Grafana:** http://localhost:3000
        

Monitoramento e Experimentos
----------------------------

*   **MLflow:** Para visualizar métricas e artefatos do treino, execute `mlflow ui`.
    
*   **Métricas:** A API expõe métricas de negócio e performance no endpoint /metrics, consumidas pelo Prometheus para visualização no Grafana.