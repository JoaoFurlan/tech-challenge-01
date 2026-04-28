Tech Challenge 01 — Previsão de Churn End-to-End
================================================

Este projeto apresenta uma solução profissional de Machine Learning para prever o cancelamento de clientes (churn). A arquitetura foi desenhada seguindo os princípios de **Engenharia de Software e MLOps**, utilizando ***PyTorch*** para a rede neural e ***FastAPI*** para o serviço de predição.

🎯 Contexto de Negócio (Método STAR)
------------------------------------
- **Situation**: Operadoras de telecomunicações enfrentam alta rotatividade de clientes. Identificar o churn antes que ele ocorra é vital para a saúde financeira.

- **Task**: Criar um modelo preditivo robusto (MLP), um pipeline de dados reprodutível e uma API monitorada.

- **Action**: Realizada EDA profunda para identificar correlações (como o impacto do tipo de contrato e cobranças eletrônicas), desenvolvimento de rede neural com Early Stopping e containerização com Docker.

- **Result**: Modelo MLP com AUC-ROC de 0.86 e threshold ajustado para 0.3 para priorizar o Recall (identificar mais clientes em risco).

___

🏗️ Estrutura do Projeto
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

___

⚙️ Diferenciação de Pipelines
------------------------------
Para atender aos requisitos técnicos, o projeto separa:
- **Pipeline de Dados (Orquestração)**: O fluxo modular que carrega (`load_data.py`), limpa (`preprocess.py`) e transforma os dados. A integridade é garantida pelo *Pandera*.
- **Pipeline de Processamento (Inferência)**: O uso de artefatos (`joblib`) garante que a API aplique exatamente a mesma normalização (*StandardScaler*) e codificação (*OneHotEncoder*) usada no treino, evitando o training-serving skew.
___

📊 Ciclo de Treinamento e Avaliação
--------------------------------
O treinamento da rede neural segue um protocolo rigoroso de divisão de dados para garantir que as métricas reportadas reflitam a performance real do modelo em produção.
### 1. Metodologia de Divisão (Split Triplo)
Os dados são segmentados em três conjuntos distintos:
- **Treino (64%)**: Utilizado pelo otimizador para ajuste dos pesos da MLP.
- **Validação (16%)**: Utilizado exclusivamente pelo *Early Stopping* para monitorar a perda e interromper o treino no momento ideal (neste caso, na época 13).
- **Teste (20%)**: Conjunto de **Holdout** isolado, utilizado apenas para a geração das métricas finais apresentadas abaixo.

### 2. Performance no Conjunto de Teste (Inédito)
As métricas abaixo foram extraídas após a aplicação do **Threshold de 0.3**, priorizando a sensibilidade do modelo na identificação de clientes em risco:

| Métrica | Valor (Teste) | Impacto de Negócio |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.8420** | Alta capacidade de distinção entre clientes fiéis e potenciais cancelamentos. |
| **Recall** | **0.7620** | **Foco Principal:** Identificamos 76% dos clientes que realmente cancelariam. |
| **F1-Score** | **0.6176** | Equilíbrio sólido entre precisão e capacidade de captura (recall). |
| **Acurácia** | **0.7495** | Percentual geral de acertos considerando o threshold agressivo de 0.3. |

<br>

#### Matriz de Confusão (Dados de Teste)
Abaixo, a distribuição das previsões do modelo. Note o baixo volume de Falsos Negativos (clientes que o modelo diz que ficariam, mas cancelam), validando a escolha do threshold.

![Matriz de Confusão Final](reports/figures/confusion_matrix_final.png)

<br>

### 3. Rastreamento com MLflow
Todo o ciclo de vida do modelo — incluindo hiperparâmetros, curvas de perda (Loss) por época e artefatos binários (`.pt` e `.joblib`) — é registrado automaticamente.
- **Para visualizar**: Execute `mlflow ui` no terminal e acesse *http://localhost:5000*.
___


🚀 Instalação e Setup
------------------
**1. Preparação do Ambiente**
```
# Criar ambiente virtual
python -m venv .venv

# Ativar (Windows)
.venv\Scripts\activate
# Ativar (Linux/Mac)
source .venv/bin/activate

# Instalar dependências (Single Source of Truth: pyproject.toml)
python -m pip install -e ".[dev]"
```

**2. Execução via Comandos (Cross-Platform)**

| Ação | Atalho (Make) | Comando Manual (Terminal) |
| :--- | :--- | :--- |
| **Instalar Tudo** | `make install` | `python -m pip install -e ".[dev]"` |
| **Treinar Modelo** | `make train` | `python main.py --train` |
| **Rodar Testes** | `make test` | `pytest tests/` |
| **Rodar Linter** | `make lint` | `ruff check .` |
| **API Local** | `make run` | `uvicorn src.api.app:app --reload` |


___

🐳 *Docker* e Monitoramento
--------------------------
A solução está totalmente conteinerizada, incluindo a stack de observabilidade:
1. Subir tudo: `docker-compose up --build`
2. Endpoints:
- **API (Swagger)**: http://localhost:8000/docs
   - **Prometheus**: http://localhost:9090
   - **Grafana**: http://localhost:3000 *(Login: admin / admin)*

___


🌐 Deploy em Nuvem (Render)
---------------------------
A API também foi implantada em ambiente de produção utilizando o [***Render***](https://render.com/). Você pode testar a inferência diretamente pelo navegador sem necessidade de setup local.
- **Link da Documentação (Swagger)**: [https://churn-prediction-o703.onrender.com/docs](https://churn-prediction-o703.onrender.com/docs)
- **Endpoint de Saúde**: `https://churn-prediction-o703.onrender.com/health`
<br>

⚠️ O serviço utiliza instâncias gratuitas que entram em modo de repouso após um período de inatividade. Por isso, **o primeiro acesso pode levar entre 30 a 60 segundos** para "acordar" o servidor. Caso a página não carregue de imediato, por favor, aguarde alguns instantes e atualize a página.


___
        
🔍 Testando a API (Cliente Aleatório)
-------------------------------------
Para facilitar a validação da API sem a necessidade de construir um JSON manualmente, incluímos um utilitário:

1. Acesse o **Swagger UI**: http://localhost:8000/docs

2. Utilize o endpoint `GET /random_customer`. Ele retornará os dados de um cliente real do dataset (removendo o target).

3. Copie o resultado e cole no corpo do `POST /predict`.

4. O sistema retornará a probabilidade de churn e a mensagem de classificação baseada no threshold de 0.3.
___

🧠 Detalhes Técnicos
---------------------
- **Modelo**: MLP (Multi-Layer Perceptron) em *PyTorch* com Dropout (0.2).
- **Early Stopping**: Monitoramento da perda de validação com paciência de 10 épocas.
- **Threshold de Decisão**: Reduzido para 0.3 para aumentar o Recall (conforme análise no notebook `02_mlp_modeling.ipynb`).
- **Testes Automatizados**: Cobertura de limpeza de dados (Unitário), contrato de dados (Schema), carregamento de pesos (Smoke) e endpoints HTTP (Integração).

___
**Desenvolvido por:** Bruno Piatto, João Furlan, Paulo Krempel

*Grupo 20 - FIAP 9MLET* | *Pos Tech Machine Learning Engineering (FIAP).*