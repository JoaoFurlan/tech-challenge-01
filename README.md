# Tech Challenge 01 — Previsão de Churn

Projeto desenvolvido para o **Tech Challenge 01 da Pós Tech em Machine Learning Engineering (FIAP)**.

O objetivo é construir um **modelo preditivo de churn** capaz de identificar clientes com risco de cancelamento em uma operadora de telecomunicações.

O modelo principal **foi uma rede neural MLP implementada em PyTorch**, comparada com modelos baseline utilizando **Scikit-Learn**.

---

## Estrutura do projeto

```text
tc-01-previsao-churn
│
├── data/
│   ├── raw/        (dataset original)
│   └── processed/  (dataset tratado para modelagem)
│
├── notebooks/      (análise exploratória e experimentos)
├── src/            (código fonte reutilizável)
├── models/         (artefatos de modelos treinados)
├── tests/          (testes automatizados)
└── docs/           (documentação do projeto)

```
---

## Dataset

**Telco Customer Churn (IBM)**

Dataset público contendo informações de clientes de telecomunicações e variáveis relacionadas a:

- contratos
- serviços contratados
- faturamento
- cancelamento de clientes (churn)

Disponível em:  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Acesso em: **07/03/2026**

---

## Modelagem

Foram utilizados diferentes modelos para previsão de churn:

- Logistic Regression (baseline linear)
- Random Forest (baseline baseado em árvores)
- MLP (Multi-Layer Perceptron) implementado em PyTorch

A rede neural foi treinada utilizando:

- batching com **PyTorch DataLoader**
- função de perda **BCEWithLogitsLoss**
- otimizador **Adam**
- **early stopping** para reduzir risco de overfitting

---

## Rastreamento de experimentos

Todos os experimentos foram registrados utilizando **MLflow**, permitindo acompanhar:

- parâmetros de treinamento
- métricas de avaliação
- artefatos gerados durante o experimento

Entre os artefatos registrados estão:

- modelos treinados
- tabela comparativa de resultados

---

## Métricas de avaliação

Os modelos foram avaliados utilizando múltiplas métricas:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Também foi analisado o impacto da alteração do **threshold de decisão** na rede neural para avaliar o trade-off entre precisão e recall.

---

## Tecnologias utilizadas

- Python
- Pandas
- Scikit-Learn
- PyTorch
- MLflow
- Matplotlib
- Seaborn