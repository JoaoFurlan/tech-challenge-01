# Tech Challenge 01 — Previsão de Churn

Projeto desenvolvido para o **Tech Challenge 01 da Pós Tech em Machine Learning Engineering (FIAP)**.

O objetivo é construir um **modelo preditivo de churn** capaz de identificar clientes com risco de cancelamento em uma operadora de telecomunicações.

O modelo principal será uma **rede neural MLP implementada em PyTorch**, comparada com modelos baseline utilizando **Scikit-Learn**.

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

## Tecnologias utilizadas

- Python
- Pandas
- Scikit-Learn
- PyTorch
- Matplotlib
- Seaborn