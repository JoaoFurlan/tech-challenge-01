# Tech Challenge 01 — Previsão de Churn

Projeto desenvolvido para o Tech Challenge 01 da Pós Tech em Machine Learning Engineering (FIAP).

O objetivo é construir um modelo preditivo de churn capaz de identificar clientes com risco de cancelamento em uma operadora de telecomunicações.

O modelo principal será uma rede neural MLP implementada em PyTorch e comparada com modelos baseline utilizando Scikit-Learn.

---

## Estrutura do projeto
```text
tc-01-previsao-churn
│
├── data/ (datasets utilizados no projeto)
├── notebooks/ (análise exploratória e experimentos)
├── src/ (código fonte reutilizável)
├── models/ (modelos treinados)
├── tests/ (testes automatizados)
└── docs/ (documentação do projeto) 

```
---

## Dataset

Telco Customer Churn (IBM)

Dataset público contendo informações de clientes de telecomunicações e variáveis relacionadas a contratos, serviços e faturamento.

Foi utilizado a versão atualizada, contendo marcadores geográficos e score de satisfação, disponível em:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn/discussion/441991