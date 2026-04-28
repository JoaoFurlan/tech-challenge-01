# Model Card - Churn Prediction MLP

Este documento fornece informações detalhadas sobre o modelo de predição de churn, garantindo transparência sobre seu funcionamento, limitações e métricas de performance.

## 1. Detalhes do Modelo
- **Desenvolvido por:** Grupo 20 - FIAP 9MLET (Bruno Piatto, João Furlan, Paulo Krempel).
- **Data:** Abril de 2026.
- **Tipo de Modelo:** Rede Neural Multicamadas (Multi-Layer Perceptron - MLP).
- **Framework:** PyTorch.
- **Versão:** 1.0.0.

## 2. Uso Pretendido
- **Caso de Uso:** Identificação proativa de clientes com alta probabilidade de cancelar serviços de telecomunicações.
- **Público-alvo:** Equipes de Customer Success e Marketing para ações de retenção.
- **Escopo:** O modelo foi desenhado para processar dados estruturados de perfis de clientes (demográfico, serviços e financeiro).

## 3. Fatores de Treinamento
- **Instrumentação:** Rastreamento completo via **MLflow**.
- **Otimizador:** Adam (Learning Rate: 0.001).
- **Função de Perda:** BCEWithLogitsLoss (ajustada para desbalanceamento de classes).
- **Pre-processamento:**
    - Variáveis Categóricas: OneHotEncoding (com tratamento de categorias desconhecidas).
    - Variáveis Numéricas: StandardScaler (Normalização Z-score).

## 4. Dados
- **Fonte:** Dataset Telco Customer Churn (IBM).
- **Divisão dos Dados:**
    - **Treino (64%)**: Ajuste de pesos.
    - **Validação (16%)**: Monitoramento de Early Stopping.
    - **Teste (20%)**: Avaliação final (Holdout).

## 5. Métricas de Performance
As métricas abaixo referem-se ao conjunto de **teste (dados inéditos)** utilizando um **threshold de 0.3**:

| Métrica | Valor |
| :--- | :--- |
| **ROC-AUC** | 0.8420 |
| **Recall (Sensibilidade)** | 0.7620 |
| **F1-Score** | 0.6176 |
| **Acurácia** | 0.7495 |

## 6. Considerações Éticas e Limitações
- **Privatização:** O modelo não utiliza nomes de clientes ou dados sensíveis protegidos por LGPD (apenas características de consumo e perfil geral).
- **Limitações:** - O modelo assume que o comportamento histórico de churn se repetirá no futuro.
    - A performance pode degradar se houver mudanças drásticas no mercado ou novos tipos de planos não mapeados no treinamento original.
- **Threshold**: O uso do threshold de 0.3 prioriza o Recall. Isso significa que o modelo aceita ter mais "alarmes falsos" (Falsos Positivos) para garantir que o maior número possível de cancelamentos reais seja detectado.

## 7. Como utilizar
Os artefatos do modelo estão disponíveis na pasta `/models`:
- `model_weights.pt`: Pesos da rede neural.
- `scaler.joblib` / `encoder.joblib`: Transformadores de features.
- `feature_names.joblib`: Ordem exata das colunas para inferência.