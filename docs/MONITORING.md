Plano de Monitorização e Playbook de Resposta
================================================

Este documento detalha a estratégia de observabilidade para a API de Predição de Churn, cobrindo métricas de infraestrutura, performance do modelo e procedimentos de resposta a incidentes.

Estratégia de Monitorização
------------------------------------
A nossa solução utiliza o padrão de **Monitorização de Sinais Dourados** (Golden Signals), focando-se em Latência, Erros e Tráfego.

**1. Métricas de Infraestrutura (Prometheus)**
Coletadas automaticamente através do `prometheus-fastapi-instrumentator` integrado na `app.py`.

- **Latência de Requisição (p95 e p99):** Tempo que 95% e 99% das chamadas levam para responder. Vital para garantir a experiência do utilizador na arquitetura *Real-time*.

- **Taxa de Erros (HTTP 5xx):** Monitorização de falhas críticas no código ou no carregamento do modelo.

- **Taxa de Erros de Cliente (HTTP 4xx):** Indica se sistemas externos estão a enviar JSONs mal formados (fora do Schema Pydantic).

- **Throughput (RPS):** Volume de requisições por segundo para prever a necessidade de escalonamento.

**2. Métricas de Modelo (ML Specific)**

- **Análise de Tendência de Predição (via Logs):** Monitoramento do comportamento do modelo através dos logs de execução (pasta `/logs`), onde cada predição e sua probabilidade são registradas. Um aumento atípico na probabilidade média pode indicar *Data Drift*.

- **Tempo de Inferência:** Medição específica do tempo gasto dentro da função `predict_new_customer` (excluindo o overhead da rede).

- **Saúde do Singleton:** Verificação se os artefatos (`.pt` e `.joblib`) foram carregados corretamente no arranque da aplicação (*lifespan event*).

___

Configuração de Alertas
-----------------------

Os alertas são disparados quando os indicadores ultrapassam os limiares de aceitação definidos para o negócio:

| Alerta | Gatilho (Trigger) | Severidade |
| :--- | :--- | :--- |
| **API Down** | Uptime < 100% nos últimos 2 min | Crítica |
| **Alta Latência** | p95 > 800ms por mais de 5 min | Atenção |
| **Falha Crítica** | Taxa de Erro 5xx > 5% do tráfego total | Crítica |
| **Drift de Predição** | Taxa de Churn > 60% (Anomalia de modelo) | Atenção |


___
Playbook de Resposta a Incidentes (Manual de Ação)
-----------------------
Procedimentos passo-a-passo para a equipa técnica em caso de falha.

**Cenário A: API Inacessível ou Retornando Erro 500**


1. **Diagnóstico**: Verificar logs do container via `docker logs churn-api`.

2. **Verificação**: Validar se o ficheiro `model_weights.pt` e os scalers estão presentes na pasta `/models`.

3. **Ação**: Reiniciar o serviço via `docker-compose restart api`. Se o erro persistir, verificar se houve alteração na versão do Python ou dependências no `pyproject.toml`.


**Cenário B: Degradação de Performance (Alta Latência)**
1. **Diagnóstico**: Observar o dashboard do Grafana para identificar se a latência subiu após um aumento de tráfego.

2. **Ação**: Verificar se a instância (Render/Docker) está com CPU ou RAM no limite. Escalar a infraestrutura ou otimizar a função de pré-processamento.

**Cenário C: Suspeita de Perda de Precisão (Model Drift)**
1. **Diagnóstico**: Cruzar os logs de predição da API com os dados reais de cancelamento coletados pelo time de negócio. Verificar se a média das probabilidades retornadas pela API divergiu significativamente da baseline de treino.

2. **Ação**:
   - Coletar amostra de dados reais que falharam na predição.
   - Executar o pipeline de retreino utilizando o comando `make train`.
   - Comparar o novo AUC-ROC e, se superior, realizar o deploy da nova versão do modelo.


___


Visualização
-----------------------
O acesso aos dashboards de monitorização é feito via:

- **Grafana**: http://localhost:3000 (Login: `admin` / `admin`)
- **Prometheus**: http://localhost:9090
> Nota: O dashboard do Grafana está pré-configurado para consumir a fonte de dados do Prometheus e exibir o painel de latência e volume de predições.
