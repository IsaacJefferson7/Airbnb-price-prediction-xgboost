# Previsão de Preços de Imóveis no Airbnb - NYC 🏙️

Este projeto utiliza técnicas de aprendizado de máquina para prever o preço de diárias de imóveis anunciados no Airbnb em Nova York. Diversos modelos de regressão foram testados, sendo o XGBoost o que apresentou melhor desempenho.

## 🎯 Objetivos do Projeto

- Limpar e preparar os dados.
- Explorar visualmente e estatisticamente o conjunto de dados.
- Aplicar e comparar diferentes modelos de regressão.
- Ajustar hiperparâmetros do melhor modelo.
- Prever o preço de um novo imóvel.

## 📦 Dados Utilizados

O conjunto de dados contém informações sobre imóveis do Airbnb em NYC, como:

- Localização (latitude, longitude, bairro)
- Tipo de acomodação
- Número de reviews
- Disponibilidade ao longo do ano
- Quantidade de imóveis do anfitrião
- Preço (variável alvo)

## 🧼 Etapas de Limpeza e Preparação

- Remoção de colunas irrelevantes (ID, nome do host etc.)
- Tratamento de valores ausentes
- Remoção de outliers extremos (abaixo do percentil 5% e acima do 95%)
- Codificação de variáveis categóricas com `get_dummies`

## 🤖 Modelagem e Avaliação

Modelos testados:

- Regressão Linear
- Regressão Ridge
- Regressão Lasso
- Random Forest
- Gradient Boosting
- **XGBoost (modelo final escolhido)**

### 🔧 Melhores hiperparâmetros do XGBoost (via GridSearchCV)

```python
{
  'colsample_bytree': 0.8,
  'learning_rate': 0.1,
  'max_depth': 7,
  'n_estimators': 100,
  'subsample': 0.8
}
