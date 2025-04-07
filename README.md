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

📊 Métricas de Avaliação
Modelo	R²	MAE	MSE
Regressão Linear	0.5111	23.01	881.54
Ridge	0.5103	23.04	882.98
Lasso	0.3920	26.33	1096.35
Random Forest	0.5428	22.03	824.47
Gradient Boosting	0.5369	22.50	835.01
XGBoost	0.5505	22.00	810.34
🏡 Previsão para Novo Imóvel

Foi feita uma previsão para um imóvel com as seguintes características:

{
  'latitude': 40.75362,
  'longitude': -73.98377,
  'minimo_noites': 1,
  'numero_de_reviews': 45,
  'reviews_por_mes': 0.38,
  'calculado_host_listings_count': 2,
  'disponibilidade_365': 355,
  'bairro_group_Manhattan': 1,
  'bairro_Midtown': 1,
  'room_type_Entire home/apt': 1
}

📌 Preço previsto: US$ 157.73
🛠️ Tecnologias Utilizadas

    Python (Pandas, NumPy, Scikit-learn)

    XGBoost

    Matplotlib e Seaborn

    Jupyter Notebook


✅ Conclusão

Após testes com diversos modelos, o XGBoost se destacou como o mais eficiente na previsão de preços, alcançando um R² de 0.5505, o melhor entre os avaliados. Ele foi escolhido como modelo final por equilibrar bom desempenho e robustez.
