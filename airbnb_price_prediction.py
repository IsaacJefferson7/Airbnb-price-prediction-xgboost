import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import datetime

# Carrega o dataset e o transforma em um dataframe.
df = pd.read_csv('teste_indicium_precificacao.csv')
df.head()



##.............................................................ANÁLISE PRIMITIVA DOS DADOS.............................................................##

# Imprime a quantidade de dados nulos existentes em cada coluna do dataframe.
print(df.isnull().sum())

print(((df.isnull().sum() / df.shape[0] * 100).round(2)).sort_values(ascending=False).astype(str) + "%")

# Imprime os tipos de dados de cada coluna para confirmar se estão no formato adequado.
print(df.dtypes)

df.hist(bins=15, figsize=(15,10));

df[['price', 'minimo_noites', 'numero_de_reviews', 'reviews_por_mes', 'calculado_host_listings_count', 'disponibilidade_365']].describe()

df.price.plot(kind='box', vert=False, figsize=(15, 3),)
plt.show()

# ver quantidade de valores acima de 240 para price
print("\nprice: valores acima de 240")
print("{} entradas".format(len(df[df.price > 240])))
print("{:.4f}%".format((len(df[df.price > 240]) / df.shape[0])*100))

df.minimo_noites.plot(kind='box', vert=False, figsize=(15, 3),)
plt.show()

# ver quantidade de valores acima de 30 para price
print("\nminimo_noites: valores acima de 30")
print("{} entradas".format(len(df[df.minimo_noites > 30])))
print("{:.4f}%".format((len(df[df.minimo_noites > 30]) / df.shape[0])*100))



##.............................................................LIMPEZA DE DADOS.............................................................##

# remover os *outliers* em um novo DataFrame
df_clean = df.copy()
df_clean.drop(df_clean[df_clean.price > 240].index, axis=0, inplace=True)
df_clean.drop(df_clean[df_clean.minimo_noites > 30].index, axis=0, inplace=True)

# plotar o histograma para as variáveis numéricas
df_clean.hist(bins=15, figsize=(15,10));

print(df_clean.isnull().sum())

# Transforma os dados da coluna "ultima_review" para o formato de datetime.
df_clean['ultima_review'] = pd.to_datetime(df_clean['ultima_review'])

# Busca e imprime as datas de menor e maior valor, criando um parâmetro.
min_date_review = df_clean['ultima_review'].min()
max_date_review = df_clean['ultima_review'].max()

dt = 0
datas = []
while (dt < 7779):
    random_date = min_date_review + (max_date_review - min_date_review) * random.random()
    datas.append(random_date)
    dt += 1

df_clean['ultima_review'] = df_clean['ultima_review'].fillna(pd.Series(datas, index=df_clean[df_clean['ultima_review'].isnull()].index))

# Preenche os dados nulos da coluna "reviews_por_mes" com a mediana dos dados existentes na coluna.
df_clean['reviews_por_mes'] = df_clean['reviews_por_mes'].transform(lambda x: x.fillna(x.median()))

print(df_clean.isnull().sum())

# Transforma os dados da coluna "ultima_review" para o formato de datetime.
df['ultima_review'] = pd.to_datetime(df['ultima_review'])

# Busca e imprime as datas de menor e maior valor, criando um parâmetro.
min_date_review = df['ultima_review'].min()
max_date_review = df['ultima_review'].max()

dt = 0
datas = []
while (dt < 10052):
    random_date = min_date_review + (max_date_review - min_date_review) * random.random()
    datas.append(random_date)
    dt += 1

df['ultima_review'] = df['ultima_review'].fillna(pd.Series(datas, index=df[df['ultima_review'].isnull()].index))

# Preenche os dados nulos da coluna "reviews_por_mes" com a mediana dos dados existentes na coluna.
df['reviews_por_mes'] = df['reviews_por_mes'].transform(lambda x: x.fillna(x.median()))



from scipy.spatial import cKDTree

# Substitui o "host_name" nulo pelo "host_name" com as coordenadas mais próximas.
df_treino = df_clean[df_clean['host_name'].notnull()]
tree = cKDTree(df_treino[['latitude', 'longitude']])

for index, row in df_clean[df_clean['host_name'].isnull()].iterrows():
    _, idx = tree.query([row['latitude'], row['longitude']])
    df_clean.at[index, 'host_name'] = df_treino.iloc[idx]['host_name']


# Substitui o "nome" nulo pelo "nome" com as coordenadas mais próximas.
    df_treino = df_clean[df_clean['nome'].notnull()]
tree = cKDTree(df_treino[['latitude', 'longitude']])

for index, row in df_clean[df_clean['nome'].isnull()].iterrows():
    _, idx = tree.query([row['latitude'], row['longitude']])
    df_clean.at[index, 'nome'] = df_treino.iloc[idx]['nome']

    from scipy.spatial import cKDTree


# Substitui o "host_name" nulo pelo "host_name" com as coordenadas mais próximas.
df_treino = df[df['host_name'].notnull()]
tree = cKDTree(df_treino[['latitude', 'longitude']])

for index, row in df[df['host_name'].isnull()].iterrows():
    _, idx = tree.query([row['latitude'], row['longitude']])
    df.at[index, 'host_name'] = df_treino.iloc[idx]['host_name']


# Substitui o "nome" nulo pelo "nome" com as coordenadas mais próximas.
    df_treino = df[df['nome'].notnull()]
tree = cKDTree(df_treino[['latitude', 'longitude']])

for index, row in df[df['nome'].isnull()].iterrows():
    _, idx = tree.query([row['latitude'], row['longitude']])
    df.at[index, 'nome'] = df_treino.iloc[idx]['nome']


df.head(5)

print(df_clean.isnull().sum())



# Criação dos subplots
plt.figure(figsize=(9, 3))

# Primeiro histograma
plt.subplot(1, 2, 1)
plt.hist(df['minimo_noites'], bins=100)
plt.title('Dados pré limpeza')

# Segundo histograma
plt.subplot(1, 2, 2)
plt.hist(df_clean['minimo_noites'], bins=100)
plt.title('Dados pós limpeza')

#Título geral
plt.suptitle('Reviews por mês')

# Exibir os gráficos
plt.tight_layout()
plt.show()

print("\nDados pré limpeza:")
print("Mínimo de noites:", df['minimo_noites'].min())
print("Mínimo de noites:", df['minimo_noites'].max())

print("\nDados pós limpeza:")
print("Mínimo de noites:", df_clean['minimo_noites'].min())
print("Mínimo de noites:", df_clean['minimo_noites'].max())


# Criação dos subplots
plt.figure(figsize=(9, 3))

# Primeiro histograma
plt.subplot(1, 2, 1)
plt.hist(df['price'], bins=500)
plt.title('Dados pré limpeza')

# Segundo histograma
plt.subplot(1, 2, 2)
plt.hist(df_clean['price'], bins=100)
plt.title('Dados pós limpeza')

# Título geral
plt.suptitle('Preço')

# Exibir os gráficos
plt.tight_layout()
plt.show()

print("\nDados pré limpeza:")
print("Preço mínimo:", df['price'].min())
print("Preço máximo:", df['price'].max())

print("\nDados pós limpeza:")
print("Preço mínimo:", df_clean['price'].min())
print("Preço máximo:", df_clean['price'].max())


## Análise Exploratória dos Dados (EDA)

df_clean.describe()

#Adiciona apenas as colunas que contenham valores numéricos para a visualização na matriz de correlação
df_numeric = df_clean.select_dtypes(include=[np.number])
plt.figure(figsize=(8, 6))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.show()

# Define o tamanho da figura e o número de subplots (3 linha, 3 colunas)
fig, axs = plt.subplots(3, 3, figsize=(20, 15))


#Gráficos:

# 1. Preço vs Reviews por Mês
sns.regplot(data=df_clean, x='reviews_por_mes', y='price', ax=axs[0, 0], color='blue', line_kws={'color': 'brown'})
axs[0, 0].set_title('Preço vs Reviews por Mês')
axs[0, 0].set_xlabel('Reviews por Mês')
axs[0, 0].set_ylabel('Preço ($)')

# 2. Preço vs Número de Reviews
sns.regplot(data=df_clean, x='numero_de_reviews', y='price', ax=axs[0, 1], color='purple', line_kws={'color': 'brown'})
axs[0, 1].set_title('Preço vs Número de Reviews')
axs[0, 1].set_xlabel('Número de Reviews')
axs[0, 1].set_ylabel('')

# 3. Gráfico: Preço vs Número Mínimo de Noites
sns.regplot(data=df_clean, x='minimo_noites', y='price', ax=axs[0, 2], color='red', line_kws={'color': 'brown'})
axs[0, 2].set_title('Preço vs Número Mínimo de Noites')
axs[0, 2].set_xlabel('Número Mínimo de Noites')
axs[0, 2].set_ylabel('')

# 4. Gráfico: Preço vs Disponibilidade
sns.regplot(data=df_clean, x='disponibilidade_365', y='price', ax=axs[1, 0], color='green', line_kws={'color': 'brown'})
axs[1, 0].set_title('Preço vs Disponibilidade')
axs[1, 0].set_xlabel('Disponibilidade (dias)')
axs[1, 0].set_ylabel('Preço ($)')

# 5. Preço vs Quantidade de anúncios por host
sns.regplot(data=df_clean, x='calculado_host_listings_count', y='price', ax=axs[1, 1], color='black', line_kws={'color': 'brown'})
axs[1, 1].set_title('Preço vs Quantidade de anúncios por host')
axs[1, 1].set_xlabel('Quantidade de anúncios por usuário')
axs[1, 1].set_ylabel('')

# 6. Preço vs Tipos de quartos
sns.scatterplot(data=df_clean, x='room_type', y='price', ax=axs[1, 2], color='yellow', alpha=0.6)
axs[1, 2].set_title('Preço vs Tipos de espaço')
axs[1, 2].set_xlabel('Tipo de espaço')
axs[1, 2].set_ylabel('')

# 7. Preço vs Latitude
sns.regplot(data=df_clean, x='latitude', y='price', ax=axs[2, 0], color='orange', line_kws={'color': 'brown'})
axs[2, 0].set_title('Preço vs Latitude')
axs[2, 0].set_xlabel('Latitude')
axs[2, 0].set_ylabel('Preço ($)')

# 8. Preço vs Longitude
sns.regplot(data=df_clean, x='longitude', y='price', ax=axs[2, 1], color='cyan', line_kws={'color': 'brown'})
axs[2, 1].set_title('Preço vs Longitude')
axs[2, 1].set_xlabel('Longitude')
axs[2, 1].set_ylabel('')

# 9. Preço vs Bairro
sns.scatterplot(data=df_clean, x='bairro_group', y='price', ax=axs[2, 2], color='pink', alpha=0.6)
axs[2, 2].set_title('Preço vs Bairro')
axs[2, 2].set_xlabel('Bairro')
axs[2, 2].set_ylabel('')

# Ajusta o layout para evitar sobreposição
plt.tight_layout()
plt.show()


##.............................................................PERGUNTA 1.............................................................##

# Define o tamanho da figura e o número de subplots (3 linha, 3 colunas)
fig, axs = plt.subplots(2, 2, figsize=(18, 10))


#Gráficos:

# 1. Preço vs Reviews por Mês
sns.scatterplot(data=df_clean, x='reviews_por_mes', y='bairro_group', ax=axs[0, 0], color='blue', alpha=0.6)
axs[0, 0].set_title('Bairro vs Reviews por Mês')
axs[0, 0].set_xlabel('Reviews por Mês')
axs[0, 0].set_ylabel('Bairro')

# 2. Bairro vs Número de Reviews
sns.scatterplot(data=df_clean, x='numero_de_reviews', y='bairro_group', ax=axs[0, 1], color='purple', alpha=0.6)
axs[0, 1].set_title('Bairro vs Número de Reviews')
axs[0, 1].set_xlabel('Número de Reviews')
axs[0, 1].set_ylabel('')

# 3. Gráfico: Bairro vs Número Mínimo de Noites
sns.scatterplot(data=df_clean, x='minimo_noites', y='bairro_group', ax=axs[1, 0], color='red', alpha=0.6)
axs[1, 0].set_title('Preço vs Número Mínimo de Noites')
axs[1, 0].set_xlabel('Número Mínimo de Noites')
axs[1, 0].set_ylabel('Bairro')

# 4. Gráfico: Bairro vs Preço
sns.scatterplot(data=df_clean, x='price', y='bairro_group', ax=axs[1, 1], color='green', alpha=0.6)
axs[1, 1].set_title('Preço vs Disponibilidade')
axs[1, 1].set_xlabel('preço')
axs[1, 1].set_ylabel('')

# Ajusta o layout para evitar sobreposição
plt.tight_layout()
plt.show()


##.............................................................PERGUNTA 2.............................................................##

# Define o tamanho da figura e o número de subplots (3 linha, 3 colunas)
fig, axs = plt.subplots(1, 2, figsize=(18, 6))


#Gráficos:

# 1. Gráfico: Preço vs Número Mínimo de Noites
sns.regplot(data=df_clean, x='minimo_noites', y='price', ax=axs[0], color='red', line_kws={'color': 'brown'})
axs[0].set_title('Preço vs Número Mínimo de Noites')
axs[0].set_xlabel('Número Mínimo de Noites')
axs[0].set_ylabel('')

# 2. Gráfico: Preço vs Disponibilidade
sns.regplot(data=df_clean, x='disponibilidade_365', y='price', ax=axs[1], color='green', line_kws={'color': 'brown'})
axs[1].set_title('Preço vs Disponibilidade')
axs[1].set_xlabel('Disponibilidade (dias)')
axs[1].set_ylabel('Preço ($)')

# Ajusta o layout para evitar sobreposição
plt.tight_layout()
plt.show()


##.............................................................PERGUNTA 3.............................................................##

# Criando faixas de preço
df_clean['faixa_preco'] = pd.cut(df_clean['price'], bins=[0, 69, 175, df_clean['price'].max()], labels=['Baixo', 'Médio', 'Alto'])

from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

stop_words = set(stopwords.words('english'))

# Função para processar texto
def processar_texto(texto):
    tokens = word_tokenize(texto.lower().translate(str.maketrans('', '', string.punctuation)))
    palavras_filtradas = [palavra for palavra in tokens if palavra not in stop_words]
    return palavras_filtradas

# Inicializando variáveis
faixa_alto_palavras = []
faixa_medio_baixo_palavras = []

# Coletando palavras da faixa de preço "Alto"
locais_alto = df_clean[df_clean['faixa_preco'] == 'Alto']['nome'].tolist()
for local in locais_alto:
    faixa_alto_palavras.extend(processar_texto(local))

# Coletando palavras das faixas de preço "Médio" e "Baixo"
locais_outros = df_clean[df_clean['faixa_preco'].isin(['Médio', 'Baixo'])]['nome'].tolist()
for local in locais_outros:
    faixa_medio_baixo_palavras.extend(processar_texto(local))

# Contabilizando as palavras
frequencia_alto = Counter(faixa_alto_palavras)
frequencia_outros = Counter(faixa_medio_baixo_palavras)

# Obtendo palavras exclusivas da faixa "Alto"
palavras_exclusivas_alto = {palavra: freq for palavra, freq in frequencia_alto.items() if palavra not in frequencia_outros}

# Exibindo as palavras exclusivas da faixa "Alto"
print("\nPalavras exclusivas na faixa de preço 'Alto':")
print(sorted(palavras_exclusivas_alto.items(), key=lambda x: x[1], reverse=True)[:15])  # Exibindo as 10 mais comuns


##.............................................................PREVISÃO DE NOVOS DADOS.............................................................##

print(df_clean.isnull().sum())

df_clean.dtypes

!pip install xgboost    # Caso usar o código no jupyter notebook e a biblioteca não tiver sido instalada.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor

# Remover colunas irrelevantes
df_clean = df_clean.drop(columns=["id", "nome", "host_id", "host_name", "ultima_review", "faixa_preco"], errors="ignore")

# Tratar valores ausentes
df_clean["reviews_por_mes"] = df_clean["reviews_por_mes"].fillna(0)

# Codificar variáveis categóricas
df_clean = pd.get_dummies(df_clean, columns=["bairro_group", "bairro", "room_type"], drop_first=True)

# Separar variáveis
X = df_clean.drop(columns=["price"])
y = df_clean["price"]

# Remover outliers
mask = (y > y.quantile(0.05)) & (y < y.quantile(0.95))
X = X[mask]
y = y[mask]

# Dividir e treinar
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# XGBoost Regressor
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# Avaliação
print("XGBoost Regressor:")
print("R²:", r2_score(y_test, y_pred_xgb))
print("MAE:", mean_absolute_error(y_test, y_pred_xgb))
print("MSE:", mean_squared_error(y_test, y_pred_xgb))




from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Dicionário com os modelos
modelos = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'RandomForest': RandomForestRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42),
}

# Treinar e avaliar cada modelo
for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"\n{nome}")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")


##.............................................................PREVISÃO DO PREÇO DO NOVO IMÓVEL.............................................................##

# Informações do novo imóvel
novo_imovel = {
    'latitude': 40.75362,
    'longitude': -73.98377,
    'minimo_noites': 1,
    'numero_de_reviews': 45,
    'reviews_por_mes': 0.38,
    'calculado_host_listings_count': 2,
    'disponibilidade_365': 355,
    'bairro_group_Manhattan': 1,
    'bairro_Midtown': 1,
    'room_type_Entire home/apt': 1,
}

# Criar DataFrame com uma linha
X_novo = pd.DataFrame([novo_imovel])

# Identificar colunas faltantes em relação ao treino
colunas_faltantes = [col for col in X_train.columns if col not in X_novo.columns]

# Adicionar colunas faltantes com valor 0
faltantes = pd.DataFrame(0, index=X_novo.index, columns=colunas_faltantes)
X_novo = pd.concat([X_novo, faltantes], axis=1)

# Garantir a mesma ordem de colunas que o X_train
X_novo = X_novo[X_train.columns]

# Previsão
preco_previsto = xgb.predict(X_novo)[0]
print(f"Preço previsto para o novo imóvel: ${preco_previsto:.2f}")
