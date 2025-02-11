import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
# Cargar el CSV de análisis de sentimiento
sentimentalTopics_path = "/home/mario/Documents/camiApp/data/sentiment_topic_analysis.csv"
sentimentalTopics = pd.read_csv(sentimentalTopics_path)

# Filtrar los tópicos que están en el rango de 0 a 9
df_filtered = sentimentalTopics[sentimentalTopics['topic'].isin(range(10))]


# Mostrar los primeros 5 registros del DataFrame filtrado
#print(df_filtered.head())


#Histograma de Sentimientos por Tópico

plt.figure(figsize=(12, 6))
sns.countplot(data=df_filtered, x="topic", hue="sentiment_label", palette="coolwarm")
plt.xlabel("Tópico")
plt.ylabel("Número de Tweets")
plt.title("Distribución de Sentimientos por Tópico")
plt.legend(["Negativo", "Neutral", "Positivo"])
plt.xticks(rotation=90)
plt.show()

#Heatmap de Sentimiento Promedio por Tópico
topic_sentiment_avg = df_filtered.groupby("topic")["sentiment_score"].mean().reset_index()


plt.figure(figsize=(12, 6))
sns.heatmap(topic_sentiment_avg.pivot(index="topic", columns="sentiment_score", values="sentiment_score"), 
            cmap="RdYlGn", annot=True, fmt=".2f")
plt.xlabel("Sentimiento")
plt.ylabel("Tópico")
plt.title("Sentimiento Promedio por Tópico")
plt.show()
#Qué tópicos tienden a ser más positivos o negativos. Identifica temas con opiniones extremas.


#Gráfico de Dispersión: Relación entre Polaridad y Tópico
plt.figure(figsize=(12, 6))
sns.stripplot(data=df_filtered, x="topic", y="sentiment_score", jitter=True, alpha=0.5, palette="coolwarm")
plt.xlabel("Tópico")
plt.ylabel("Sentimiento (Polaridad)")
plt.title("Distribución de Polaridad por Tópico")
plt.xticks(rotation=90)
plt.show()
#isualizar la variabilidad del sentimiento dentro de cada tópico.Si un tópico tiene una gran dispersión en sentimientos (mezcla de opiniones).
#Identifica tópicos con sentimientos polarizados.


# Gráfico de torta
topic_sentiment_avg = df_filtered.groupby('topic')['sentiment_score'].mean().reset_index()
sns.heatmap(topic_sentiment_avg.pivot(index="topic", columns="sentiment_score", values="sentiment_score"), 
            cmap="RdYlGn", annot=True, fmt=".2f")
plt.xlabel("sentiment_score")
plt.ylabel("topic")
plt.title("Sentimiento Promedio por Tópico")
plt.show()


# Regresión lineal
model = LinearRegression()
X = df_filtered[['topic']]  # Características de tópicos
y = df_filtered['sentiment_score']  # Objetivo de sentimiento

model.fit(X, y)

# Predicciones
y_pred = model.predict(X)

# Visualización de la regresión lineal
plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='blue', label='Datos reales')
plt.plot(X, y_pred, color='red', linewidth=2, label='Línea de regresión')
plt.xlabel("Tópico")
plt.ylabel("Sentimiento (Polaridad)")
plt.title("Regresión Lineal: Sentimiento vs Tópico")
plt.legend()
plt.show()