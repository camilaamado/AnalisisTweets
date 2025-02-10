## Análisis de tópicos y distribución de tópicos 
######################### graficos##############################

### Análisis de tópicos y distribución de tópicos

from sklearn.metrics import silhouette_score
import pandas as pd
import joblib
from bertopic import BERTopic
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud

# =========================
# CARGAR MODELO Y DATOS
# =========================

# Cargar el modelo BERTopic y los datos
topic_model = joblib.load("/home/mario/Documents/camiApp/data/BERTopic_model.pkl")
cleaned_tweets = joblib.load("/home/mario/Documents/camiApp/data/CleanAndEmbeddins/cleaned_tweets.pkl")

# Obtener la información de los documentos
document_info = topic_model.get_document_info(cleaned_tweets)

# Obtener los tópicos asignados a los tweets
topics = document_info['Topic']

# Contar la cantidad de tweets por tópico
topic_counts = topics.value_counts()

# Filtrar tópicos más frecuentes (excluir -1, que es ruido)
topic_counts_filtered = topic_counts[topic_counts.index != -1]

# =========================
# GRÁFICO DE TORTA
# =========================

# Seleccionar los 5 tópicos más comunes
top_n = 5
top_topics = topic_counts_filtered.head(top_n)

# Crear el gráfico de torta
plt.figure(figsize=(8, 8))
plt.pie(top_topics.values, 
        labels=[f"Tópico {t}" for t in top_topics.index], 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=plt.cm.Paired.colors)

plt.title('Distribución de los 5 Tópicos Más Frecuentes')
plt.show()


# =========================
# NUBE DE PALABRAS
# =========================

# Seleccionar los tópicos más comunes
selected_topics = top_topics.index

# Diccionario para almacenar palabras y su importancia
word_weights = {}

for topic in selected_topics:
    words_weights = topic_model.get_topic(topic)  # Obtener palabras del tópico
    for word, weight in words_weights:
        if word in word_weights:
            word_weights[word] += weight
        else:
            word_weights[word] = weight

# Generar la nube de palabras
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_weights)

# Mostrar la nube de palabras
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Nube de Palabras de los Principales Tópicos")
plt.show()


# =========================
# GRÁFICO DE BARRAS - DISTRIBUCIÓN DE TÓPICOS
# =========================

plt.figure(figsize=(12, 6))
topic_counts_filtered.plot(kind="bar", color="skyblue", edgecolor="black")
plt.xlabel("Tópico")
plt.ylabel("Número de Tweets")
plt.title("Distribución de Tweets por Tópico")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()


# =========================
# MAPA DE SIMILITUD ENTRE TÓPICOS
# =========================

fig = topic_model.visualize_heatmap()
fig.show()

# =========================
# DIAGRAMA INTERACTIVO DE TÓPICOS
# =========================

fig = topic_model.visualize_barchart(top_n_topics=10)
fig.show()
