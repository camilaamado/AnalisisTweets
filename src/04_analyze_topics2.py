### Análisis de tópicos y distribución de tópicos

from sklearn.metrics import silhouette_score
import pandas as pd
import joblib
from bertopic import BERTopic

# Cargar el modelo BERTopic y los datos
topic_model = joblib.load("/home/mario/Documents/camiApp/data/BERTopic_model.pkl")
cleaned_tweets = joblib.load("/home/mario/Documents/camiApp/data/CleanAndEmbeddins/cleaned_tweets.pkl")

# Obtener la información de los documentos
document_info = topic_model.get_document_info(cleaned_tweets)

# Obtener los tópicos asignados a los tweets
topics = document_info['Topic']

# Calcular el Silhouette Score
#embeddings = topic_model.transform(cleaned_tweets)
#silhouette_score_value = silhouette_score(embeddings, topics, metric='cosine')
#print(f"Silhouette Score: {silhouette_score_value}")

# Ver los términos más comunes por cada tópico
for topic_id in range(len(topic_model.get_topics())):
    print(f"Tópico {topic_id}: {topic_model.get_topic(topic_id)}")

# Contar la cantidad de tweets por tópico
topic_counts = topics.value_counts()

# Mostrar los tópicos con más tweets
print("Tópicos con más tweets:")
print(topic_counts.head())  # Muestra los primeros 5 tópicos con más tweets

# Filtrar los tópicos con más tweets
top_n = 5  # Número de tópicos más frecuentes a mostrar
top_topics = topic_counts.head(top_n)

# Ver los tweets que corresponden a los tópicos más frecuentes
for topic, count in top_topics.items():
    print(f"\nTópico {topic} (Tweets: {count}):")
    topic_tweets = [tweet for i, tweet in enumerate(cleaned_tweets) if topics[i] == topic]
    print(topic_tweets[:5])  # Muestra los primeros 5 tweets de cada tópicos