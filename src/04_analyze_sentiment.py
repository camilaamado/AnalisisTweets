from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import joblib
import numpy as np
import pandas as pd

sentiments = joblib.load("/home/mario/Documents/camiApp/data/SentimentResults.pkl")
print(sentiments[:10])  # Muestra los primeros 10 resultados

# Calcular la distribución de sentimientos

sentiment_counts = pd.Series(sentiments).value_counts()
print(sentiment_counts)


import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns

# 📌 Cargar modelos y datos
topic_model = joblib.load("/home/mario/Documents/camiApp/data/BERTopic_model.pkl")
cleaned_tweets = joblib.load("/home/mario/Documents/camiApp/data/CleanAndEmbeddins/cleaned_tweets.pkl")
topic_embeddings = joblib.load("/home/mario/Documents/camiApp/data/CleanAndEmbeddins/topic_embeddings.pkl")

# Asegurar que los embeddings sean 2D
topic_embeddings = np.array(topic_embeddings)

if topic_embeddings.ndim == 1:  
    topic_embeddings = topic_embeddings.reshape(-1, 1)  # Convertir a 2D si es necesario


topics = topic_model.transform(cleaned_tweets, topic_embeddings)[0]



# 📌 Asignar tópicos a los tweets usando los embeddings pre-generados
topics = topic_model.transform(cleaned_tweets, topic_embeddings)[0]
print("Tópicos asignados a los tweets:", topics[:10])  # Mostrar los primeros 10 tópicos

# 📌 Cargar modelo de sentimiento RoBERTa
sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# 📌 Analizar el sentimiento de cada tweet
sentiment_labels = []
sentiment_scores = []

for tweet in cleaned_tweets:
    encoded_tweet = tokenizer(tweet, return_tensors="pt", truncation=True, padding=True, max_length=512)
    output = sentiment_model(**encoded_tweet)
    
    scores = softmax(output.logits.detach().numpy()[0])  # Aplicamos softmax para obtener probabilidades
    sentiment = np.argmax(scores)  # 0 = negativo, 1 = neutral, 2 = positivo
    polarity_score = scores[2] - scores[0]  # Positivo - Negativo (mayor = más positivo)

    sentiment_labels.append(sentiment)
    sentiment_scores.append(polarity_score)

# 📌 Crear DataFrame con los resultados
sentimentalTopics = pd.DataFrame({
    "tweet": cleaned_tweets,
    "topic": topics,
    "sentiment_label": sentiment_labels,  # 0 = negativo, 1 = neutral, 2 = positivo
    "sentiment_score": sentiment_scores  # Polaridad (positivo - negativo)
})

# 📌 Guardar los resultados en un archivo CSV
sentimentalTopics.to_csv("/home/mario/Documents/camiApp/data/sentiment_topic_analysis.csv", index=False)

print("Análisis de sentimiento y tópicos completado y guardado correctamente.")