
#pip install streamlit pandas plotly joblib bertopic transformers scikit-learn

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from bertopic import BERTopic
from transformers import pipeline
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# ========================
#  Carga de datos
# ========================

# Cargar datos desde el archivo CSV

archivo_csv = '/home/mario/Documents/camiApp/sentiment_tweets3.csv'
data = pd.read_csv(archivo_csv, encoding='latin-1', header=None)
data = data.iloc[:, [1]] # Seleccionar solo la segunda columna
data.columns = ['tweet']

# ========================
# Preprocesamiento de texto
# ========================

def clean_text(text):
    text = re.sub(r'http\S+', '', text)   # Eliminar URLs
    text = re.sub(r'@\S+', '', text)      # Eliminar menciones
    text = re.sub(r'#\S+', '', text)      # Eliminar hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Eliminar caracteres especiales
    return text.lower()

data['clean_tweet'] = data['tweet'].apply(clean_text)
cleaned_tweets = data['clean_tweet'].tolist()

# ========================
# Modelado de tópicos con BERTopic
# ========================

def train_topic_model(cleaned_tweets):
    custom_stop_words = list(ENGLISH_STOP_WORDS.union({"tweet", "tweets", "twitter", "user", "account", "mention", "hashtag"
}))
    topic_model = BERTopic(nr_topics=5, top_n_words=5) #5 tópicos y 5 palabras por tópico
    topic_model.vectorizer_model.set_params(stop_words=custom_stop_words) 
    topics, probabilities = topic_model.fit_transform(cleaned_tweets) 
    return topic_model, topics
topic_model, topics = train_topic_model(cleaned_tweets)

# Guardar el modelo y los tópicos
joblib.dump(topic_model, "bertopic_model.pkl")
joblib.dump(topics, "topics_results.pkl")
# Guardar cleaned_tweets en un archivo .pkl
joblib.dump(cleaned_tweets, "cleaned_tweets.pkl")

# ========================
# Análisis de Sentimiento
# ========================

def analyze_sentiment(data, topics):
    sentiment_analyzer = pipeline('sentiment-analysis')
    sentiment_results = []
    for i, tweet in enumerate(data['clean_tweet']):
        sentiment = sentiment_analyzer(tweet)
        sentiment_results.append({
            'tweet': tweet,
            'topic': topics[i],
            'sentiment': sentiment[0]['label'],
            'score': sentiment[0]['score']
        })
    return sentiment_results

sentiment_results = analyze_sentiment(data, topics)

# Guardar resultados
sentiment_df = pd.DataFrame(sentiment_results)
sentiment_df.to_csv("sentiment_results.csv", index=False)
joblib.dump(sentiment_results, "sentiment_results.pkl")