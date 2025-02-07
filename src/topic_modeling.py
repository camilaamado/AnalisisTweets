
##### Entrenamiento y análisis de tópicos con BERTopic #####




import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from bertopic import BERTopic
from transformers import pipeline
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS



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

