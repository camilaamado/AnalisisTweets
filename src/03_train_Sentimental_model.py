#entrenar modelo de analisis de sentimiento usando ROBERTA

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import joblib
import numpy as np
from scipy.special import softmax
import pandas as pd

# Cargar los tweets limpios
cleaned_tweets = joblib.load("/home/mario/Documents/camiApp/data/CleanAndEmbeddins/cleaned_tweets.pkl")

# Inicializar el modelo y tokenizer de RoBERTa para análisis de sentimiento
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Clasificar sentimiento de cada tweet
sentiments = []

for tweet in cleaned_tweets:
    # Tokenizar el tweet
    tokens = tokenizer(tweet, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Obtener la predicción del modelo
    with torch.no_grad():
        output = model(**tokens)
    
    # Obtener la clase con mayor probabilidad
    sentiment = torch.argmax(output.logits, dim=1).item()
    sentiments.append(sentiment)

# Guardar los resultados de sentimiento
joblib.dump(sentiments, "/home/mario/Documents/camiApp/data/SentimentResults.pkl")

print("Análisis de sentimiento completado y guardado correctamente.")

