
##################################     EMBEDDINGS PARA TOPICOS DE TWEETS  ################################################
# # Generación y almacenamiento de embeddings
# Este script muestra cómo generar y guardar embeddings de un conjunto de textos utilizando un modelo de sentence-transformers que está integrado dentro de BERTopic.
    #Modelo ideal para creacion de embeddings de textos cortos como tweets y clasificacion de topicos. 

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import joblib
import pickle

# Cargar los tweets limpios
cleaned_tweets = joblib.load("/home/mario/Documents/camiApp/data/CleanAndEmbeddins/cleaned_tweets.pkl")

# Asegurar que los tweets sean una lista de strings
if isinstance(cleaned_tweets, pd.DataFrame):  # Si es un DataFrame, tomar la columna correcta
    cleaned_tweets = cleaned_tweets["text"].tolist()  # Cambia "text" por el nombre correcto de la columna
elif isinstance(cleaned_tweets, pd.Series):  # Si es una Serie, convertir directamente a lista
    cleaned_tweets = cleaned_tweets.tolist()

# Cargar el modelo de embeddings DistilBERT
embedding_model = SentenceTransformer("distilbert-base-nli-mean-tokens")

# Crear el modelo BERTopic con los embeddings de DistilBERT
topic_model = BERTopic(embedding_model=embedding_model)

# Ajustar el modelo a los tweets y generar embeddings
topics, embeddings = topic_model.fit_transform(cleaned_tweets)

# Guardar los embeddings en un archivo .pkl
with open("/home/mario/Documents/camiApp/data/CleanAndEmbeddins/topic_embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("Embeddings generados y guardados correctamente.")
