##################################     EMBEDDINGS PARA TWEETS Y TOPICOS    ################################################
# # Generación y almacenamiento de embeddings
# Este script muestra cómo generar y guardar embeddings de un conjunto de textos utilizando un modelo de sentence-transformers que está integrado dentro de BERTopic.
    #Modelo ideal para creacion de embeddings de textos cortos como tweets y clasificacion de topicos. 

from bertopic import BERTopic  # Modelo de tópicos
import pickle 
import joblib
import numpy as np

cleaned_tweets = joblib.load("/home/mario/Documents/camiApp/data/cleaned_tweets.pkl")
topics = joblib.load("/home/mario/Documents/camiApp/savedModels/topics_results.pkl")
topic_model = joblib.load("/home/mario/Documents/camiApp/savedModels/bertopic_model.pkl")


###################################################################  CREAR EMBEDDINGS  ###################################################################

embeddings = topic_model._extract_embeddings(cleaned_tweets) #Toma los textos en cleaned_tweets y los convierte en embeddings
print(type(embeddings))   # Imprime el tipo de embeddings. Debe ser un numpy array o una lista de listas
print(len(embeddings))    # cantidad de embeddings generados debe coincidir con cantidad de tcleaned_tweets
print(embeddings[0])      # Muestra el embedding del primer tweet, cantidad de números en cada embedding dependerá del modelo de sentence-transformers que se haya utilizado.


############################################################# GUARDAR Y CARGAR EMBEDDINGS  #############################################################
import pickle

# Guarda los embeddings en un archivo .pkl
with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

# Cargar los embeddings desde el archivo .pkl
with open("embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

########################################################################################################################################################
