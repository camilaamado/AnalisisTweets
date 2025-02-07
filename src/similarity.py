############################  Analizar similitud entre tweets con similitud coseno #################



import joblib
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

# Cargar los tweets limpios 
cleaned_tweets = joblib.load("/home/mario/Documents/camiApp/data/cleaned_tweets.pkl")
embeddings = joblib.load("/home/mario/Documents/camiApp/src/embeddingsCreate/embeddingsTopics.pkl")



#La similitud coseno mide cuán similares son dos embeddings en un espacio vectorial. Valores cercanos a 1 indican alta similitud, mientras que valores cercanos a 0 indican poca relación.


# Calculamos la matriz de similitudes coseno entre todos los tweets
cosine_sim_matrix = cosine_similarity(embeddings)

# Mostramos la matriz (opcional, puede ser grande si tienes muchos tweets)
#print(cosine_sim_matrix)


#Heatmap de la matriz de similitudes

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(cosine_sim_matrix, cmap="coolwarm", annot=False)
plt.title("Matriz de Similitud Coseno entre Tweets")
plt.xlabel("Tweet ID")
plt.ylabel("Tweet ID")
plt.show()
