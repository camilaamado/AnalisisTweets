## Entrenar BERTopic y guardar modelo
from bertopic import BERTopic
import joblib
import numpy as np

# Cargar los tweets limpios
cleaned_tweets = joblib.load("/home/mario/Documents/camiApp/data/CleanAndEmbeddins/cleaned_tweets.pkl")

# Cargar los embeddings pre-generados
topic_embeddings = joblib.load("/home/mario/Documents/camiApp/data/CleanAndEmbeddins/topic_embeddings.pkl")

# Si los embeddings son 1D, conviértelos en 2D
if len(np.shape(topic_embeddings)) == 1:
    topic_embeddings = np.array(topic_embeddings).reshape(-1, 1)  # Reformatea a 2D


# Inicializar BERTopic
topic_model = BERTopic()

# Ajustar el modelo usando los embeddings pre-generados
topics, probs = topic_model.fit_transform(cleaned_tweets, topic_embeddings)

# Guardar el modelo BERTopic
joblib.dump(topic_model, "/home/mario/Documents/camiApp/data/BERTopic_model.pkl")

print("Modelo BERTopic entrenado y guardado correctamente.")

#Verificación del formato de los embeddings
import numpy as np

# Verifica la forma de los embeddings
print(np.shape(topic_embeddings))  # Debería devolver algo como (n_samples, n_features)
