## Análisis de tópicos y distribución de tópicos usandi UMAP##########################

import joblib
import numpy as np
import umap
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from umap import UMAP

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
# VISUALIZACIÓN UMAP 2D
# =========================

# Reducir los embeddings de los tópicos a 2D
umap_model = UMAP(n_neighbors=15, n_components=2, metric='cosine', random_state=42)
umap_embeddings = umap_model.fit_transform(topic_model.topic_embeddings_)

# Crear el scatter plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=topic_counts.index, cmap="viridis", alpha=0.5)

# Agregar leyenda
plt.colorbar(scatter, label="Tópico")
plt.xlabel("UMAP Dim 1")
plt.ylabel("UMAP Dim 2")
plt.title("Proyección UMAP de los Tópicos")
plt.show()
