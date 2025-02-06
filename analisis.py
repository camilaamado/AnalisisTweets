#analisis de modelo de tópicos y análisis de sentimientos

#Analisis de topicos

import joblib
from bertopic import BERTopic

# Cargar el modelo BERTopic guardado
topic_model = joblib.load("bertopic_model.pkl")

# Obtener los tópicos y sus palabras más comunes
topics = topic_model.get_topics()

# Iterar sobre los tópicos y obtener las 10 palabras más comunes
for topic_id, words in topics.items():
    print(f"Tópico {topic_id}:")
    top_words = [word for word, _ in words[:10]]  # Las 10 palabras más comunes
    print(top_words)
    print("\n")

# Verificar si el modelo tiene tópicos
#print(topic_model.get_topics())


#######################

# Ver los tópicos generados
topics = topic_model.get_topics()

# Ver todos los tópicos y las palabras asociadas
print(topics)

# Ver el número de tópicos generados
print(f"Cantidad de tópicos generados: {topic_model.get_topics()}")
