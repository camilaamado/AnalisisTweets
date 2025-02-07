#Importar las librerías necesarias
import joblib
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt


# Cargar los tweets limpios 
cleaned_tweets = joblib.load("cleaned_tweets.pkl")
# Cargar los tópicos asignados a cada tweet
topics = joblib.load("topics_results.pkl")
#cargar el modelo
topic_model = joblib.load("bertopic_model.pkl")
# Cambiar los nombres de los tópicos
topics = np.array(topics) + 1

#################### Distribución de los tópicos######################

topic_counts = pd.Series(topics).value_counts()
porcentaje = topic_counts / topic_counts.sum() * 100

plt.figure(figsize=(8, 6))
topic_counts.plot(kind='bar', color='red')
for i in range(len(topic_counts)):
    plt.text(i, topic_counts[i], str(round(porcentaje[i], 2)) + "%", ha='center', va='bottom')
plt.title("Distribución de los Tópicos")
plt.xlabel("Tópicos")
plt.ylabel("Número de Tweets")
plt.xticks(rotation=0)
plt.show()

######################Palabras clave por Tópico##########################

topic_info = topic_model.get_topic_info()

# Mostrar las 5 palabras clave más importantes de cada tópico
for topic_id in range(len(topic_model.get_topics())):
    print(f"Topic {topic_id}: {topic_model.get_topic(topic_id)}")

topic_info.to_csv("/home/mario/Documents/camiApp/data/fivetopics_Berttopics.csv", index=False)# Guardar en un archivo CSV


###################### Similitudes entre los Tópicos utilizando UMAP ##########################

