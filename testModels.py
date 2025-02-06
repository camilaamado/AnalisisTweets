import joblib
import pandas as pd
from bertopic import BERTopic
from collections import Counter
import matplotlib.pyplot as plt

# Cargar datos desde el archivo CSV
data = pd.read_csv("sentiment_tweets3.csv", header=None, encoding='latin-1')

# Asignar nombres a las columnas
data.columns = ['col1', 'tweet', 'col3']  # Ajusta los nombres según el número de columnas en tu archivo CSV

print(data.head())
print(len(data))  # Cantidad de tweets en el DataFrame original

cleaned_tweets = data['tweet'].tolist()
print(len(cleaned_tweets))  # Cantidad de tweets después del preprocesamiento

topics = joblib.load("topics_results.pkl")
print(len(topics))  # Cantidad de tópicos generados por BERTopic

# Analizar la distribución de los tópicos
topic_counts = Counter(topics)
print(topic_counts)

# Porcentaje de tweets por tópico
total_tweets = len(topics)
topic_percentages = {topic: count / total_tweets * 100 for topic, count in topic_counts.items()}
print(topic_percentages)

# Visualización de la distribución de los tópicos
plt.bar(topic_percentages.keys(), topic_percentages.values())
plt.xlabel('Tópico')
plt.ylabel('Porcentaje de Tweets')
plt.title('Distribución de Tópicos')
plt.show()







#Análisis de palabras clave por tópico
topic_model = joblib.load("bertopic_model.pkl")
top_words_per_topic = topic_model.get_topic_info()
print(top_words_per_topic)




#Exploración de los tweets de un tópico específico
# Cargar los datos
data = pd.read_csv("sentiment_tweets3.csv")  
topics = joblib.load("topics_results.pkl") # Cargar los tópicos generados por BERTopic
data["topic"] = topics

# Filtrar tweets de un tópico específico
topic_number = 0  
tweets_in_topic = data[data["topic"] == topic_number]
print(tweets_in_topic)



#Distribución de Sentimientos por Tópico
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos de sentimiento
sentiment_df = pd.read_csv("sentiment_results.csv")

# Contar los sentimientos por tópico
sentiment_counts = sentiment_df.groupby(["topic", "sentiment"]).size().unstack()

# Graficar la distribución
sentiment_counts.plot(kind="bar", stacked=True, figsize=(10,6))
plt.title("Distribución de Sentimientos por Tópico")
plt.xlabel("Tópico")
plt.ylabel("Cantidad de Tweets")
plt.legend(title="Sentimiento")
plt.show()

