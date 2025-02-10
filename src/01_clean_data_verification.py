#Verificar tweets vacíos o con pocas palabras

import pandas as pd
import joblib
from utils.text_processing import preprocess_text
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re



cleaned_tweets = joblib.load("/home/mario/Documents/camiApp/data/CleanAndEmbeddins/cleaned_tweets.pkl")

#Chequeo de tweets con pocas palabras
short_tweets = [tweet for tweet in cleaned_tweets if len(tweet.split()) <= 1]
print(f"🔍 Tweets con 1 palabras o menos: {len(short_tweets)}")
print(short_tweets[:10])  # Muestra algunos ejemplos

cleaned_tweets = [tweet for tweet in cleaned_tweets if len(tweet.split()) > 1] # Elimina tweets con 1 palabra o menos



#Revisar la distribución de longitud de tweets después de la limpieza
tweet_lengths = [len(tweet.split()) for tweet in cleaned_tweets]

plt.hist(tweet_lengths, bins=30, edgecolor='black')
plt.xlabel('Número de palabras')
plt.ylabel('Frecuencia')
plt.title('Distribución de longitud de tweets')
plt.show()

print(f"📊 Longitud promedio: {sum(tweet_lengths)/len(tweet_lengths):.2f} palabras")
print(f"📊 Longitud máxima: {max(tweet_lengths)} palabras")
print(f"📊 Longitud mínima: {min(tweet_lengths)} palabras")


#Revisar palabras más frecuentes (nube de palabras)
all_text = " ".join(cleaned_tweets)
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()  #✅ Si hay palabras irrelevantes, podríamos ajustar la lista de STOPWORDS.


#Buscar caracteres raros o símbolos que no deberían estar presentes
suspicious_tweets = [tweet for tweet in cleaned_tweets if re.search(r"[^a-zA-Z0-9\s]", tweet)]
print(f"🔍 Tweets con caracteres raros: {len(suspicious_tweets)}")
print(suspicious_tweets[:10])  # Mostrar algunos ejemplos. ✅ Si hay muchos, se puede ajustar la función remove_special_characters().


 #Validar que no haya duplicados
print(f"🛠 Tweets únicos: {len(set(cleaned_tweets))} de {len(cleaned_tweets)} totales") #✅ Tweets únicos: 9,844 de 9,983, lo que significa que había 139 duplicados (1.4%), que es un número bajo.

#✅ Si hay muchos repetidos, podemos eliminarlos con: cleaned_tweets = list(set(cleaned_tweets)).



# ========================
# Guardado de datos
# ========================

# Guardar solo la columna de tweets limpios en el archivo .pkl
output_path = "/home/mario/Documents/camiApp/data/CleanAndEmbeddins/cleaned_tweets.pkl"
joblib.dump(cleaned_tweets, output_path)

# Cargar y verificar los datos guardados
loaded_tweets = joblib.load(output_path)
print(f"✅ Tweets guardados correctamente: {len(loaded_tweets)} tweets")
print(f"🔍 Ejemplo de tweets limpios: {loaded_tweets[:5]}")  # Mostrar algunos ejemplos