import streamlit as st
import pandas as pd
import joblib

# ========================
# Cargar los resultados y el modelo desde los archivos .pkl
# ========================
topic_model = joblib.load("/home/mario/Documents/camiApp/bertopic_model.pkl")
topics = joblib.load("/home/mario/Documents/camiApp/topics_results.pkl")
sentiment_results = joblib.load("/home/mario/Documents/camiApp/sentiment_results.pkl")

# Convertir los resultados de sentimientos a un DataFrame
sentiment_df = pd.DataFrame(sentiment_results)

# ========================
# Visualización con Streamlit
# ========================

# Mostrar la tabla con los resultados de sentimiento
st.title("Análisis de Sentimientos de Tweets")
st.write(sentiment_df)

# Mostrar un gráfico de distribución de los temas
topic_counts = sentiment_df['topic'].value_counts()
st.subheader("Distribución de Temas")
st.bar_chart(topic_counts)

# Mostrar algunos tweets y su análisis de sentimiento
st.subheader("Tweets y Análisis de Sentimiento")
st.write(sentiment_df[['tweet', 'topic', 'sentiment', 'score']].head(10))