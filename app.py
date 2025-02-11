import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ========================
# Cargar los resultados 
# ========================
#sentimentalTopics = pd.read_csv("/home/mario/Documents/camiApp/data/sentiment_topic_analysis.csv")

# Filtrar los tópicos que están en el rango de 0 a 9
#df_filtered = sentimentalTopics[sentimentalTopics['topic'].isin(range(10))]


import pandas as pd
import streamlit as st

# URL de descarga directa
url = 'https://drive.google.com/uc?id=1ll_Ml0pS3RTNLP087dv5xEHen-LbcBIQ'

# Cargar el DataFrame desde Google Drive
df_filtered = pd.read_csv(url)

# ========================
# Visualización con Streamlit
# ========================

# Crear un gráfico boxplot de sentimientos por tópico
plt.figure(figsize=(10, 6))
sns.boxplot(x='topic', y='sentiment_score', data=df_filtered)
plt.xlabel('Tópico')
plt.ylabel('Sentimiento')
plt.title('Distribución de Sentimientos por Tópico')
st.pyplot(plt)

# Gráfico interactivo
topic_selected = st.selectbox('Selecciona un Tópico:', df_filtered['topic'].unique())
filtered_data = df_filtered[df_filtered['topic'] == topic_selected]
plt.figure(figsize=(16, 16))
sns.boxplot(x='topic', y='sentiment_score', data=filtered_data)
plt.xlabel('Tópico')
plt.ylabel('Sentimiento')
plt.title(f'Distribución de Sentimientos para el Tópico {topic_selected}')
st.pyplot(plt)

