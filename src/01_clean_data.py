############################### Limpieza y preprocesamiento de tweets ###############################

import pandas as pd
import joblib
from utils.text_processing import preprocess_text

# ========================
#  Carga de datos
# ========================

def load_data(file_path):
    """Carga un archivo CSV y selecciona la columna de texto."""
    data = pd.read_csv(file_path, encoding='latin-1', header=None, usecols=[1])
    data.columns = ['text']
    return data

archivo_csv = "/home/mario/Documents/camiApp/data/raw/sentiment_tweets3.csv"
data = load_data(archivo_csv)

# Aplicar limpieza llamando a la función preprocess_text
data["cleaned_tweets"] = data["text"].apply(preprocess_text)

# Filtrar tweets vacíos después de la limpieza
filtered_tweets = [tweet for tweet in data["cleaned_tweets"].tolist() if tweet.strip()]

# ========================
# Guardado de datos
# ========================

# Guardar solo la columna de tweets limpios en el archivo .pkl
output_path = "/home/mario/Documents/camiApp/data/CleanAndEmbeddins/cleaned_tweets.pkl"
joblib.dump(filtered_tweets, output_path)



# Cargar y verificar los datos guardados
loaded_tweets = joblib.load(output_path)
print(f"✅ Tweets guardados correctamente: {len(loaded_tweets)} tweets")
print(f"🔍 Ejemplo de tweets limpios: {loaded_tweets[:5]}")  # Mostrar algunos ejemplos
