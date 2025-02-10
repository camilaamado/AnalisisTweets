import pandas as pd  
import joblib
from transformers import AutoTokenizer, AutoModel
import torch
import pickle

# Cargar tweets limpios
cleaned_tweets = joblib.load("/home/mario/Documents/camiApp/data/CleanAndEmbeddins/cleaned_tweets.pkl")

# Asegurar que sean una lista de strings
if isinstance(cleaned_tweets, pd.DataFrame):
    cleaned_tweets = cleaned_tweets["text"].tolist()
elif isinstance(cleaned_tweets, pd.Series):
    cleaned_tweets = cleaned_tweets.tolist()

# Filtrar tweets vacíos antes de generar embeddings
cleaned_tweets = [tweet for tweet in cleaned_tweets if tweet.strip() != ""]

# Cargar el modelo y tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Función para obtener los embeddings
def get_embedding(text):
    tokens = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state.mean(dim=1).squeeze().numpy()

# Generar embeddings para todos los tweets
embeddings = [get_embedding(tweet) for tweet in cleaned_tweets]

# Guardar los embeddings en un archivo
with open("/home/mario/Documents/camiApp/data/CleanAndEmbeddins/sentiment_embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("Embeddings de sentimiento generados y guardados correctamente.")
