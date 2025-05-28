from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
from openai import OpenAI  # Nécessite une clé API
from dotenv import load_dotenv

# RAG : Retrieval-Augmented Generation

load_dotenv() # Charge les variables d'environnement depuis le fichier .env

# Notre "corpus" de documents
documents = [
    "Rome est la capitale de l'Italie.",
    "Le Colisée est un monument emblématique romain.",
    "Paris est célèbre pour sa gastronomie.",
    "Canberra est la capitale de l’Australie.",
    "La Tour Eiffel se trouve en Belgique."
]

# Chargement du modèle pré-entraîné
model = SentenceTransformer("all-MiniLM-L6-v2")

# Embedding et indexation FAISS
vectors = model.encode(documents).astype("float32")
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

question = "Où se trouve la Tour Eiffel ?"
query_vector = model.encode([question]).astype("float32")

# Recherche des documents les plus proches
D, I = index.search(query_vector, k=3)

# Sélection des passages pertinents
retrieved_docs = [documents[i] for i in I[0]]
context = "\n".join(retrieved_docs)

print("contexte : ", context)

client = OpenAI()

response = client.completions.create(
  model="gpt-3.5-turbo-instruct",
  prompt=f"Tu es un assistant utile qui répond en t'appuyant uniquement sur les documents suivants.\n{context}\n\nQuestion: {question}",
  temperature=0
)

print("Réponse générée :", response.choices[0].text)
