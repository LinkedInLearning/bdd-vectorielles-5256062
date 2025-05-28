from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np

documents = [
    "Je cherche un bon restaurant italien à Paris.",
    "Quelle est la capitale de l’Australie ?",
    "Les pandas vivent en Asie.",
    "Où peut-on manger italien à Paris ?",
    "Quels sont les meilleurs musées de Rome ?"
]

# Chargement du modèle pré-entraîné
model = SentenceTransformer("all-MiniLM-L6-v2")

# On va créer un vecteur de mots pour chaque phrase
vectors = model.encode(documents)
vectors = np.array(vectors).astype("float32")  # FAISS exige float32

dimension = vectors.shape[1]  # ici, 384

index = faiss.IndexFlatL2(dimension)  # index "brut", sans approximation
index.add(vectors)  # on stocke les vecteurs

query = "Où manger italien à Paris ?"
query_vector = model.encode([query]).astype("float32")

# Recherche des 3 vecteurs les plus proches
distances, indices = index.search(query_vector, k=3)

# Afficher les résultats
for i, idx in enumerate(indices[0]):
    print(f"Résultat {i+1}: {documents[idx]} (distance: {distances[0][i]:.4f})")
