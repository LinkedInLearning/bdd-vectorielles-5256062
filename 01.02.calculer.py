from sentence_transformers import SentenceTransformer, util
import torch

phrase1 = "Comment trouver un bon restaurant italien à Paris ?"
phrase2 = "Où peut-on manger italien à Paris ?"
phrase3 = "Les pandas vivent en Asie."

# On va créer un vecteur de mots pour chaque phrase

# Chargement du modèle pré-entraîné
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encodage en vecteurs
vect1 = model.encode(phrase1, convert_to_tensor=True)
vect2 = model.encode(phrase2, convert_to_tensor=True)
vect3 = model.encode(phrase3, convert_to_tensor=True)

# Calcul de la similarité cosinus entre les vecteurs
score_12 = util.cos_sim(vect1, vect2)
score_13 = util.cos_sim(vect1, vect3)

print(f"Similarité cosinus phrase 1 / 2 : {score_12.item():.2f}")
print(f"Similarité cosinus phrase 1 / 3 : {score_13.item():.2f}")

# Calcul de la distance euclidienne entre les vecteurs
score_12 = torch.linalg.vector_norm(vect1 - vect2, ord=2)
score_13 = torch.linalg.vector_norm(vect1 - vect3, ord=2)

print(f"distance euclidienne phrase 1 / 2 : {score_12.item():.2f}")
print(f"distance euclidienne phrase 1 / 3 : {score_13.item():.2f}")

# Calcul du produit scalaire entre les vecteurs
score_12 = (vect1 * vect2).sum()
score_13 = (vect1 * vect3).sum()

print(f"Produit scalaire phrase 1 / 2 : {score_12.item():.2f}")
print(f"Produit scalaire phrase 1 / 3 : {score_13.item():.2f}")
