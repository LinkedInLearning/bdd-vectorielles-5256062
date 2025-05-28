from sentence_transformers import SentenceTransformer, util

phrase1 = "Comment trouver un bon restaurant italien à Paris ?"

# On va créer un vecteur de mots pour chaque phrase

# Chargement du modèle pré-entraîné
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encodage en vecteurs
# vect1 = model.encode(phrase1, convert_to_tensor=True)
vect1 = model.encode(phrase1)

# print(f"Vecteur 1 : {vect1[:10]} de type {type(vect1)}")
print(f"Vecteur 1 : {vect1} de type {type(vect1)} à {len(vect1)} dimensions")
