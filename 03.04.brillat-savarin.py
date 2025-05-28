from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import re
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

# Lecture du fichier texte
with open("pg22741.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Nettoyage et découpage en phrases
sentences = sent_tokenize(text, language='french')
sentences = [re.sub(r'(\s+|\n)', ' ', s).strip() for s in sentences]

print(f"Nombre de phrases à encoder : {len(sentences)}")

# Chargement du modèle texte 
model = SentenceTransformer('all-MiniLM-L6-v2')  # Dimension de 384
embeddings = model.encode(sentences, convert_to_numpy=True)

# Lecture du fichier texte
with open("pg22741.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Nettoyage et découpage en phrases
sentences = sent_tokenize(text, language='french')
sentences = [re.sub(r'(\s+|\n)', ' ', s).strip() for s in sentences]

# Affichage de la taille des embeddings
print(f"Taille des embeddings : {embeddings.shape}")

# Connexion à Milvus
connections.connect("default", host="localhost", port="19530")

# Définition du schéma de la collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000)
]
schema = CollectionSchema(fields, description="Embeddings de la Physiologie du Gout")

# Création de la collection
collection_name = "physiologie_du_gout"

# Vérification de l'existence de la collection
if utility.has_collection(collection_name):
    print(f"La collection '{collection_name}' existe déjà. Nous la supprimons.")
    utility.drop_collection(collection_name)

collection = Collection(name=collection_name, schema=schema)

# Insertion des données
data = [
    {"embedding": emb.tolist(), "text": sent}
    for emb, sent in zip(embeddings, sentences)
]

insert_result = collection.insert(data)
collection.flush()  # Assure la persistance des données

# Affichage du nombre d'éléments insérés
print(f"Nombre d'éléments insérés : {collection.num_entities}")

# Création de l'index
collection.create_index(field_name="embedding", index_params={
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {"M": 16, "efConstruction": 200}
})

# Chargement de la collection en mémoire
collection.load()

