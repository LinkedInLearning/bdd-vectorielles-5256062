from pymilvus import connections

# Connexion à Milvus en local

connections.connect("default", host="127.0.0.1", port="19530")

print("Connexion à Milvus réussie.")
