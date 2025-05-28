from pymilvus import connections, Collection
import torch
import open_clip
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import zipfile, io
from PIL import Image

# Zip contenant les images
zip_path = "C:\\fromages\\archive.zip"

load_dotenv() # Charge les variables d'environnement depuis le fichier .env

# Chargement du modèle CLIP (doit être identique à celui utilisé pour l'indexation)
model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model.eval()

# Chargement des collections
connections.connect("default", host="localhost", port="19530")
collection_text = Collection("physiologie_du_gout")
collection_images = Collection("fromages_clip")

collection_text.load()
collection_images.load()

# query = "Une phrase qui parle d'un très bon fromage"
query = "Un fromage à pâte molle typiquement mangé au dessert"

# Chargement du modèle texte 
model_text = SentenceTransformer('all-MiniLM-L6-v2')  # Dimension de 384
embeddings = model_text.encode(query, convert_to_numpy=True)

# Recherche dans Milvus
results = collection_text.search(
    data=[embeddings],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"ef": 64}},
    limit=5,
    output_fields=["text"]
)

client = OpenAI()

# Affichage des résultats
print(f"\n🔎 Résultats pour la requête : \"{query}\"")
for hit in results[0]:
    text = hit.entity.get("text")
    print(f"🧀 Texte : {text} | Score : {hit.score:.4f}")

    # traduire la phrase
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"Translate the following sentence to English:\n\n{text}",
        temperature=0
    )

    # Traduction avec OpenAI
    translated = response.choices[0].text.strip()
    print(f"Translated : {translated}")

    # Encodage CLIP
    tokens = tokenizer([translated])
    with torch.no_grad():
        clip_vec = model.encode_text(tokens)
        clip_vec /= clip_vec.norm(dim=-1, keepdim=True)

    query_vector = clip_vec[0].cpu().numpy()

    results = collection_images.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"ef": 64}},
        limit=1,
        output_fields=["filename", "label"]
    )

    if results and results[0]:
        hit = results[0][0]
        filename = hit.entity.get("filename")
        label = hit.entity.get("label")
        print(f"🧀 {filename} | Label : {label} | Score : {hit.score:.4f}")

        with zipfile.ZipFile(zip_path, 'r') as archive:
            # Cherche le chemin complet dans le zip (avec dossier)
            matching_file = next((f for f in archive.namelist() if f.endswith(f"/{filename}")), None)
            
            if matching_file:
                print(f"✅ Fichier trouvé dans le zip : {matching_file}")
                with archive.open(matching_file) as file:
                    img_bytes = file.read()
                    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    
                    # Affichage avec matplotlib
                    plt.imshow(image)
                    plt.axis("off")
                    plt.title(filename)
                    plt.show()
            else:
                print("❌ Image non trouvée dans le zip")