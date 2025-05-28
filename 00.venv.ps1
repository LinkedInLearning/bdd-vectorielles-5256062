mkdir venv
cd .\venv\
python -m venv .\vecteurs
.\vecteurs\Scripts\Activate.ps1

# installations
pip install sentence-transformers hf_xet

# pour démo FAISS
pip install faiss-cpu

# vérifier si on est en venv
pip -V
pip list # pour vérifier les paquets installés

# pour démo RAG
pip install python-dotenv openai  
