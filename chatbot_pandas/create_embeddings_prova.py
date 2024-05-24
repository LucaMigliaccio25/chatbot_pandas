"""
Questo script legge il contenuto di tutti i file .txt in una directory specificata, divide i documenti in chunk (con AI21SemanticTextSplitter) e ottiene le embeddings per ogni chunk utilizzando il modello di embedding di PremAI. Le embeddings e i chunk vengono salvati in un file pickle.
"""

import os
import glob
import pickle
from langchain_community.embeddings import PremAIEmbeddings
from config import PREMAI_API_KEY
from config import AI21_API_KEY
from langchain_ai21 import AI21SemanticTextSplitter

# Imposta la chiave API come variabile d'ambiente
os.environ["PREMAI_API_KEY"] = PREMAI_API_KEY
os.environ["AI21_API_KEY"] = AI21_API_KEY

# Definisci il modello e inizializza l'embedder
model = "text-embedding-3-large"
embedder = PremAIEmbeddings(project_id=4316, model=model)

# Directory contenente i file .txt
directory_path = 'txt'

# Ottieni una lista di tutti i file .txt nella directory
txt_files = glob.glob(os.path.join(directory_path, '*.txt'))

# Leggi il contenuto di ogni file e conservalo in una lista
documents = []
for file_path in txt_files:
    with open(file_path, 'r', encoding='utf-8') as file:
        document_content = file.read()
        documents.append(document_content)


# Dividi i documenti in chunk
all_chunks = []
chunk_embeddings = []

semantic_text_splitter = AI21SemanticTextSplitter()
for document in documents:
    chunks = semantic_text_splitter.split_text(document)
    all_chunks.extend(chunks)
    embeddings = embedder.embed_documents(chunks)
    chunk_embeddings.extend(embeddings)

# Stampa i chunk
print("Chunk embeddings created and saved successfully.")

# Salva i chunk e le embeddings in un file pickle
with open('chunk_embeddings.pkl', 'wb') as f:
    pickle.dump((all_chunks, chunk_embeddings), f)