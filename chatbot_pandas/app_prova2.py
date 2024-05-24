import streamlit as st
import os
import pickle
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatPremAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_community.embeddings import PremAIEmbeddings
from config import PREMAI_API_KEY

# Imposta la chiave API come variabile d'ambiente
os.environ["PREMAI_API_KEY"] = PREMAI_API_KEY

# Carica i chunk e le embeddings dal file pickle
with open('chunk_embeddings.pkl', 'rb') as f:
    all_chunks, chunk_embeddings = pickle.load(f)

# Inizializza l'embedder
model = "text-embedding-3-large"
embedder = PremAIEmbeddings(project_id=4316, model=model)

# Funzione per trovare i chunk più simili alla query
def find_most_similar_chunks(query_embedding, chunk_embeddings, all_chunks, top_k=4):
    similarities = cosine_similarity([query_embedding], chunk_embeddings).flatten()
    most_similar_indices = similarities.argsort()[-top_k:][::-1]
    return [(all_chunks[i], similarities[i]) for i in most_similar_indices]

# Streamlit app
st.title("Chatbot Pandas 🐼✨")

# Input di testo per la query dell'utente
user_query = st.text_input("Inserisci la tua domanda:")

if user_query:
    # Embed della query utente
    query_embedding = embedder.embed_query(user_query)

    # Trova i chunk più simili
    most_similar_chunks = find_most_similar_chunks(query_embedding, chunk_embeddings, all_chunks)




    # Combina i chunk più simili per generare una risposta
    combined_text = " ".join([chunk for chunk, _ in most_similar_chunks])

    # Genera una risposta utilizzando i chunk più rilevanti
    chat = ChatPremAI(project_id=4316)
    system_message = SystemMessage(content=combined_text)
    human_message = HumanMessage(content=user_query)
    response = chat.invoke([system_message, human_message])

    # Visualizza la risposta
    st.subheader("Risposta del chatbot:")
    st.write(response.content)
