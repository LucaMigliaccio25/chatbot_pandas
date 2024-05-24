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

# Load documents and embeddings from the pickle file
with open('embeddings.pkl', 'rb') as f:
    documents, doc_result = pickle.load(f)

# Initialize the embedder
model = "embed-multilingual"
embedder = PremAIEmbeddings(project_id=4316, model=model)

# Streamlit app
st.title("Chatbot RAG con PremAI")

# Text input for user query
user_query = st.text_input("Inserisci la tua domanda:")

if user_query:
    # Embed the user's query
    query_embedding = embedder.embed_query(user_query)

    # Compute cosine similarities between the query and document embeddings
    similarities = cosine_similarity([query_embedding], doc_result).flatten()

    # Find the most similar document
    most_similar_index = np.argmax(similarities)
    most_similar_document = documents[most_similar_index]

    # Generate a response using the most relevant document
    chat = ChatPremAI(project_id=4316)
    system_message = SystemMessage(content=most_similar_document)
    human_message = HumanMessage(content=user_query)

    response=chat.invoke([system_message, human_message])

    # Display the response
    st.balloons()
    st.subheader("Risposta del chatbot:")
    st.write(response.content)