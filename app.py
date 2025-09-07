import streamlit as st
from document_processing import load_pdf, split_text, get_embeddings
from vector_store import VectorStore
from sentence_transformers import SentenceTransformer
import numpy as np

st.title("📚 AI/ML Retrieval-Augmented Generation (RAG) System")

# Load models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize vector store
vector_store = VectorStore(dim=384)
try:
    vector_store.load()
    st.success("Vector index loaded successfully.")
except:
    st.warning("No existing index found. Please upload documents.")

# Upload documents
uploaded_file = st.file_uploader("Upload PDF Document", type=['pdf'])
if uploaded_file is not None:
    text = load_pdf(uploaded_file)
    chunks = split_text(text)
    embeddings = get_embeddings(chunks)
    vector_store.add_embeddings(embeddings, chunks)
    vector_store.save()
    st.success("Document processed and embeddings stored.")

# Handle queries
query = st.text_input("Enter your query:")
if query:
    query_embedding = embedding_model.encode([query])
    results = vector_store.search(np.array(query_embedding), k=5)
    st.subheader("Relevant Information:")
    for i, res in enumerate(results):
        st.write(f"{i+1}. {res}")

    # Use LLM to generate final answer
    from transformers import pipeline
    generator = pipeline('text-generation', model='gpt2', device=-1)  # CPU
    context = " ".join(results)
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    output = generator(prompt, max_length=200, do_sample=True, temperature=0.7)
    st.subheader("Generated Answer:")
    st.write(output[0]['generated_text'].split("Answer:")[-1].strip())
