import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def load_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text

def split_text(text, max_chunk_size=500):
    chunks = []
    paragraphs = text.split('\n')
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) < max_chunk_size:
            current_chunk += " " + para
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def get_embeddings(chunks):
    embeddings = embedding_model.encode(chunks)
    return np.array(embeddings)
