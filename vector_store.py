import faiss
import numpy as np
import pickle

class VectorStore:
    def __init__(self, dim, index_file='faiss_index.pkl'):
        self.index = faiss.IndexFlatL2(dim)
        self.index_file = index_file
        self.metadata = []

    def add_embeddings(self, embeddings, chunks):
        self.index.add(embeddings)
        self.metadata.extend(chunks)

    def save(self):
        faiss.write_index(self.index, 'faiss.index')
        with open(self.index_file, 'wb') as f:
            pickle.dump(self.metadata, f)

    def load(self):
        self.index = faiss.read_index('faiss.index')
        with open(self.index_file, 'rb') as f:
            self.metadata = pickle.load(f)

    def search(self, query_embedding, k=5):
        D, I = self.index.search(query_embedding, k)
        results = [self.metadata[i] for i in I[0]]
        return results
