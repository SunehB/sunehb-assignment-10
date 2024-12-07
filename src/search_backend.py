import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

class ImageSearchEngine:
    def __init__(self, embeddings_path, image_folder, model):
        self.df = pd.read_pickle(embeddings_path)
        self.image_folder = image_folder
        self.model = model
        self.original_embeddings = np.vstack(self.df['embedding'].values)

        # Pre-fit PCA for all embeddings
        self.pca = PCA(n_components=512)  # CLIP ViT-B/32 default is 512-d
        self.pca.fit(self.original_embeddings)

    def get_pca_embeddings(self, emb, k):
        reduced = self.pca.transform(emb.reshape(1,-1))[:, :k]
        return reduced / np.linalg.norm(reduced, axis=1, keepdims=True)
    
    def get_pca_database(self, k):
        # Cache this if needed for efficiency
        def reduce_fn(x):
            reduced = self.pca.transform(x.reshape(1,-1))[:, :k]
            return reduced / np.linalg.norm(reduced, axis=1, keepdims=True)
        return self.df['embedding'].apply(reduce_fn)

    def search(self, query_embedding, top_k=5, use_pca=False, pca_k=512):
        if use_pca and pca_k < 512:
            query_reduced = self.get_pca_embeddings(query_embedding, pca_k)[0]
            reduced_db = self.get_pca_database(pca_k)
            # Compute similarity
            self.df['similarity'] = reduced_db.apply(lambda x: float(np.dot(query_reduced, x.T)))
        else:
            # Direct embeddings
            self.df['similarity'] = self.df['embedding'].apply(lambda x: float(np.dot(query_embedding, x)))
        
        # Return top K
        top_results = self.df.nlargest(top_k, 'similarity')
        return top_results[['file_name', 'similarity']]
