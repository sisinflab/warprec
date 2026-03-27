"""
Product Quantization utilities for model compression.
Simplified implementation without FAISS dependency.
"""

import numpy as np
from sklearn.cluster import KMeans


class ProductQuantizerSimple:
    """
    Simple Product Quantizer implementation using sklearn KMeans.
    Divides vectors into M subspaces, quantizes each independently.
    """
    
    def __init__(self, M=16, Ks=256, random_state=42):
        """
        Args:
            M: Number of subspaces/subquantizers
            Ks: Number of centroids per subspace (codebook size)
            random_state: Random seed for reproducibility
        """
        self.M = M
        self.Ks = Ks
        self.random_state = random_state
        self.subquantizers = []
        self.d_sub = None
        
    def fit(self, X):
        """
        Train the product quantizer on data X.
        
        Args:
            X: Data matrix of shape (N, D)
        """
        N, D = X.shape
        self.d_sub = D // self.M
        
        if D % self.M != 0:
            print(f"⚠️  Warning: D={D} not divisible by M={self.M}. "
                  f"Using d_sub={self.d_sub} (D={self.d_sub * self.M})")
        
        print(f"Training PQ: M={self.M}, Ks={self.Ks}, d_sub={self.d_sub}")
        
        # Train a KMeans for each subspace
        for m in range(self.M):
            start_idx = m * self.d_sub
            end_idx = start_idx + self.d_sub
            X_sub = X[:, start_idx:end_idx]
            
            kmeans = KMeans(n_clusters=self.Ks, random_state=self.random_state, n_init=10)
            kmeans.fit(X_sub)
            self.subquantizers.append(kmeans)
            print(f"  Subquantizer {m+1}/{self.M} trained (inertia: {kmeans.inertia_:.2f})")
    
    def encode_batch(self, X, batch_size=1024):
        """
        Encode vectors X into codes.
        
        Args:
            X: Data matrix of shape (N, D)
            batch_size: Batch size for processing
            
        Returns:
            codes: Integer array of shape (N, M) where each element is in [0, Ks)
        """
        N = X.shape[0]
        codes = np.zeros((N, self.M), dtype=np.uint8)
        
        for m in range(self.M):
            start_idx = m * self.d_sub
            end_idx = start_idx + self.d_sub
            X_sub = X[:, start_idx:end_idx]
            
            # Predict closest centroid for each sample
            codes[:, m] = self.subquantizers[m].predict(X_sub)
        
        return codes
    
    def decode(self, codes):
        """
        Reconstruct vectors from codes.
        
        Args:
            codes: Integer array of shape (N, M)
            
        Returns:
            X_reconstructed: Reconstructed data of shape (N, D)
        """
        N, M = codes.shape
        D = M * self.d_sub
        X_reconstructed = np.zeros((N, D), dtype=np.float32)
        
        for m in range(M):
            start_idx = m * self.d_sub
            end_idx = start_idx + self.d_sub
            centroids = self.subquantizers[m].cluster_centers_
            X_reconstructed[:, start_idx:end_idx] = centroids[codes[:, m]]
        
        return X_reconstructed


class ProductQuantizerFAISSIndexPQOPQ:
    """
    Alias for ProductQuantizerSimple to maintain API compatibility.
    """
    def __init__(self, M=16, Ks=256, random_state=42):
        self.pq = ProductQuantizerSimple(M=M, Ks=Ks, random_state=random_state)
    
    def fit(self, X):
        return self.pq.fit(X)
    
    def encode_batch(self, X, batch_size=1024):
        return self.pq.encode_batch(X, batch_size=batch_size)
    
    def decode(self, codes):
        return self.pq.decode(codes)
