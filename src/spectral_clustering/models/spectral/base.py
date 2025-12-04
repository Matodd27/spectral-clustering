import numpy as np

class BaseSpectralClustering():
    def __init__(self, n_clusters: int, kind: str='symmetric'):
        self.n_clusters = n_clusters
        self.labels_ = None
        self.embedding_ = None
        self.kind = kind

    def fit(self, S: np.ndarray):
        from spectral_clustering.graphs.constructors import compute_laplacian
        from scipy.sparse.linalg import eigsh
        from sklearn.cluster import KMeans
        
        L = compute_laplacian(S, kind=self.kind)
        vals, vecs = eigsh(L, k=self.n_clusters, which='SM')
        self.embedding_ = vecs
        km = KMeans(self.n_clusters, n_init=20)
        self.labels_ = km.fit_predict(self.embedding_)
        return self
    
    def fit_predict(self, S):
        self.fit(S)
        return self.labels_