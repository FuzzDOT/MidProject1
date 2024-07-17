from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class PCA_Handler:
    """A class to handle various dimensionality reduction operations."""
    
    def __init__(self, n_components, method='PCA'):
        """Initialize the DimensionalityReductionHandler with a specified number of components and method.
        
        Parameters:
            n_components (int): Number of dimensions to reduce to.
            method (str): The dimensionality reduction method to use ('PCA' or 't-SNE').
        """
        self.n_components = n_components
        self.method = method
        
        if method == 'PCA':
            self.reducer = PCA(n_components=n_components)
        elif method == 't-SNE':
            self.reducer = TSNE(n_components=n_components)
        else:
            raise ValueError("Method must be 'PCA' or 't-SNE'")
    
    def fit_transform(self, X):
        """Transform the input data X using the pre-fit dimensionality reduction model.
        
        Parameters:
            X (numpy.ndarray): Input data.
        
        Returns:
            numpy.ndarray: Transformed data after dimensionality reduction.
        """
        return self.reducer.fit_transform(X)