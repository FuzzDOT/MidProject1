from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score as ss
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt
import os

class KMeans_Model:
    """
    A class to perform K-Means clustering and analyze clustering results using Silhouette and Elbow methods.
    
    Attributes
    ----------
    max_iters : int, default=300
        Maximum number of iterations for KMeans clustering.
    clusters_silhouette : int or None
        Number of clusters with the best silhouette score.
    best_silhouette_score : float
        Best silhouette score achieved during silhouette analysis.
    labels : array-like of shape (n_samples,)
        Cluster labels assigned to each point during fitting.
    
    """
    
    def __init__(self, max_iters=300):
        """
        Initialize a KMeans_Model instance.
        
        Parameters
        ----------
        max_iters : int, default=300
            Maximum number of iterations for KMeans clustering.
        """
        self.max_iters = max_iters
        self.clusters_silhouette = None
        self.clusters_elbow = None
        self.best_silhouette_score = -1
        self.labels_silhouette = None
        self.labels_elbow = None
        self.elbow_predicted_clusters = None
        
    def fit_silhouette_analysis(self, X):
        """
        Perform KMeans clustering on X and perform silhouette analysis to determine the optimal number of clusters.
        
        Saves a plot of Silhouette Scores vs. Number of Clusters.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        """
        n_clusters = range(2, 10)
        silhouette_scores = []
        
        for n in n_clusters:
            model = KMeans(n_clusters=n, n_init=10, max_iter=100)
            model.fit(X)
            
            silhouette_score = ss(X, model.labels_)
            silhouette_scores.append(silhouette_score)
            
            if silhouette_score > self.best_silhouette_score:
                self.best_silhouette_score = silhouette_score
                self.clusters_silhouette = n
                self.labels_silhouette = model.labels_
                
        
        # Create a plot to visualize Silhouette Scores
        plt.plot(n_clusters, silhouette_scores)
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Score vs. Number of Clusters")
        # plt.savefig(r'visualizations\silhouette_analysis.png')
        # path1 = ".."
        # path2 = "visualizations"
        # path3 = "silhouette_analysis.png"
        # path = os.path.join(path1, path2, path3)
        path = r"C:\Users\fuzzi\Desktop\AI MethA+ Workspace\Midterm\MidProject\visualizations\silhouette_analysis.png"
        plt.savefig(path)
        plt.close()
        
    def fit_elbow_analysis(self, X):
        """
        Perform KMeans clustering on X and use the elbow method to determine the optimal number of clusters.
        
        Saves a visualization of Elbow analysis.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        """
                
        Elbow_vis = KElbowVisualizer(KMeans(), k=10)
        Elbow_vis.fit(X)
        
        self.clusters_elbow = Elbow_vis.elbow_value_
        model = KMeans(n_clusters=self.clusters_elbow, n_init=10, max_iter=300)
        model.fit(X)
        self.labels_elbow = model.labels_
        self.elbow_predicted_clusters = model.fit_predict(X)
        
        # path1 = ".."
        # path2 = "visualizations"
        # path3 = "elbow_analysis.png"
        # path = os.path.join(path1, path2, path3)
        path = r"C:\Users\fuzzi\Desktop\AI MethA+ Workspace\Midterm\MidProject\visualizations\elbow_analysis.png"
        Elbow_vis.show(outpath=path)