from abc import ABC, abstractmethod
import numpy as np
from sklearn.cluster import KMeans

class DataGenerator(ABC):
    """Abstract base class for generating labeled data"""
    
    def __init__(self, num_elements, dim):
        self.num_elements = num_elements
        self.dim = dim
    
    @abstractmethod
    def generate_data(self):
        """Generate data and labels"""
        pass
    
    def get_distribution_metrics(self, labels):
        """Calculate label distribution metrics"""
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts / len(labels)))

class UniformDataGenerator(DataGenerator):
    """Generates uniformly distributed labeled data"""
    
    def generate_data(self):
        """Generate uniform data with three labels distributed equally"""
        data = np.float32(np.random.random((self.num_elements, self.dim)))
        
        num_per_label = self.num_elements // 3
        labels = np.array(['a'] * num_per_label + 
                         ['b'] * num_per_label + 
                         ['c'] * (self.num_elements - 2 * num_per_label))
        
        p = np.random.permutation(len(data))
        data = data[p]
        labels = labels[p]
        
        distribution = self.get_distribution_metrics(labels)
        return data, labels, distribution

class SkewedDataGenerator(DataGenerator):
    """Generates skewed distributed labeled data"""
    
    def generate_data(self):
        """Generate skewed data with three labels distributed as 60%, 30%, 10%"""
        data = np.float32(np.random.random((self.num_elements, self.dim)))
        
        label_a_count = int(self.num_elements * 0.6)  # 60%
        label_b_count = int(self.num_elements * 0.3)  # 30%
        label_c_count = self.num_elements - label_a_count - label_b_count  # 10%
        
        labels = np.array(['a'] * label_a_count + 
                         ['b'] * label_b_count + 
                         ['c'] * label_c_count)
        
        p = np.random.permutation(len(data))
        data = data[p]
        labels = labels[p]
        
        distribution = self.get_distribution_metrics(labels)
        return data, labels, distribution
    



class CorrelatedDataGenerator(DataGenerator):
    """Generates data where certain features are correlated with labels"""
    
    def __init__(self, num_elements, dim, num_clusters=3):
        super().__init__(num_elements, dim)
        self.num_clusters = num_clusters
        
    def generate_data(self):
        """
        Generate correlated data where points with the same label tend to cluster together.
        Uses k-means to create distinct regions in feature space that correlate with labels.
        """
        # Generate cluster centers
        cluster_centers = np.random.random((self.num_clusters, self.dim))
        
        # Initialize k-means with predetermined centers
        kmeans = KMeans(n_clusters=self.num_clusters, init=cluster_centers, n_init=1)
        
        # Generate initial random data
        data = np.float32(np.random.random((self.num_elements, self.dim)))
        
        # Fit k-means to create clusters
        kmeans.fit(data)
        
        # Assign labels based on clusters
        cluster_assignments = kmeans.predict(data)
        
        # Create correlated data by moving points closer to their cluster centers
        correlation_strength = 0.7  # Controls how strongly points cluster
        for i in range(self.num_elements):
            cluster_idx = cluster_assignments[i]
            # Move points toward their cluster center
            data[i] = (1 - correlation_strength) * data[i] + correlation_strength * cluster_centers[cluster_idx]
        
        # Assign labels based on clusters
        labels = np.array(['a', 'b', 'c'])[cluster_assignments]
        
        # Calculate distribution
        distribution = self.get_distribution_metrics(labels)
        
        return np.float32(data), labels, distribution