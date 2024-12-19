import numpy as np
import time
from collections import defaultdict
import hnswlib

class ANNEvaluator:
    """
    Core ANN evaluation logic independent of data generation strategy
    """
    def __init__(self, dim=16, num_elements=3000):
        self.dim = dim
        self.num_elements = num_elements
        self.metrics = defaultdict(list)
        self.data = None
        self.labels = None
        self.index = None
    
    def set_data(self, data, labels, distribution):
        """Set data and labels from any data generator"""
        self.data = data
        self.labels = labels
        self.metrics['label_distribution'] = distribution
    
    def build_index(self):
        """Build HNSW index with the current data"""
        if self.data is None:
            raise ValueError("Data must be set before building index")
            
        self.index = hnswlib.Index(space='cosine', dim=self.dim)
        self.index.init_index(max_elements=self.num_elements, ef_construction=100, M=16)
        self.index.set_ef(20)
        self.index.set_num_threads(1)
        
        start_time = time.time()
        self.index.add_items(self.data, ids=np.arange(self.num_elements))
        build_time = time.time() - start_time
        self.metrics['build_time'] = build_time
    
    def create_label_filter(self, target_label):
        """Create filter function for a specific label"""
        def filter_function(idx):
            return self.labels[idx] == target_label
        return filter_function
    
    def calculate_recall(self, filtered_results, true_results, query_points, target_label, k):
        """Calculate recall@k for filtered nearest neighbor search results"""
        recall = 0
        n_queries = len(query_points)
        
        target_mask = self.labels == target_label
        target_data = self.data[target_mask]
        target_indices = np.where(target_mask)[0]
        
        for i in range(n_queries):
            distances = np.linalg.norm(target_data - query_points[i], axis=1)
            true_neighbor_indices = target_indices[np.argsort(distances)[:k]]
            filtered_neighbor_indices = filtered_results[i]
            intersection = set(filtered_neighbor_indices) & set(true_neighbor_indices)
            recall += len(intersection) / k
        
        return recall / n_queries
    
    def evaluate_query_performance(self, num_queries=100, k=10):
        """Evaluate query performance with comprehensive metrics"""
        if self.index is None:
            raise ValueError("Index must be built before evaluation")
            
        query_points = np.float32(np.random.random((num_queries, self.dim)))
        
        #  unfiltered
        start_time = time.time()
        unfiltered_labels, unfiltered_distances = self.index.knn_query(query_points, k=k, num_threads=1)
        unfiltered_time = time.time() - start_time
        
        filter_times = {}
        recall_scores = {}
        filter_specificity = {}
        
        # Per each label metrics
        for label in ['a', 'b', 'c']:
            filter_func = self.create_label_filter(label)
            
            # Latency
            start_time = time.time()
            filtered_labels, filtered_distances = self.index.knn_query(
                query_points, k=k, num_threads=1, filter=filter_func
            )
            filter_time = time.time() - start_time
            filter_times[label] = filter_time / num_queries
            
            # Filter Specificity
            points_passing_filter = sum(filter_func(i) for i in range(self.num_elements))
            filter_specificity[label] = points_passing_filter / self.num_elements
            
            # Accuracy Impact (Recall)
            recall_scores[label] = self.calculate_recall(
                filtered_labels, 
                unfiltered_labels,
                query_points,
                label,
                k
            )
        
        self.metrics.update({
            'query_latency': {
                'unfiltered': unfiltered_time / num_queries,
                'filtered': filter_times
            },
            'filter_specificity': filter_specificity,
            'recall_scores': recall_scores,
            'filter_friction': {
                'latency_overhead': {
                    label: filter_times[label]/self.metrics['query_latency']['unfiltered'] 
                    for label in filter_times
                },
                'specificity': filter_specificity,
                'recall_impact': recall_scores
            }
        })
        
        return self.metrics