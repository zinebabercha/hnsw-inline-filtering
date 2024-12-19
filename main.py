from data_generators import UniformDataGenerator, SkewedDataGenerator , CorrelatedDataGenerator
from ann_evaluator import ANNEvaluator

def print_evaluation_results(metrics):
    """Print formatted evaluation results"""
    print("\nEvaluation Results:")
    print(f"Build Time: {metrics['build_time']:.3f} seconds")
    
    print("\nLabel Distribution:")
    for label, freq in metrics['label_distribution'].items():
        print(f"- Label {label}: {freq*100:.1f}%")
    
    print(f"\nQuery Latency:")
    print(f"- Unfiltered: {metrics['query_latency']['unfiltered']*1000:.2f} ms per query")
    for label, latency in metrics['query_latency']['filtered'].items():
        print(f"- Filtered (label {label}): {latency*1000:.2f} ms per query")
    
    print(f"\nFilter Specificity:")
    for label, specificity in metrics['filter_specificity'].items():
        print(f"- Label {label}: {specificity*100:.1f}%")
        
    print("\nRecall Scores:")
    for label, recall in metrics['recall_scores'].items():
        print(f"- Label {label}: {recall*100:.1f}%")
        
    print("\nFilter Friction:")
    print("Latency Overhead:")
    for label, overhead in metrics['filter_friction']['latency_overhead'].items():
        print(f"- Label {label}: {overhead:.2f}x")
    print("Recall Impact:")
    for label, impact in metrics['filter_friction']['recall_impact'].items():
        print(f"- Label {label}: {impact*100:.1f}%")

def run_evaluation(data_generator):
    """Run evaluation with specified data generator"""
    evaluator = ANNEvaluator()
    
    print(f"Generating labeled data using {data_generator.__class__.__name__}...")
    data, labels, distribution = data_generator.generate_data()
    evaluator.set_data(data, labels, distribution)
    
    print("Building index...")
    evaluator.build_index()
    
    print("Evaluating performance...")
    metrics = evaluator.evaluate_query_performance()
    
    print_evaluation_results(metrics)

def main():
    dim = 16
    num_elements = 3000
    
    #  Uniform distribution
    print("\nRunning evaluation with uniform distribution...")
    uniform_generator = UniformDataGenerator(num_elements, dim)
    run_evaluation(uniform_generator)
    
    # Skewed distribution
    print("\nRunning evaluation with skewed distribution...")
    skewed_generator = SkewedDataGenerator(num_elements, dim)
    run_evaluation(skewed_generator)

    # Correlated distribution
    print("\nRunning evaluation with correlated distribution...")
    correlated_generator = CorrelatedDataGenerator(num_elements, dim)
    run_evaluation(correlated_generator)


if __name__ == "__main__":
    main()