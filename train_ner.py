import os
from Module.ner_processor import NERProcessor

def train_ner_model():
    # Initialize NER processor
    ner = NERProcessor()
    
    # Prepare training data
    print("Preparing training data...")
    training_data = ner.prepare_training_data("json_data")
    print(f"Prepared {len(training_data)} training examples")
    
    # Create output directory
    os.makedirs("trained_models/ner", exist_ok=True)
    
    # Train model
    print("Training NER model...")
    test_data = ner.train_model(training_data, "trained_models/ner", n_iter=50)
    
    # Evaluate model
    print("\nEvaluating model...")
    results = ner.evaluate_model(test_data)
    
    # Print results
    print("\nOverall Metrics:")
    print(f"Precision: {results['precision']:.2%}")
    print(f"Recall: {results['recall']:.2%}")
    print(f"F1 Score: {results['f1']:.2%}")
    
    print("\nPer-Entity Metrics:")
    for entity, metrics in results['per_entity_metrics'].items():
        print(f"\n{entity}:")
        print(f"  Precision: {metrics['precision']:.2%}")
        print(f"  Recall: {metrics['recall']:.2%}")
        print(f"  F1 Score: {metrics['f1']:.2%}")

if __name__ == "__main__":
    train_ner_model()