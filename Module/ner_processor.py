import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import json
import os
import random
from typing import List, Dict, Tuple
import re

class NERProcessor:
    def __init__(self, model_path: str = None):
        """Initialize NER processor with optional pre-trained model"""
        if model_path and os.path.exists(model_path):
            self.nlp = spacy.load(model_path)
        else:
            # Create a blank English model with only NER
            self.nlp = spacy.blank("en")
            if "ner" not in self.nlp.pipe_names:
                self.nlp.add_pipe("ner")
            
    def prepare_training_data(self, json_dir: str) -> List[Tuple[str, Dict]]:
        """Convert JSON data to spaCy training format with improved text preparation"""
        training_data = []
        
        for filename in os.listdir(json_dir):
            if filename.endswith('.json'):
                with open(os.path.join(json_dir, filename), 'r') as f:
                    data = json.load(f)
                    # Create a more natural text format
                    text_parts = []
                    for field, value in data['extracted_fields'].items():
                        text_parts.append(f"{field.replace('_', ' ').title()}: {value}")
                    text = ' '.join(text_parts)
                    
                    entities = []
                    # Convert fields to entity annotations with improved boundary detection
                    offset = 0
                    for field, value in data['extracted_fields'].items():
                        str_value = str(value)
                        start_idx = text.find(str_value, offset)
                        if start_idx != -1:
                            # Ensure we capture complete tokens
                            while start_idx > 0 and text[start_idx-1].isalnum():
                                start_idx -= 1
                            end_idx = start_idx + len(str_value)
                            while end_idx < len(text) and text[end_idx].isalnum():
                                end_idx += 1
                            
                            entities.append((start_idx, end_idx, field.upper()))
                            offset = end_idx
                    
                    training_data.append((text, {"entities": entities}))
        
        return training_data

    def train_model(self, training_data: List[Tuple[str, Dict]], output_dir: str, n_iter: int = 50):
        """Train NER model with improved parameters"""
        # Get the NER pipe
        ner = self.nlp.get_pipe("ner")
        
        # Add labels
        for _, annotations in training_data:
            for _, _, label in annotations["entities"]:
                ner.add_label(label)
        
        # Split training data
        random.shuffle(training_data)
        train_size = int(0.8 * len(training_data))
        train_data = training_data[:train_size]
        test_data = training_data[train_size:]
        
        # Initialize the model
        optimizer = self.nlp.initialize()
        
        # Training loop
        batch_size = 4
        for iteration in range(n_iter):
            losses = {}
            random.shuffle(train_data)
            batches = [train_data[i:i + batch_size] for i in range(0, len(train_data), batch_size)]
            
            for batch in batches:
                examples = []
                for text, annotations in batch:
                    doc = self.nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    examples.append(example)
                
                self.nlp.update(examples, drop=0.2, losses=losses)
            
            print(f"Iteration {iteration + 1}, Losses: {losses}")
        
        # Save model
        self.nlp.to_disk(output_dir)
        return test_data
    
    def evaluate_model(self, test_data: List[Tuple[str, Dict]]) -> Dict:
        """Evaluate model performance with detailed metrics"""
        results = {
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "examples": [],
            "per_entity_metrics": {}
        }
        
        entity_counts = {}
        for _, annotations in test_data:
            for _, _, label in annotations["entities"]:
                if label not in entity_counts:
                    entity_counts[label] = {"tp": 0, "fp": 0, "fn": 0}
        
        for text, annotations in test_data:
            doc = self.nlp(text)
            pred_entities = set((ent.start_char, ent.end_char, ent.label_) for ent in doc.ents)
            true_entities = set(annotations["entities"])
            
            # Count matches for each entity type
            for pred in pred_entities:
                if pred in true_entities:
                    entity_counts[pred[2]]["tp"] += 1
                else:
                    entity_counts[pred[2]]["fp"] += 1
                    
            for true in true_entities:
                if true not in pred_entities:
                    entity_counts[true[2]]["fn"] += 1
            
            # Store example results
            results["examples"].append({
                "text": text,
                "predicted": list(pred_entities),
                "actual": list(true_entities)
            })
        
        # Calculate per-entity metrics
        for entity, counts in entity_counts.items():
            precision = counts["tp"] / (counts["tp"] + counts["fp"]) if (counts["tp"] + counts["fp"]) > 0 else 0
            recall = counts["tp"] / (counts["tp"] + counts["fn"]) if (counts["tp"] + counts["fn"]) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results["per_entity_metrics"][entity] = {
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        
        # Calculate overall metrics
        total_tp = sum(counts["tp"] for counts in entity_counts.values())
        total_fp = sum(counts["fp"] for counts in entity_counts.values())
        total_fn = sum(counts["fn"] for counts in entity_counts.values())
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results.update({
            "precision": precision,
            "recall": recall,
            "f1": f1
        })
        
        return results
    
    def process_text(self, text: str) -> Dict:
        """Process text using trained NER model with improved pattern matching"""
        # Preprocess text
        text = text.replace('\n', ' ').strip()
        text = re.sub(r'\s+', ' ', text)
        
        doc = self.nlp(text)
        entities = {}
        
        # NER extraction with confidence threshold
        for ent in doc.ents:
            if len(ent.text.strip()) > 1:  # Filter out single-character entities
                entities[ent.label_.lower()] = ent.text.strip()
        
        # Enhanced regex patterns with named groups
        patterns = {
            "id_number": r"(?:ID|Number|#):\s*(?P<value>\b[A-Z0-9]{6,}\b)",
            "date": r"(?:Date|DOB):\s*(?P<value>\d{2}[-/]\d{2}[-/]\d{4})",
            "email": r"(?:Email|E-mail):\s*(?P<value>\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)",
            "phone": r"(?:Phone|Mobile|Tel):\s*(?P<value>(?:\+\d{1,3}[-\s]?)?\d{3}[-\s]?\d{3}[-\s]?\d{4})",
            "name": r"(?:Name):\s*(?P<value>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            "college": r"(?:College|Institution):\s*(?P<value>[A-Za-z\s.,&\-]+)",
            "branch": r"(?:Branch|Department):\s*(?P<value>[A-Za-z\s]+)"
        }
        
        # Combine regex matches with NER results
        for field, pattern in patterns.items():
            if field not in entities:
                matches = re.search(pattern, text, re.IGNORECASE)
                if matches:
                    entities[field] = matches.group('value').strip()
        
        # Post-process extracted entities
        for field, value in entities.items():
            # Remove common OCR artifacts
            value = re.sub(r'[^\w\s@.-]', '', value)
            value = value.strip()
            entities[field] = value
            
        return entities