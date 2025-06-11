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
        """Process text using trained NER model with confidence scores"""
        # Preprocess text
        text = text.replace('\n', ' ').strip()
        text = re.sub(r'\s+', ' ', text)
        
        doc = self.nlp(text)
        entities = {}
        
        # NER extraction with confidence scores
        for ent in doc.ents:
            if len(ent.text.strip()) > 1:  # Filter out single-character entities
                entities[ent.label_.lower()] = {
                    "text": ent.text.strip(),
                    "confidence": 0.85  # Base confidence for NER matches
                }
        
        # Enhanced regex patterns with named groups for our specific ID card format
        patterns = {
            "name": r"Name:\s*(?P<value>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            "college": r"College:\s*(?P<value>JNTU\s*Kakinada)",
            "roll_number": r"Roll\s*number:\s*(?P<value>22JNT\d{4})",
            "branch": r"Branch:\s*(?P<value>Computer\s*Science)",
            "valid_upto": r"Valid\s*upto:\s*(?P<value>20\d{2})"
        }
        
        # Combine regex matches with NER results
        for field, pattern in patterns.items():
            if field not in entities:
                matches = re.search(pattern, text, re.IGNORECASE)
                if matches:
                    entities[field] = {
                        "text": matches.group('value').strip(),
                        "confidence": 0.75  # Base confidence for regex matches
                    }
        
        # Post-process extracted entities
        for field, value in entities.items():
            # Clean the extracted text
            cleaned_text = re.sub(r'[^\w\s@.-]', '', value["text"])
            value["text"] = cleaned_text.strip()
            
            # Adjust confidence based on field-specific rules
            if field == "roll_number" and re.match(r'^22JNT\d{4}$', cleaned_text):
                value["confidence"] += 0.2
            elif field == "college" and "JNTU KAKINADA" in cleaned_text.upper():
                value["confidence"] += 0.2
            elif field == "branch" and "COMPUTER SCIENCE" in cleaned_text.upper():
                value["confidence"] += 0.2
            elif field == "valid_upto" and re.match(r'^20\d{2}$', cleaned_text):
                value["confidence"] += 0.2
            elif field == "name" and re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', cleaned_text):
                value["confidence"] += 0.2
            
            # Cap confidence at 1.0
            value["confidence"] = min(value["confidence"], 1.0)
        
        return entities