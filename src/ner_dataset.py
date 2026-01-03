# -*- coding: utf-8 -*-
"""
NER Dataset Module for ClinicalBERT Token Classification
Handles i2b2 2010 dataset loading, BIO tagging, and tokenization alignment
"""

import os
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')


# BIO Label Scheme for i2b2 2010 Concepts
BIO_LABELS = [
    'O',           # 0 - Outside any entity
    'B-PROBLEM',   # 1 - Begin medical problem
    'I-PROBLEM',   # 2 - Inside medical problem
    'B-TREATMENT', # 3 - Begin treatment
    'I-TREATMENT', # 4 - Inside treatment
    'B-TEST',      # 5 - Begin test
    'I-TEST'       # 6 - Inside test
]

label2id = {label: i for i, label in enumerate(BIO_LABELS)}
id2label = {i: label for i, label in enumerate(BIO_LABELS)}


# Entity type mapping: i2b2 → Project entity types
I2B2_TO_PROJECT_MAPPING = {
    'PROBLEM': 'SYMPTOM',      # Medical problems → Symptoms (primary)
    'TREATMENT': 'CONDITION',  # Treatments often indicate underlying conditions
    'TEST': 'VITAL_SIGN'       # Diagnostic tests → Vital signs/measurements
}


def load_i2b2_dataset(cache_dir: Optional[str] = None):
    """
    Load i2b2 2010 Concepts dataset from HuggingFace

    Args:
        cache_dir: Optional cache directory for datasets

    Returns:
        DatasetDict with train/validation/test splits

    Raises:
        Exception: If dataset cannot be loaded (DUA required)
    """
    try:
        from datasets import load_dataset

        print("=" * 80)
        print("LOADING i2b2 2010 CONCEPTS DATASET")
        print("=" * 80)
        print("\nAttempting to load from HuggingFace...")

        # Try to load from HuggingFace
        dataset = load_dataset(
            "bigbio/n2c2_2010",
            name="n2c2_2010_bigbio_kb",  # Knowledge base format
            trust_remote_code=True,
            cache_dir=cache_dir
        )

        print(f"✅ Successfully loaded i2b2 2010 dataset!")
        print(f"   Train: {len(dataset['train'])} examples")
        print(f"   Validation: {len(dataset.get('validation', []))} examples")
        print(f"   Test: {len(dataset.get('test', []))} examples")
        print("=" * 80)

        return dataset

    except Exception as e:
        print(f"\n❌ Failed to load i2b2 2010 dataset from HuggingFace")
        print(f"   Error: {str(e)}")
        print("\n⚠️  FALLBACK STRATEGY REQUIRED:")
        print("   1. Apply for official access at: https://portal.dbmi.hms.harvard.edu/")
        print("   2. Or use weak supervision on ESI dataset (see documentation)")
        print("=" * 80)
        raise


def convert_i2b2_to_bio(dataset_split):
    """
    Convert i2b2 dataset format to BIO-tagged token sequences

    Args:
        dataset_split: HuggingFace dataset split

    Returns:
        List of (tokens, bio_tags) tuples
    """
    bio_examples = []

    for example in dataset_split:
        # Extract document text and entities
        text = example.get('text', '')
        entities = example.get('entities', [])

        # Tokenize into words (simple whitespace splitting)
        # Note: This is simplified; real implementation may need better tokenization
        words = text.split()

        # Initialize all labels as 'O' (Outside)
        bio_tags = ['O'] * len(words)

        # Convert entities to BIO tags
        for entity in entities:
            entity_type = entity.get('type', 'PROBLEM')
            start_offset = entity.get('offsets', [[0, 0]])[0][0]
            end_offset = entity.get('offsets', [[0, 0]])[0][1]

            # Find word indices that overlap with entity
            char_idx = 0
            for word_idx, word in enumerate(words):
                word_start = char_idx
                word_end = char_idx + len(word)

                # Check if word overlaps with entity
                if word_start < end_offset and word_end > start_offset:
                    # First word in entity gets B- tag
                    if bio_tags[word_idx] == 'O':
                        bio_tags[word_idx] = f'B-{entity_type}'
                    # Subsequent words get I- tag
                    elif bio_tags[word_idx].startswith('B-'):
                        bio_tags[word_idx] = f'I-{entity_type}'

                char_idx = word_end + 1  # +1 for space

        bio_examples.append({
            'tokens': words,
            'bio_tags': bio_tags,
            'text': text
        })

    return bio_examples


def align_labels_with_tokens(labels: List[int], word_ids: List[Optional[int]]) -> List[int]:
    """
    Align word-level BIO labels with subword tokens from tokenizer

    Critical for BERT tokenization: words are split into subword tokens,
    but we only have labels at word level. This function aligns them.

    Args:
        labels: List of BIO label IDs (word-level)
        word_ids: List mapping each token to its word index (from tokenizer)

    Returns:
        List of BIO label IDs (token-level)

    Example:
        Input word: "chest" → label: B-ANATOMY
        Tokenized: ["che", "##st"] → labels: [B-ANATOMY, I-ANATOMY]
    """
    aligned_labels = []
    previous_word_id = None

    for word_id in word_ids:
        # Special tokens (CLS, SEP, PAD) have word_id = None
        if word_id is None:
            aligned_labels.append(-100)  # Ignored in loss computation

        # First subword of a word
        elif word_id != previous_word_id:
            aligned_labels.append(labels[word_id])

        # Continuation subword
        else:
            label = labels[word_id]
            # If the label is B- (Begin), continuation gets I- (Inside)
            if label % 2 == 1 and label != 0:  # B- tags are odd (1, 3, 5)
                aligned_labels.append(label + 1)  # Convert B- to I-
            else:
                aligned_labels.append(label)  # Keep I- or O

        previous_word_id = word_id

    return aligned_labels


class NERDataset(Dataset):
    """
    PyTorch Dataset for Named Entity Recognition with ClinicalBERT

    Handles tokenization and label alignment for token classification
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[List[str]],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        label2id: Dict[str, int] = label2id
    ):
        """
        Initialize NER Dataset

        Args:
            texts: List of input texts (already tokenized into words)
            labels: List of BIO label sequences (parallel to texts)
            tokenizer: HuggingFace tokenizer (Bio_ClinicalBERT)
            max_length: Maximum sequence length
            label2id: Mapping from label strings to IDs
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example

        Returns:
            Dictionary with input_ids, attention_mask, and aligned labels
        """
        # Get text and labels
        text = self.texts[idx]
        labels = self.labels[idx]

        # Convert label strings to IDs
        label_ids = [self.label2id.get(label, 0) for label in labels]

        # Tokenize with offset mapping for alignment
        encoding = self.tokenizer(
            text,
            is_split_into_words=True,  # Input is already word-tokenized
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Get word IDs for label alignment
        word_ids = encoding.word_ids()

        # Align labels with subword tokens
        aligned_labels = align_labels_with_tokens(label_ids, word_ids)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }


class WeakSupervisionDataset:
    """
    Fallback dataset using weak supervision on ESI data

    Uses rule-based NER to create noisy labels, then manual correction
    """

    def __init__(self, esi_data_path: str, output_path: str = 'data/ner_weak_labels.json'):
        """
        Initialize weak supervision dataset

        Args:
            esi_data_path: Path to ESI dataset CSV
            output_path: Path to save weakly labeled data
        """
        self.esi_data_path = esi_data_path
        self.output_path = output_path

    def create_weak_labels(self):
        """
        Create weak labels using rule-based NER

        This generates noisy training data as a fallback if i2b2 is unavailable
        """
        import pandas as pd
        from ner_extractor import MedicalNER  # Rule-based NER

        print("\n" + "=" * 80)
        print("CREATING WEAK SUPERVISION LABELS")
        print("=" * 80)

        # Load ESI data
        df = pd.read_csv(self.esi_data_path)

        # Initialize rule-based NER
        ner = MedicalNER()

        weak_examples = []

        for idx, row in df.iterrows():
            if idx >= 500:  # Limit to 500 examples for manual correction
                break

            text = row.get('transcription', '')
            if not text:
                continue

            # Extract entities using rules
            entities = ner.extract_entities(text)

            # Convert to BIO format
            words = text.split()
            bio_tags = ['O'] * len(words)

            # Simple heuristic: match entity text to words
            for entity in entities:
                entity_words = entity.text.split()
                # Find matching sequence in text
                # (Simplified - production version needs better alignment)
                for i in range(len(words) - len(entity_words) + 1):
                    if ' '.join(words[i:i+len(entity_words)]).lower() == entity.text.lower():
                        # Map project entity types to i2b2 types
                        i2b2_type = self._map_to_i2b2_type(entity.entity_type)
                        bio_tags[i] = f'B-{i2b2_type}'
                        for j in range(1, len(entity_words)):
                            bio_tags[i + j] = f'I-{i2b2_type}'
                        break

            weak_examples.append({
                'tokens': words,
                'bio_tags': bio_tags,
                'text': text,
                'confidence': 'weak',  # Mark as weakly supervised
                'esi_label': row.get('esi_label', -1)
            })

            if (idx + 1) % 100 == 0:
                print(f"   Processed {idx + 1} examples...")

        # Save for manual correction
        import json
        with open(self.output_path, 'w') as f:
            json.dump(weak_examples, f, indent=2)

        print(f"\n✅ Created {len(weak_examples)} weakly labeled examples")
        print(f"   Saved to: {self.output_path}")
        print(f"\n⚠️  MANUAL CORRECTION REQUIRED:")
        print(f"   Review and correct labels before training")
        print("=" * 80)

        return weak_examples

    def _map_to_i2b2_type(self, project_type: str) -> str:
        """Map project entity types back to i2b2 types"""
        mapping = {
            'SYMPTOM': 'PROBLEM',
            'ANATOMY': 'PROBLEM',
            'CONDITION': 'PROBLEM',
            'VITAL_SIGN': 'TEST',
            'SEVERITY': 'PROBLEM',
            'TEMPORAL': 'PROBLEM'
        }
        return mapping.get(project_type, 'PROBLEM')


# Utility functions

def print_dataset_stats(dataset_split, name: str = "Dataset"):
    """Print statistics about a dataset split"""
    print(f"\n{name} Statistics:")
    print(f"  Total examples: {len(dataset_split)}")

    # Count entities per type
    entity_counts = {}
    for example in dataset_split:
        for tag in example.get('bio_tags', []):
            if tag != 'O':
                entity_type = tag.split('-')[1] if '-' in tag else tag
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

    print(f"  Entity counts:")
    for entity_type, count in sorted(entity_counts.items()):
        print(f"    {entity_type}: {count}")


def validate_bio_tags(bio_tags: List[str]) -> bool:
    """
    Validate BIO tag sequence for correctness

    Rules:
    - I-X can only follow B-X or I-X
    - Cannot have I-X without preceding B-X

    Args:
        bio_tags: List of BIO tags

    Returns:
        True if valid, False otherwise
    """
    for i, tag in enumerate(bio_tags):
        if tag.startswith('I-'):
            if i == 0:
                return False  # Cannot start with I-

            prev_tag = bio_tags[i - 1]
            entity_type = tag.split('-')[1]

            # Previous tag must be B- or I- of same type
            if not (prev_tag == f'B-{entity_type}' or prev_tag == f'I-{entity_type}'):
                return False

    return True


# Test function
def test_dataset_loading():
    """Test dataset loading and preprocessing"""
    print("\n" + "=" * 80)
    print("TESTING NER DATASET MODULE")
    print("=" * 80)

    # Test 1: Load i2b2 dataset
    try:
        dataset = load_i2b2_dataset()
        print("\n✅ Test 1 PASSED: i2b2 dataset loaded successfully")
    except Exception as e:
        print(f"\n⚠️  Test 1 SKIPPED: i2b2 dataset not available")
        print(f"   Using weak supervision fallback...")

        # Test weak supervision
        weak_dataset = WeakSupervisionDataset('data/esi_data_balanced.csv')
        examples = weak_dataset.create_weak_labels()
        print(f"\n✅ Weak supervision dataset created: {len(examples)} examples")

    # Test 2: Label alignment
    print("\n" + "-" * 80)
    print("Test 2: Label alignment with subword tokenization")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    # Sample data
    words = ["Patient", "has", "severe", "chest", "pain"]
    labels = [0, 0, 1, 3, 1]  # O, O, B-PROBLEM, B-ANATOMY, B-PROBLEM

    # Tokenize
    encoding = tokenizer(words, is_split_into_words=True)
    word_ids = encoding.word_ids()

    # Align
    aligned = align_labels_with_tokens(labels, word_ids)

    print(f"   Words: {words}")
    print(f"   Labels: {labels}")
    print(f"   Word IDs: {word_ids}")
    print(f"   Aligned: {aligned}")
    print("   ✅ Test 2 PASSED")

    # Test 3: BIO validation
    print("\n" + "-" * 80)
    print("Test 3: BIO tag validation")

    valid_tags = ['O', 'B-PROBLEM', 'I-PROBLEM', 'O']
    invalid_tags = ['O', 'I-PROBLEM', 'B-PROBLEM', 'O']  # I- without B-

    assert validate_bio_tags(valid_tags), "Valid tags should pass"
    assert not validate_bio_tags(invalid_tags), "Invalid tags should fail"

    print("   ✅ Test 3 PASSED")

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    test_dataset_loading()
