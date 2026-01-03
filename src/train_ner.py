# -*- coding: utf-8 -*-
"""
Training Pipeline for ClinicalBERT NER Model
Fine-tunes Bio_ClinicalBERT on i2b2 2010 for Named Entity Recognition
"""

import os
import sys
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report
)
try:
    from src.ner_dataset import (
        load_i2b2_dataset,
        convert_i2b2_to_bio,
        NERDataset,
        BIO_LABELS,
        label2id,
        id2label,
        WeakSupervisionDataset
    )
except ImportError:
    from ner_dataset import (
        load_i2b2_dataset,
        convert_i2b2_to_bio,
        NERDataset,
        BIO_LABELS,
        label2id,
        id2label,
        WeakSupervisionDataset
    )
import warnings
warnings.filterwarnings('ignore')


def compute_metrics(p):
    """
    Compute entity-level metrics using seqeval

    This is critical for NER evaluation: we care about entity-level F1,
    not token-level accuracy. An entity is correct only if all its tokens
    are correctly predicted.

    Args:
        p: Predictions object from Trainer

    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens with label = -100)
    true_labels = []
    true_predictions = []

    for prediction, label in zip(predictions, labels):
        true_label = []
        true_pred = []

        for p_idx, l_idx in zip(prediction, label):
            if l_idx != -100:  # Skip special tokens
                true_label.append(id2label[l_idx])
                true_pred.append(id2label[p_idx])

        true_labels.append(true_label)
        true_predictions.append(true_pred)

    # Compute entity-level metrics
    results = {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }

    return results


def train_ner_model(
    data_source='i2b2',  # 'i2b2' or 'weak'
    output_dir='models/clinicalbert_ner',
    model_name="emilyalsentzer/Bio_ClinicalBERT",
    max_length=512,
    batch_size=8,
    learning_rate=2e-5,
    num_epochs=5,  # Optimal for fine-tuning (was 15)
    warmup_steps=100,
    weight_decay=0.01,
    early_stopping_patience=3,
    save_steps=500,
    logging_steps=50,
    seed=42
):
    """
    Train ClinicalBERT for Named Entity Recognition

    Args:
        data_source: 'i2b2' or 'weak' (weak supervision fallback)
        output_dir: Directory to save model checkpoints
        model_name: HuggingFace model identifier
        max_length: Maximum sequence length
        batch_size: Training batch size
        learning_rate: Learning rate for AdamW
        num_epochs: Number of training epochs
        warmup_steps: Number of warmup steps
        weight_decay: L2 regularization weight
        early_stopping_patience: Stop if no improvement for N epochs
        save_steps: Save checkpoint every N steps
        logging_steps: Log metrics every N steps
        seed: Random seed for reproducibility

    Returns:
        Trained Trainer object
    """
    print("\n" + "=" * 80)
    print("CLINICALBERT NER TRAINING PIPELINE")
    print("=" * 80)

    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load Dataset
    print("\n📁 STEP 1: Loading Dataset")
    print("-" * 80)

    if data_source == 'i2b2':
        try:
            # Load i2b2 2010 from HuggingFace
            dataset = load_i2b2_dataset()

            # Convert to BIO format
            train_bio = convert_i2b2_to_bio(dataset['train'])
            val_bio = convert_i2b2_to_bio(dataset.get('validation', dataset.get('test', dataset['train'][:100])))

            print(f"✅ Loaded i2b2 2010 dataset")
            print(f"   Train examples: {len(train_bio)}")
            print(f"   Validation examples: {len(val_bio)}")

        except Exception as e:
            print(f"❌ Failed to load i2b2 dataset: {e}")
            print(f"   Falling back to weak supervision...")
            data_source = 'weak'

    if data_source == 'weak':
        # Use weak supervision on ESI dataset
        weak_ds = WeakSupervisionDataset('data/esi_data_balanced.csv')

        # Check if weak labels exist
        if os.path.exists('data/ner_weak_labels.json'):
            import json
            with open('data/ner_weak_labels.json', 'r') as f:
                examples = json.load(f)
        else:
            examples = weak_ds.create_weak_labels()

        # Split into train/val (80/20)
        split_idx = int(0.8 * len(examples))
        train_bio = examples[:split_idx]
        val_bio = examples[split_idx:]

        print(f"✅ Loaded weak supervision dataset")
        print(f"   Train examples: {len(train_bio)}")
        print(f"   Validation examples: {len(val_bio)}")

    # Step 2: Initialize Tokenizer and Model
    print("\n🤖 STEP 2: Initializing Model")
    print("-" * 80)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(BIO_LABELS),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True  # Ignore size mismatch for classification head
    )

    print(f"✅ Loaded {model_name}")
    print(f"   Number of labels: {len(BIO_LABELS)}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Step 3: Create Datasets
    print("\n📊 STEP 3: Creating PyTorch Datasets")
    print("-" * 80)

    train_dataset = NERDataset(
        texts=[ex['tokens'] for ex in train_bio],
        labels=[ex['bio_tags'] for ex in train_bio],
        tokenizer=tokenizer,
        max_length=max_length,
        label2id=label2id
    )

    val_dataset = NERDataset(
        texts=[ex['tokens'] for ex in val_bio],
        labels=[ex['bio_tags'] for ex in val_bio],
        tokenizer=tokenizer,
        max_length=max_length,
        label2id=label2id
    )

    print(f"✅ Created datasets")
    print(f"   Train size: {len(train_dataset)}")
    print(f"   Validation size: {len(val_dataset)}")

    # Step 4: Data Collator
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True
    )

    # Step 5: Training Arguments
    print("\n⚙️  STEP 4: Configuring Training")
    print("-" * 80)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=16,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,  # Keep only 3 best checkpoints
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        gradient_accumulation_steps=2,  # Effective batch size = 16
        dataloader_num_workers=0,  # Windows compatibility
        report_to="none",  # Disable wandb/tensorboard
        seed=seed
    )

    print(f"✅ Training configuration:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size} (effective: {batch_size * 2})")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Max length: {max_length}")
    print(f"   Device: {'GPU (FP16)' if torch.cuda.is_available() else 'CPU'}")

    # Step 6: Initialize Trainer
    print("\n🎓 STEP 5: Initializing Trainer")
    print("-" * 80)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
    )

    print(f"✅ Trainer initialized")
    print(f"   Early stopping patience: {early_stopping_patience} epochs")

    # Step 7: Train
    print("\n🚀 STEP 6: Training Model")
    print("=" * 80)

    try:
        trainer.train()
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("⚠️  TRAINING INTERRUPTED BY USER")
        print("=" * 80)

    # Step 8: Evaluate on validation set
    print("\n📈 STEP 7: Final Evaluation")
    print("-" * 80)

    eval_results = trainer.evaluate()

    print(f"\n✅ Validation Results:")
    print(f"   Precision: {eval_results['eval_precision']:.4f}")
    print(f"   Recall:    {eval_results['eval_recall']:.4f}")
    print(f"   F1 Score:  {eval_results['eval_f1']:.4f}")

    # Check if target F1 is met
    if eval_results['eval_f1'] >= 0.80:
        print(f"\n🎯 TARGET F1 ACHIEVED (≥0.80)!")
    else:
        print(f"\n⚠️  Target F1 not reached. Consider:")
        print(f"   - More training data")
        print(f"   - Data augmentation")
        print(f"   - Hyperparameter tuning")
        print(f"   - Adding CRF layer")

    # Step 9: Save Best Model
    print("\n💾 STEP 8: Saving Model")
    print("-" * 80)

    best_model_path = os.path.join(output_dir, 'best_model')
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)

    print(f"✅ Model saved to: {best_model_path}")

    # Save training info
    import json
    training_info = {
        'data_source': data_source,
        'model_name': model_name,
        'num_train_examples': len(train_dataset),
        'num_val_examples': len(val_dataset),
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'final_f1': eval_results['eval_f1'],
        'final_precision': eval_results['eval_precision'],
        'final_recall': eval_results['eval_recall'],
        'labels': BIO_LABELS
    }

    with open(os.path.join(output_dir, 'training_info.json'), 'w') as f:
        json.dump(training_info, f, indent=2)

    print("\n" + "=" * 80)
    print("🎉 TRAINING PIPELINE COMPLETED")
    print("=" * 80)
    print(f"\n📁 Model location: {best_model_path}")
    print(f"📊 Final F1 Score: {eval_results['eval_f1']:.4f}")
    print("\n Next steps:")
    print("   1. Test with: python -c 'from src.train_ner import test_trained_model; test_trained_model()'")
    print("   2. Integrate with: src/ner_extractor.py (MedicalNER_ML class)")
    print("   3. Evaluate with: python evaluate_ner.py")
    print("=" * 80 + "\n")

    return trainer


def test_trained_model(model_path='models/clinicalbert_ner/best_model'):
    """
    Test the trained NER model on sample clinical text

    Args:
        model_path: Path to saved model checkpoint
    """
    print("\n" + "=" * 80)
    print("TESTING TRAINED NER MODEL")
    print("=" * 80)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()

    # Test cases
    test_cases = [
        "Patient has severe chest pain radiating to left arm.",
        "55-year-old male with crushing chest pain, BP 90/60, HR 110.",
        "Sudden onset dyspnea and diaphoresis. History of hypertension.",
        "Child with fever, cough, and mild abdominal pain for 2 days."
    ]

    for i, text in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST CASE {i}")
        print(f"{'=' * 80}")
        print(f"Input: {text}")
        print(f"{'-' * 80}")

        # Tokenize
        words = text.split()
        inputs = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors='pt',
            truncation=True,
            max_length=512
        )

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)

        # Convert predictions to labels
        predicted_labels = [id2label[p.item()] for p in predictions[0]]

        # Get word IDs
        word_ids = inputs.word_ids()

        # Extract entities
        print(f"\nExtracted Entities:")
        current_entity = None
        for word_id, label in zip(word_ids, predicted_labels):
            if word_id is None:
                continue

            if label.startswith('B-'):
                if current_entity:
                    print(f"  - {current_entity['text']}: {current_entity['type']}")
                current_entity = {
                    'text': words[word_id],
                    'type': label[2:]  # Remove 'B-'
                }
            elif label.startswith('I-') and current_entity:
                current_entity['text'] += f" {words[word_id]}"
            elif current_entity:
                print(f"  - {current_entity['text']}: {current_entity['type']}")
                current_entity = None

        if current_entity:
            print(f"  - {current_entity['text']}: {current_entity['type']}")

    print(f"\n{'=' * 80}")
    print("TESTING COMPLETED")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train ClinicalBERT for NER')
    parser.add_argument('--data-source', type=str, default='i2b2', choices=['i2b2', 'weak'],
                        help='Data source: i2b2 or weak supervision')
    parser.add_argument('--output-dir', type=str, default='models/clinicalbert_ner',
                        help='Output directory for model checkpoints')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--max-length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--test', action='store_true',
                        help='Test trained model instead of training')

    args = parser.parse_args()

    if args.test:
        test_trained_model(args.output_dir + '/best_model')
    else:
        trainer = train_ner_model(
            data_source=args.data_source,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length
        )
