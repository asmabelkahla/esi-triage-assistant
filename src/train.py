# train.py
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Classe Dataset personnalisée
class ESIDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Fonction de calcul des métriques
def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    labels = p.label_ids
    
    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    weighted_f1 = f1_score(labels, preds, average='weighted')
    
    # Métrique critique : sous-triage (ESI 1-2 prédit comme 3-5)
    critical_errors = 0
    critical_total = 0
    
    for true, pred in zip(labels, preds):
        if true in [0, 1]:  # ESI 1-2 (labels 0-1)
            critical_total += 1
            if pred in [2, 3, 4]:  # Prédit comme ESI 3-5 (labels 2-4)
                critical_errors += 1
    
    under_triage_rate = critical_errors / critical_total if critical_total > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'critical_under_triage': under_triage_rate,
        'critical_cases_total': critical_total
    }

# Fonction principale d'entraînement
def train_esi_classifier(data_path='data/esi_data.csv', output_dir='clinicalbert_esi_results'):
    """Entraîner le modèle ClinicalBERT pour la classification ESI"""
    
    logger.info("🚀 Démarrage de l'entraînement du modèle ESI...")
    
    # 1. Charger les données
    logger.info("📊 Chargement des données...")
    df = pd.read_csv(data_path)
    
    # Ajouter label_index (0-4)
    df['label_index'] = df['esi_label'] - 1
    
    # Afficher la distribution des classes
    class_distribution = df['esi_label'].value_counts().sort_index()
    logger.info(f"Distribution des classes ESI: \n{class_distribution}")
    
    # 2. Split train/validation
    logger.info("🎯 Split des données...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['transcription'].fillna('').astype(str).tolist(),
        df['label_index'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df['label_index'].tolist()
    )
    
    logger.info(f"Train size: {len(train_texts)}, Validation size: {len(val_texts)}")
    
    # 3. Charger le tokenizer et le modèle
    logger.info("🤖 Chargement de ClinicalBERT...")
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=5,
        id2label={0: "ESI-1", 1: "ESI-2", 2: "ESI-3", 3: "ESI-4", 4: "ESI-5"},
        label2id={"ESI-1": 0, "ESI-2": 1, "ESI-3": 2, "ESI-4": 3, "ESI-5": 4},
        ignore_mismatched_sizes=True
    )
    
    # 4. Créer les datasets
    logger.info("📁 Création des datasets...")
    train_dataset = ESIDataset(train_texts, train_labels, tokenizer)
    val_dataset = ESIDataset(val_texts, val_labels, tokenizer)
    
    # 5. Configurer les arguments d'entraînement
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,  # Augmenté pour fine-tuning
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="critical_under_triage",  # Priorité : minimiser le sous-triage
        greater_is_better=False,
        save_total_limit=3,
        report_to="none",  # Désactiver wandb/tensorboard si non utilisé
        fp16=torch.cuda.is_available(),  # Mixed precision si GPU
        gradient_accumulation_steps=2,  # Pour simuler un batch size plus grand
        learning_rate=2e-5,  # Learning rate standard pour BERT
    )
    
    # 6. Créer le Trainer avec Early Stopping
    logger.info("🏋️‍♂️ Configuration du Trainer...")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # 7. Démarrer l'entraînement
    logger.info("🎬 Début de l'entraînement...")
    trainer.train()
    
    # 8. Évaluation finale
    logger.info("📈 Évaluation finale...")
    eval_results = trainer.evaluate()
    
    print("\\n" + "="*50)
    print("📊 RÉSULTATS FINAUX")
    print("="*50)
    for key, value in eval_results.items():
        print(f"{key}: {value:.4f}")
    
    # 9. Sauvegarder le modèle et le tokenizer
    logger.info("💾 Sauvegarde du modèle...")
    trainer.save_model(f"{output_dir}/best_model")
    tokenizer.save_pretrained(f"{output_dir}/best_model")
    
    # 10. Générer un rapport de classification
    logger.info("📋 Génération du rapport de classification...")
    generate_classification_report(trainer, val_dataset, output_dir)
    
    logger.info(f"✅ Entraînement terminé! Modèle sauvegardé dans: {output_dir}/best_model")
    
    return trainer, eval_results

def generate_classification_report(trainer, val_dataset, output_dir):
    """Générer un rapport détaillé de classification"""
    
    # Faire des prédictions
    predictions = trainer.predict(val_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids
    
    # Rapport de classification
    report = classification_report(
        labels,
        preds,
        target_names=["ESI-1", "ESI-2", "ESI-3", "ESI-4", "ESI-5"],
        output_dict=True
    )
    
    # Convertir en DataFrame pour une meilleure visualisation
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{output_dir}/classification_report.csv")
    
    # Matrice de confusion
    cm = confusion_matrix(labels, preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["ESI-1", "ESI-2", "ESI-3", "ESI-4", "ESI-5"],
                yticklabels=["ESI-1", "ESI-2", "ESI-3", "ESI-4", "ESI-5"])
    plt.title('Matrice de Confusion - Classification ESI')
    plt.ylabel('Vrai')
    plt.xlabel('Prédit')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300)
    plt.close()
    
    print("\\n📊 Rapport de Classification:")
    print(classification_report(labels, preds, target_names=["ESI-1", "ESI-2", "ESI-3", "ESI-4", "ESI-5"]))
    
    return report_df

# Fonction de test
def test_model(text, model_path='data/clinicalbert_esi_results/best_model'):
    """Tester le modèle sur un texte donné"""
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Prédiction
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=1).item()
        confidence = predictions[0][predicted_class].item()
    
    esi_level = predicted_class + 1
    esi_labels = ["Réanimation immédiate", "Urgence vitale", "Urgence standard", 
                  "Urgence mineure", "Non-urgent"]
    
    print(f"\\n🔍 Texte d'entrée: {text[:200]}...")
    print(f"📊 Prédiction: ESI-{esi_level} ({esi_labels[predicted_class]})")
    print(f"🎯 Confiance: {confidence:.2%}")
    print(f"📈 Distribution: {predictions[0].tolist()}")
    
    return esi_level, confidence

if __name__ == "__main__":
    # Exécuter l'entraînement
    trainer, results = train_esi_classifier()
    
    # Tester avec des exemples
    test_cases = [
        "45-year-old male with crushing chest pain radiating to left arm, shortness of breath, diaphoresis.",
        "Patient with mild cough and runny nose for 3 days, no fever.",
        "Young female with ankle sprain after falling, able to walk with some pain.",
        "Elderly patient with confusion and slurred speech, sudden onset.",
        "Child with fever and rash, no other symptoms."
    ]
    
    print("\\n" + "="*50)
    print("🧪 TESTS AVEC DES EXEMPLES")
    print("="*50)
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\\nTest {i}:")
        test_model(test_text)