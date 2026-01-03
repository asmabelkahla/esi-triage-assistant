# -*- coding: utf-8 -*-
"""
Fine-tuning ESI Model avec données personnalisées
Adapte le modèle existant à votre format de données (descriptions courtes)
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
import numpy as np
import json
from datetime import datetime

print("="*80)
print("🏥 FINE-TUNING ESI MODEL - Adaptation à vos données")
print("="*80)

# Configuration
CONFIG = {
    "base_model_path": "clinicalbert_esi_final/checkpoint-1758",  # Modèle à fine-tuner
    "tokenizer_name": "emilyalsentzer/Bio_ClinicalBERT",
    "custom_data_path": "custom_training_data.csv",
    "output_dir": "clinicalbert_esi_finetuned_custom",
    "num_train_epochs": 5,  # Peu d'epochs pour éviter l'overfitting
    "learning_rate": 1e-5,  # Learning rate plus faible pour fine-tuning
    "batch_size": 8,
    "warmup_steps": 50,
    "weight_decay": 0.01,
    "max_length": 512,
    "test_size": 0.2,
    "random_seed": 42
}

class ESIDataset(Dataset):
    """Dataset pour ESI classification"""
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx]) - 1  # ESI 1-5 → 0-4

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    """Calcule les métriques d'évaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # Balanced Accuracy
    balanced_acc = balanced_accuracy_score(labels, predictions)

    # F1 scores
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')

    # F1 par classe
    f1_per_class = f1_score(labels, predictions, average=None)

    return {
        'balanced_accuracy': balanced_acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_esi1': f1_per_class[0],
        'f1_esi2': f1_per_class[1],
        'f1_esi3': f1_per_class[2],
        'f1_esi4': f1_per_class[3],
        'f1_esi5': f1_per_class[4],
    }

def load_data(csv_path):
    """Charge et prépare les données"""
    print(f"\n📂 Chargement des données depuis {csv_path}...")

    df = pd.read_csv(csv_path)
    print(f"✅ {len(df)} exemples chargés")

    # Vérifier la distribution
    print("\n📊 Distribution des classes:")
    print("-"*80)
    for esi in range(1, 6):
        count = len(df[df['esi_label'] == esi])
        percentage = (count / len(df)) * 100
        print(f"  ESI-{esi}: {count:3d} exemples ({percentage:5.1f}%)")

    return df

def split_data(df, test_size=0.2, random_seed=42):
    """Sépare en train/validation sets"""
    print(f"\n✂️ Séparation des données (test_size={test_size})...")

    # Stratified split pour garder la distribution des classes
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=df['esi_label']
    )

    print(f"✅ Train: {len(train_df)} exemples")
    print(f"✅ Validation: {len(val_df)} exemples")

    return train_df, val_df

def prepare_datasets(train_df, val_df, tokenizer, max_length=512):
    """Crée les datasets PyTorch"""
    print("\n🔧 Création des datasets PyTorch...")

    train_dataset = ESIDataset(
        train_df['text'].values,
        train_df['esi_label'].values,
        tokenizer,
        max_length
    )

    val_dataset = ESIDataset(
        val_df['text'].values,
        val_df['esi_label'].values,
        tokenizer,
        max_length
    )

    print(f"✅ Train dataset: {len(train_dataset)} exemples")
    print(f"✅ Val dataset: {len(val_dataset)} exemples")

    return train_dataset, val_dataset

def train_model(model, train_dataset, val_dataset, config):
    """Fine-tune le modèle"""
    print("\n🚀 Lancement du fine-tuning...")
    print("-"*80)

    # Compatibilité avec anciennes et nouvelles versions de transformers
    try:
        # Tenter avec le nouveau nom (transformers >= 4.30)
        training_args = TrainingArguments(
            output_dir=config['output_dir'],
            num_train_epochs=config['num_train_epochs'],
            per_device_train_batch_size=config['batch_size'],
            per_device_eval_batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            warmup_steps=config['warmup_steps'],
            logging_dir=f"{config['output_dir']}/logs",
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="balanced_accuracy",
            greater_is_better=True,
            save_total_limit=3,
            fp16=torch.cuda.is_available(),  # Utiliser FP16 si GPU disponible
            report_to="none"  # Pas de reporting externe
        )
    except TypeError:
        # Ancien nom pour versions plus anciennes (transformers < 4.30)
        training_args = TrainingArguments(
            output_dir=config['output_dir'],
            num_train_epochs=config['num_train_epochs'],
            per_device_train_batch_size=config['batch_size'],
            per_device_eval_batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            warmup_steps=config['warmup_steps'],
            logging_dir=f"{config['output_dir']}/logs",
            logging_steps=10,
            evaluation_strategy="epoch",  # ← Ancien nom
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="balanced_accuracy",
            greater_is_better=True,
            save_total_limit=3,
            fp16=torch.cuda.is_available(),  # Utiliser FP16 si GPU disponible
            report_to="none"  # Pas de reporting externe
        )

    # Callbacks
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=2)
    ]

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )

    # Entraînement
    print("\n⏳ Entraînement en cours...")
    print("💡 Cela peut prendre 15-30 minutes selon votre matériel")
    print("-"*80)

    train_result = trainer.train()

    print("\n✅ Fine-tuning terminé!")
    print("-"*80)
    print(f"📊 Loss finale: {train_result.training_loss:.4f}")

    return trainer

def evaluate_model(trainer, val_dataset):
    """Évalue le modèle fine-tuné"""
    print("\n📊 Évaluation du modèle fine-tuné...")
    print("-"*80)

    eval_results = trainer.evaluate(val_dataset)

    print("\n🎯 RÉSULTATS:")
    print("-"*80)
    print(f"Balanced Accuracy: {eval_results['eval_balanced_accuracy']:.4f} ({eval_results['eval_balanced_accuracy']*100:.2f}%)")
    print(f"F1 Macro: {eval_results['eval_f1_macro']:.4f}")
    print(f"F1 Weighted: {eval_results['eval_f1_weighted']:.4f}")
    print("\nF1 par classe:")
    for esi in range(1, 6):
        f1 = eval_results[f'eval_f1_esi{esi}']
        print(f"  ESI-{esi}: {f1:.4f} ({f1*100:.1f}%)")

    return eval_results

def test_on_problematic_cases(model, tokenizer):
    """Teste sur les cas problématiques identifiés"""
    print("\n🧪 TEST SUR CAS PROBLÉMATIQUES...")
    print("="*80)

    test_cases = [
        {
            "text": "45-year-old male presenting for routine prescription refill. No acute complaints. Vital signs stable: BP 120/80, HR 72, SpO2 98%. Patient feels well, no pain, no distress.",
            "expected_esi": 5,
            "description": "Renouvellement prescription (CAS ORIGINAL PROBLÉMATIQUE)"
        },
        {
            "text": "28-year-old female with minor laceration to left hand from cooking, 2cm superficial cut, bleeding controlled with pressure. Vital signs normal.",
            "expected_esi": 4,
            "description": "Lacération mineure"
        },
        {
            "text": "55-year-old male with severe crushing chest pain radiating to left arm, dyspnea, diaphoresis. BP 90/60, HR 110, SpO2 92%. Suspected acute coronary syndrome.",
            "expected_esi": 1,
            "description": "Urgence vitale (SCA)"
        },
        {
            "text": "38-year-old woman requesting medical certificate for work after minor cold 2 days ago. No fever, vital signs normal. Patient feels well.",
            "expected_esi": 5,
            "description": "Certificat médical"
        }
    ]

    model.eval()
    results = []

    for case in test_cases:
        inputs = tokenizer(
            case['text'],
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            predicted_esi = int(torch.argmax(probs).item()) + 1
            confidence = float(probs[predicted_esi - 1]) * 100

        # Probabilités détaillées
        prob_dict = {i+1: float(probs[i]) * 100 for i in range(5)}

        correct = predicted_esi == case['expected_esi']

        print(f"\n{'='*80}")
        print(f"📋 {case['description']}")
        print(f"{'='*80}")
        print(f"Attendu: ESI-{case['expected_esi']}")
        print(f"Prédit:  ESI-{predicted_esi} ({confidence:.1f}%)")
        print(f"Résultat: {'✅ CORRECT' if correct else '❌ INCORRECT'}")
        print("\nProbabilités:")
        for esi in range(1, 6):
            emoji = ["🔴", "🟠", "🟡", "🟢", "🔵"][esi - 1]
            bar = "█" * int(prob_dict[esi] / 2)
            print(f"  {emoji} ESI-{esi}: {prob_dict[esi]:5.1f}% {bar}")

        results.append({
            'description': case['description'],
            'expected': case['expected_esi'],
            'predicted': predicted_esi,
            'confidence': confidence,
            'correct': correct,
            'probabilities': prob_dict
        })

    # Résumé
    correct_count = sum(1 for r in results if r['correct'])
    print(f"\n{'='*80}")
    print(f"📊 RÉSUMÉ: {correct_count}/{len(results)} cas corrects ({correct_count/len(results)*100:.0f}%)")
    print(f"{'='*80}")

    return results

def save_results(config, eval_results, test_results):
    """Sauvegarde les résultats"""
    print("\n💾 Sauvegarde des résultats...")

    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": config,
        "evaluation_metrics": {
            k: float(v) if isinstance(v, (np.floating, float)) else v
            for k, v in eval_results.items()
        },
        "test_cases_results": test_results
    }

    output_path = f"{config['output_dir']}/finetuning_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Résultats sauvegardés dans {output_path}")

def main():
    """Fonction principale"""

    # 1. Charger les données
    df = load_data(CONFIG['custom_data_path'])

    # Vérifier qu'on a assez de données
    if len(df) < 20:
        print("\n⚠️ ATTENTION: Vous avez moins de 20 exemples!")
        print("Pour un fine-tuning efficace, il est recommandé d'avoir au moins 50-100 exemples.")
        response = input("Voulez-vous continuer quand même? (y/n): ")
        if response.lower() != 'y':
            print("Fine-tuning annulé.")
            return

    # 2. Séparer les données
    train_df, val_df = split_data(df, CONFIG['test_size'], CONFIG['random_seed'])

    # 3. Charger le modèle et tokenizer
    print(f"\n📂 Chargement du modèle de base depuis {CONFIG['base_model_path']}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(CONFIG['tokenizer_name'])
        model = AutoModelForSequenceClassification.from_pretrained(
            CONFIG['base_model_path'],
            num_labels=5
        )
        print("✅ Modèle chargé avec succès")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        print("\nVérifiez que:")
        print(f"  1. Le dossier '{CONFIG['base_model_path']}' existe")
        print(f"  2. Vous avez accès à internet pour télécharger le tokenizer")
        return

    # 4. Préparer les datasets
    train_dataset, val_dataset = prepare_datasets(
        train_df, val_df, tokenizer, CONFIG['max_length']
    )

    # 5. Fine-tuner le modèle
    trainer = train_model(model, train_dataset, val_dataset, CONFIG)

    # 6. Évaluer
    eval_results = evaluate_model(trainer, val_dataset)

    # 7. Tester sur cas problématiques
    test_results = test_on_problematic_cases(trainer.model, tokenizer)

    # 8. Sauvegarder le modèle final
    print(f"\n💾 Sauvegarde du modèle fine-tuné...")
    final_model_path = f"{CONFIG['output_dir']}/final_model"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"✅ Modèle sauvegardé dans {final_model_path}")

    # 9. Sauvegarder les résultats
    save_results(CONFIG, eval_results, test_results)

    # 10. Instructions finales
    print("\n" + "="*80)
    print("🎉 FINE-TUNING TERMINÉ AVEC SUCCÈS!")
    print("="*80)
    print("\n📌 PROCHAINES ÉTAPES:")
    print("-"*80)
    print("1. Vérifiez les résultats ci-dessus")
    print(f"2. Le modèle fine-tuné est dans: {final_model_path}")
    print("3. Pour l'utiliser dans l'application, modifiez app_medical_simple.py:")
    print(f"   Ligne 278: Changez le chemin vers '{final_model_path}'")
    print("\n4. Relancez l'application:")
    print("   streamlit run app_medical_simple.py")
    print("\n5. Testez à nouveau votre cas de prescription refill")
    print("-"*80)
    print("\n💡 CONSEIL:")
    print("Si les résultats ne sont pas satisfaisants:")
    print("  - Ajoutez plus d'exemples similaires à vos cas réels")
    print("  - Relancez ce script avec le dataset enrichi")
    print("="*80)

if __name__ == "__main__":
    main()
