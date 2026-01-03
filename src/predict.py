# predict.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np

class ESIClassifier:
    def __init__(self, model_path='data/clinicalbert_esi_results/best_model'):
        """Initialiser le classifieur ESI"""
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
        
        # Définitions ESI
        self.esi_definitions = {
            1: {"label": "ESI-1", "description": "Réanimation immédiate", "color": "🟥"},
            2: {"label": "ESI-2", "description": "Urgence vitale", "color": "🟧"},
            3: {"label": "ESI-3", "description": "Urgence standard", "color": "🟨"},
            4: {"label": "ESI-4", "description": "Urgence mineure", "color": "🟩"},
            5: {"label": "ESI-5", "description": "Non-urgent", "color": "🟦"}
        }
    
    def load_model(self):
        """Charger le modèle et le tokenizer"""
        print(f"📦 Chargement du modèle depuis: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        print("✅ Modèle chargé avec succès!")
    
    def predict(self, text, return_probs=False):
        """
        Prédire le niveau ESI pour un texte donné
        
        Args:
            text (str): Description clinique du patient
            return_probs (bool): Retourner aussi les probabilités
        
        Returns:
            dict: Résultats de la prédiction
        """
        # Tokenization
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors='pt'
        ).to(self.device)
        
        # Prédiction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        # Formatage des résultats
        esi_level = predicted_class + 1
        confidence = probabilities[0][predicted_class].item()
        
        result = {
            "esi_level": esi_level,
            "esi_label": self.esi_definitions[esi_level]["label"],
            "description": self.esi_definitions[esi_level]["description"],
            "confidence": confidence,
            "color": self.esi_definitions[esi_level]["color"]
        }
        
        if return_probs:
            result["probabilities"] = {
                f"ESI-{i+1}": prob.item() for i, prob in enumerate(probabilities[0])
            }
        
        return result
    
    def predict_batch(self, texts):
        """
        Prédire pour un batch de textes
        
        Args:
            texts (list): Liste de descriptions cliniques
        
        Returns:
            list: Liste des résultats
        """
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results
    
    def display_prediction(self, text):
        """Afficher une prédiction de manière formatée"""
        result = self.predict(text, return_probs=True)
        
        print("\\n" + "="*60)
        print("🏥 ASSISTANT DE TRIAGE INTELLIGENT")
        print("="*60)
        print(f"\\n📝 Présentation du patient:")
        print(f"   {text[:200]}..." if len(text) > 200 else f"   {text}")
        
        print(f"\\n🎯 RÉSULTAT DU TRIAGE:")
        print(f"   {result['color']} {result['esi_label']} - {result['description']}")
        print(f"   Confiance: {result['confidence']:.2%}")
        
        print("\\n📊 DISTRIBUTION DES PROBABILITÉS:")
        for esi, prob in result['probabilities'].items():
            bar = "█" * int(prob * 20)
            print(f"   {esi}: {prob:.2%} {bar}")
        
        print("\\n" + "="*60)
        
        return result

# Interface en ligne de commande
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Classifieur ESI pour triage médical')
    parser.add_argument('--text', type=str, help='Texte à classifier')
    parser.add_argument('--file', type=str, help='Fichier CSV avec des textes à classifier')
    parser.add_argument('--model', type=str, default='data/clinicalbert_esi_results/best_model',
                       help='Chemin vers le modèle fine-tuné')
    
    args = parser.parse_args()
    
    # Initialiser le classifieur
    classifier = ESIClassifier(model_path=args.model)
    
    if args.text:
        # Prédiction unique
        result = classifier.display_prediction(args.text)
    
    elif args.file:
        # Prédiction par lot
        df = pd.read_csv(args.file)
        if 'transcription' not in df.columns:
            print("❌ Erreur: Le fichier doit contenir une colonne 'transcription'")
            return
        
        print(f"📁 Traitement de {len(df)} entrées...")
        results = classifier.predict_batch(df['transcription'].tolist())
        
        # Ajouter les résultats au DataFrame
        df['esi_predicted'] = [r['esi_level'] for r in results]
        df['esi_label'] = [r['esi_label'] for r in results]
        df['confidence'] = [r['confidence'] for r in results]
        
        # Sauvegarder les résultats
        output_file = args.file.replace('.csv', '_predicted.csv')
        df.to_csv(output_file, index=False)
        print(f"✅ Résultats sauvegardés dans: {output_file}")
        
        # Statistiques
        print(f"\\n📊 Statistiques des prédictions:")
        print(df['esi_label'].value_counts().sort_index())
    
    else:
        # Mode interactif
        print("\\n🤖 Classifieur ESI - Mode Interactif")
        print("Tapez 'quit' pour quitter")
        print("-" * 40)
        
        while True:
            text = input("\\n📝 Entrez la présentation du patient: ")
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if text.strip():
                classifier.display_prediction(text)

if __name__ == "__main__":
    main()