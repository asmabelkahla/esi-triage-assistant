# config.py
import os

class Config:
    # Chemins
    DATA_DIR = "data"
    MODELS_DIR = "models"
    RESULTS_DIR = "results"
    
    # Chemins spécifiques
    ESI_DATA_PATH = os.path.join(DATA_DIR, "esi_data.csv")
    MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "clinicalbert_esi_classifier")
    
    # Paramètres du modèle
    MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
    MAX_LENGTH = 256
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 10
    
    # ESI Definitions
    ESI_DEFINITIONS = {
        1: {
            "label": "ESI-1",
            "name": "Réanimation immédiate",
            "description": "Patient nécessitant une réanimation immédiate",
            "examples": ["Arrêt cardiaque", "Trauma majeur", "Arrêt respiratoire"],
            "color": "#FF0000"  # Rouge
        },
        2: {
            "label": "ESI-2",
            "name": "Urgence vitale",
            "description": "Patient avec détresse vitale",
            "examples": ["Douleur thoracique", "AVC", "Détresse respiratoire"],
            "color": "#FFA500"  # Orange
        },
        3: {
            "label": "ESI-3",
            "name": "Urgence standard",
            "description": "Patient stable nécessitant des soins urgents",
            "examples": ["Fracture", "Infection sévère", "Douleur abdominale"],
            "color": "#FFFF00"  # Jaune
        },
        4: {
            "label": "ESI-4",
            "name": "Urgence mineure",
            "description": "Problème moins urgent",
            "examples": ["Entorse", "Lacération simple", "Rhume"],
            "color": "#00FF00"  # Vert
        },
        5: {
            "label": "ESI-5",
            "name": "Non-urgent",
            "description": "Problème non urgent",
            "examples": ["Vaccination", "Certificat médical", "Suivi de routine"],
            "color": "#0000FF"  # Bleu
        }
    }
    
    # NER Configuration
    NER_MODEL_PATH = "models/clinicalbert_ner/best_model"
    NER_MODE = "hybrid"  # Options: "ml", "rules", "hybrid"
    NER_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for entity extraction
    NER_MAX_LENGTH = 512  # Maximum sequence length for NER

    # Entity type mappings (i2b2 to project)
    NER_ENTITY_TYPES = [
        'SYMPTOM', 'ANATOMY', 'SEVERITY',
        'TEMPORAL', 'CONDITION', 'VITAL_SIGN'
    ]

    # Hybrid mode: which types use ML vs rules
    NER_ML_ENTITY_TYPES = ['SYMPTOM', 'CONDITION', 'ANATOMY']  # From i2b2
    NER_RULE_ENTITY_TYPES = ['SEVERITY', 'TEMPORAL', 'VITAL_SIGN']  # Rule-based

    # Créer les répertoires si nécessaire
    @staticmethod
    def setup_directories():
        for directory in [Config.DATA_DIR, Config.MODELS_DIR, Config.RESULTS_DIR]:
            os.makedirs(directory, exist_ok=True)
        print("✅ Répertoires créés avec succès")