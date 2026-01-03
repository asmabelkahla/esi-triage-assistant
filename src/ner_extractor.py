# -*- coding: utf-8 -*-
"""
Module NER (Named Entity Recognition) pour Extraction d'Entités Médicales
Objectif: F1-score ≥ 0.80 pour symptômes/anatomie
"""

import re
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MedicalEntity:
    """Représente une entité médicale extraite"""
    text: str
    entity_type: str  # 'SYMPTOM', 'ANATOMY', 'SEVERITY', 'TEMPORAL', 'CONDITION'
    start_pos: int
    end_pos: int
    confidence: float


class MedicalNER:
    """
    Extracteur d'entités médicales utilisant pattern matching et règles
    Version baseline sans ML (pour démarrage rapide)

    Pour production: utiliser scispaCy + ClinicalBERT fine-tuned
    """

    def __init__(self):
        """Initialise les dictionnaires d'entités médicales"""

        # SYMPTOMS - Symptômes communs
        self.symptoms = {
            # Pain
            'pain', 'ache', 'aching', 'discomfort', 'soreness', 'tenderness',
            'sharp pain', 'dull pain', 'burning pain', 'stabbing pain', 'throbbing pain',
            'cramping', 'pressure', 'tightness', 'crushing pain',

            # Respiratory
            'dyspnea', 'shortness of breath', 'difficulty breathing', 'breathlessness',
            'wheezing', 'stridor', 'cough', 'coughing', 'hemoptysis',
            'sputum', 'congestion', 'chest tightness',

            # Cardiovascular
            'palpitations', 'chest pain', 'angina', 'syncope', 'dizziness',
            'lightheadedness', 'vertigo', 'diaphoresis', 'sweating',
            'edema', 'swelling',

            # Neurological
            'headache', 'migraine', 'confusion', 'altered mental status',
            'weakness', 'paralysis', 'numbness', 'tingling', 'seizure',
            'tremor', 'ataxia', 'diplopia', 'blurred vision',
            'slurred speech', 'aphasia',

            # GI
            'nausea', 'vomiting', 'emesis', 'diarrhea', 'constipation',
            'abdominal pain', 'bloating', 'cramping', 'heartburn',
            'dysphagia', 'hematemesis', 'melena', 'hematochezia',

            # General
            'fever', 'chills', 'fatigue', 'malaise', 'weight loss',
            'anorexia', 'night sweats', 'rash', 'pruritus', 'itching',
            'insomnia', 'anxiety', 'depression'
        }

        # ANATOMY - Parties du corps
        self.anatomy = {
            # Head/Neck
            'head', 'skull', 'face', 'eye', 'eyes', 'ear', 'ears',
            'nose', 'mouth', 'throat', 'neck', 'jaw', 'temple', 'forehead',

            # Chest/Thorax
            'chest', 'thorax', 'breast', 'lung', 'lungs', 'heart',
            'ribs', 'sternum', 'clavicle',

            # Abdomen
            'abdomen', 'stomach', 'belly', 'liver', 'spleen', 'kidney',
            'kidneys', 'pancreas', 'gallbladder', 'intestine', 'colon',
            'appendix', 'pelvis', 'groin',

            # Extremities
            'arm', 'arms', 'shoulder', 'elbow', 'wrist', 'hand', 'hands',
            'finger', 'fingers', 'leg', 'legs', 'hip', 'knee', 'ankle',
            'foot', 'feet', 'toe', 'toes', 'thigh', 'calf',

            # Back/Spine
            'back', 'spine', 'lumbar', 'thoracic', 'cervical',

            # Internal
            'brain', 'spinal cord', 'nerve', 'artery', 'vein',
            'muscle', 'bone', 'joint', 'skin'
        }

        # SEVERITY - Indicateurs de gravité
        self.severity_terms = {
            'severe': 0.9,
            'severe': 0.9,
            'extreme': 0.95,
            'worst': 0.95,
            'excruciating': 0.95,
            'unbearable': 0.9,
            'intense': 0.85,
            'significant': 0.75,
            'moderate': 0.5,
            'mild': 0.3,
            'slight': 0.2,
            'minimal': 0.15,
            '10/10': 0.95,
            '9/10': 0.9,
            '8/10': 0.8,
            '7/10': 0.7,
            '6/10': 0.6,
            '5/10': 0.5,
            '4/10': 0.4,
            '3/10': 0.3,
            '2/10': 0.2,
            '1/10': 0.1
        }

        # TEMPORAL - Indicateurs temporels
        self.temporal_patterns = {
            'sudden': 'acute',
            'suddenly': 'acute',
            'acute': 'acute',
            'abrupt': 'acute',
            'rapid': 'acute',
            'gradual': 'chronic',
            'gradually': 'chronic',
            'chronic': 'chronic',
            'persistent': 'chronic',
            'ongoing': 'chronic',
            'recurrent': 'recurrent',
            'intermittent': 'intermittent',
            'constant': 'constant',
            'continuous': 'constant',
            'progressive': 'progressive',
            'worsening': 'progressive',
            'improving': 'improving',
            'resolved': 'resolved'
        }

        # VITAL SIGNS PATTERNS
        self.vital_sign_patterns = {
            'blood_pressure': r'(?:BP|blood pressure)[:\s]*(\d{2,3})/(\d{2,3})',
            'heart_rate': r'(?:HR|heart rate|pulse)[:\s]*(\d{2,3})',
            'respiratory_rate': r'(?:RR|respiratory rate|respirations?)[:\s]*(\d{1,2})',
            'temperature': r'(?:temp|temperature|T)[:\s]*(\d{2,3}\.?\d?)[\s°]?[CF]?',
            'spo2': r'(?:SpO2|O2 sat|oxygen saturation)[:\s]*(\d{2,3})%?'
        }

        # MEDICAL CONDITIONS
        self.conditions = {
            # Cardiac
            'hypertension', 'HTN', 'high blood pressure', 'hypotension',
            'coronary artery disease', 'CAD', 'myocardial infarction', 'MI',
            'heart attack', 'atrial fibrillation', 'AFib', 'CHF', 'heart failure',
            'arrhythmia', 'angina',

            # Metabolic
            'diabetes', 'diabetes mellitus', 'DM', 'type 1 diabetes', 'type 2 diabetes',
            'hypoglycemia', 'hyperglycemia', 'diabetic ketoacidosis', 'DKA',
            'thyroid disease', 'hyperthyroidism', 'hypothyroidism',

            # Respiratory
            'asthma', 'COPD', 'emphysema', 'chronic bronchitis', 'pneumonia',
            'pulmonary embolism', 'PE', 'pulmonary edema',

            # Neurological
            'stroke', 'CVA', 'TIA', 'seizure disorder', 'epilepsy',
            "Alzheimer's", 'dementia', "Parkinson's",

            # Other
            'cancer', 'renal failure', 'kidney disease', 'liver disease',
            'cirrhosis', 'anemia', 'depression', 'anxiety'
        }

    def extract_entities(self, text: str) -> List[MedicalEntity]:
        """
        Extrait toutes les entités médicales du texte

        Args:
            text: Texte médical à analyser

        Returns:
            Liste des entités extraites
        """
        entities = []
        text_lower = text.lower()

        # Extract symptoms
        entities.extend(self._extract_symptoms(text, text_lower))

        # Extract anatomy
        entities.extend(self._extract_anatomy(text, text_lower))

        # Extract severity
        entities.extend(self._extract_severity(text, text_lower))

        # Extract temporal
        entities.extend(self._extract_temporal(text, text_lower))

        # Extract medical conditions
        entities.extend(self._extract_conditions(text, text_lower))

        # Extract vital signs
        entities.extend(self._extract_vital_signs(text, text_lower))

        # Remove duplicates and overlaps
        entities = self._remove_overlaps(entities)

        # Sort by position
        entities.sort(key=lambda x: x.start_pos)

        return entities

    def _extract_symptoms(self, text: str, text_lower: str) -> List[MedicalEntity]:
        """Extrait les symptômes"""
        entities = []

        for symptom in self.symptoms:
            # Search for whole word matches
            pattern = r'\b' + re.escape(symptom) + r'\b'
            for match in re.finditer(pattern, text_lower):
                entities.append(MedicalEntity(
                    text=text[match.start():match.end()],
                    entity_type='SYMPTOM',
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.85
                ))

        return entities

    def _extract_anatomy(self, text: str, text_lower: str) -> List[MedicalEntity]:
        """Extrait les parties anatomiques"""
        entities = []

        for part in self.anatomy:
            pattern = r'\b' + re.escape(part) + r'\b'
            for match in re.finditer(pattern, text_lower):
                entities.append(MedicalEntity(
                    text=text[match.start():match.end()],
                    entity_type='ANATOMY',
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.90
                ))

        return entities

    def _extract_severity(self, text: str, text_lower: str) -> List[MedicalEntity]:
        """Extrait les indicateurs de sévérité"""
        entities = []

        for term, score in self.severity_terms.items():
            pattern = r'\b' + re.escape(term) + r'\b'
            for match in re.finditer(pattern, text_lower):
                entities.append(MedicalEntity(
                    text=text[match.start():match.end()],
                    entity_type='SEVERITY',
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=score
                ))

        return entities

    def _extract_temporal(self, text: str, text_lower: str) -> List[MedicalEntity]:
        """Extrait les indicateurs temporels"""
        entities = []

        for term, category in self.temporal_patterns.items():
            pattern = r'\b' + re.escape(term) + r'\b'
            for match in re.finditer(pattern, text_lower):
                entities.append(MedicalEntity(
                    text=text[match.start():match.end()],
                    entity_type='TEMPORAL',
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.80
                ))

        # Extract duration patterns (e.g., "3 days", "2 hours")
        duration_pattern = r'\b(\d+)\s+(second|minute|hour|day|week|month|year)s?\b'
        for match in re.finditer(duration_pattern, text_lower):
            entities.append(MedicalEntity(
                text=text[match.start():match.end()],
                entity_type='TEMPORAL',
                start_pos=match.start(),
                end_pos=match.end(),
                confidence=0.85
            ))

        return entities

    def _extract_conditions(self, text: str, text_lower: str) -> List[MedicalEntity]:
        """Extrait les conditions médicales"""
        entities = []

        for condition in self.conditions:
            pattern = r'\b' + re.escape(condition) + r'\b'
            for match in re.finditer(pattern, text_lower):
                entities.append(MedicalEntity(
                    text=text[match.start():match.end()],
                    entity_type='CONDITION',
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.88
                ))

        return entities

    def _extract_vital_signs(self, text: str, text_lower: str) -> List[MedicalEntity]:
        """Extrait les signes vitaux"""
        entities = []

        for vital_type, pattern in self.vital_sign_patterns.items():
            for match in re.finditer(pattern, text_lower):
                entities.append(MedicalEntity(
                    text=text[match.start():match.end()],
                    entity_type='VITAL_SIGN',
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.95
                ))

        return entities

    def _remove_overlaps(self, entities: List[MedicalEntity]) -> List[MedicalEntity]:
        """Supprime les entités qui se chevauchent, garde celle avec la plus haute confiance"""
        if not entities:
            return entities

        # Trier par position puis par confiance
        entities.sort(key=lambda x: (x.start_pos, -x.confidence))

        non_overlapping = []
        for entity in entities:
            # Vérifier si elle chevauche une entité déjà acceptée
            overlaps = False
            for accepted in non_overlapping:
                if not (entity.end_pos <= accepted.start_pos or entity.start_pos >= accepted.end_pos):
                    overlaps = True
                    break

            if not overlaps:
                non_overlapping.append(entity)

        return non_overlapping

    def get_summary(self, entities: List[MedicalEntity]) -> Dict:
        """Génère un résumé structuré des entités extraites"""
        summary = {
            'total_entities': len(entities),
            'by_type': {},
            'chief_complaint': None,
            'severity_level': None,
            'temporal_onset': None,
            'anatomy_involved': [],
            'symptoms_list': [],
            'conditions_list': [],
            'vital_signs': {}
        }

        # Compter par type
        for entity in entities:
            entity_type = entity.entity_type
            if entity_type not in summary['by_type']:
                summary['by_type'][entity_type] = 0
            summary['by_type'][entity_type] += 1

            # Collecter les informations spécifiques
            if entity_type == 'SYMPTOM':
                summary['symptoms_list'].append(entity.text)
            elif entity_type == 'ANATOMY':
                summary['anatomy_involved'].append(entity.text)
            elif entity_type == 'CONDITION':
                summary['conditions_list'].append(entity.text)
            elif entity_type == 'SEVERITY' and summary['severity_level'] is None:
                summary['severity_level'] = entity.text
            elif entity_type == 'TEMPORAL' and summary['temporal_onset'] is None:
                summary['temporal_onset'] = entity.text

        # Déduire le chief complaint (symptôme le plus sévère + anatomie)
        if summary['symptoms_list'] and summary['anatomy_involved']:
            severity = summary['severity_level'] or ''
            symptom = summary['symptoms_list'][0]
            anatomy = summary['anatomy_involved'][0]
            summary['chief_complaint'] = f"{severity} {symptom} in {anatomy}".strip()
        elif summary['symptoms_list']:
            summary['chief_complaint'] = summary['symptoms_list'][0]

        # Dédupliquer les listes
        summary['symptoms_list'] = list(set(summary['symptoms_list']))
        summary['anatomy_involved'] = list(set(summary['anatomy_involved']))
        summary['conditions_list'] = list(set(summary['conditions_list']))

        return summary

    def generate_clinical_summary(self, entities: List[MedicalEntity], text: str) -> str:
        """
        Génère un résumé clinique structuré (format requis par cahier des charges)
        """
        summary = self.get_summary(entities)

        clinical_summary = []
        clinical_summary.append("=" * 60)
        clinical_summary.append("EXTRACTED CLINICAL INFORMATION")
        clinical_summary.append("=" * 60)

        # Chief Complaint
        if summary['chief_complaint']:
            clinical_summary.append(f"\nChief Complaint: {summary['chief_complaint']}")

        # Symptoms
        if summary['symptoms_list']:
            clinical_summary.append(f"\nSymptoms: {', '.join(summary['symptoms_list'][:5])}")

        # Anatomy Involved
        if summary['anatomy_involved']:
            clinical_summary.append(f"Anatomy Involved: {', '.join(summary['anatomy_involved'][:5])}")

        # Severity
        if summary['severity_level']:
            clinical_summary.append(f"Severity: {summary['severity_level']}")

        # Temporal
        if summary['temporal_onset']:
            clinical_summary.append(f"Onset: {summary['temporal_onset']}")

        # Past Medical History
        if summary['conditions_list']:
            clinical_summary.append(f"Past Medical History: {', '.join(summary['conditions_list'])}")

        # Statistics
        clinical_summary.append(f"\nTotal Entities Extracted: {summary['total_entities']}")
        clinical_summary.append("Breakdown by Type:")
        for entity_type, count in summary['by_type'].items():
            clinical_summary.append(f"  - {entity_type}: {count}")

        clinical_summary.append("=" * 60)

        return "\n".join(clinical_summary)


# ============================================================================
# ML-BASED NER (ClinicalBERT Fine-tuned)
# ============================================================================

class MedicalNER_ML:
    """
    ML-based Named Entity Recognition using fine-tuned ClinicalBERT

    Combines ML extraction (PROBLEM, TREATMENT, TEST from i2b2) with
    rule-based extraction (SEVERITY, TEMPORAL) for hybrid approach.

    Maintains same interface as MedicalNER for backward compatibility.
    """

    def __init__(self, model_path='models/clinicalbert_ner/best_model'):
        """
        Initialize ML-based NER

        Args:
            model_path: Path to fine-tuned ClinicalBERT model
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForTokenClassification

        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Check if model exists
        import os
        if not os.path.exists(model_path):
            print(f"⚠️  Warning: ML model not found at {model_path}")
            print(f"   Falling back to rule-based NER")
            print(f"   Train model with: python src/train_ner.py")
            self.model = None
            self.use_ml = False
            # Initialize rule-based fallback
            self._init_rule_based()
            return

        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            self.use_ml = True

            # Label mappings
            self.id2label = self.model.config.id2label
            self.label2id = self.model.config.label2id

            print(f"✅ Loaded ML-based NER model from {model_path}")

        except Exception as e:
            print(f"❌ Error loading ML model: {e}")
            print(f"   Falling back to rule-based NER")
            self.model = None
            self.use_ml = False
            self._init_rule_based()
            return

        # i2b2 to project entity mapping
        self.entity_type_mapping = {
            'PROBLEM': 'SYMPTOM',      # Primary mapping
            'TREATMENT': 'CONDITION',  # Medications indicate conditions
            'TEST': 'VITAL_SIGN'       # Diagnostic tests
        }

        # Initialize rule-based extractors for non-i2b2 entities
        self._init_rule_based()

    def _init_rule_based(self):
        """Initialize rule-based extractors for SEVERITY and TEMPORAL"""
        # SEVERITY patterns
        self.severity_terms = {
            'severe': 0.9, 'extreme': 0.95, 'worst': 0.95, 'excruciating': 0.95,
            'unbearable': 0.9, 'intense': 0.85, 'significant': 0.75,
            'moderate': 0.5, 'mild': 0.3, 'slight': 0.2, 'minimal': 0.15,
            '10/10': 0.95, '9/10': 0.9, '8/10': 0.8, '7/10': 0.7,
            '6/10': 0.6, '5/10': 0.5, '4/10': 0.4, '3/10': 0.3, '2/10': 0.2, '1/10': 0.1
        }

        # TEMPORAL patterns
        self.temporal_patterns = {
            'sudden': 'acute', 'suddenly': 'acute', 'acute': 'acute', 'abrupt': 'acute',
            'rapid': 'acute', 'gradual': 'chronic', 'gradually': 'chronic',
            'chronic': 'chronic', 'persistent': 'chronic', 'ongoing': 'chronic',
            'recurrent': 'recurrent', 'intermittent': 'intermittent',
            'constant': 'constant', 'continuous': 'constant',
            'progressive': 'progressive', 'worsening': 'progressive',
            'improving': 'improving', 'resolved': 'resolved'
        }

        # VITAL SIGN patterns
        self.vital_sign_patterns = {
            'blood_pressure': r'(?:BP|blood pressure)[:\s]*(\d{2,3})/(\d{2,3})',
            'heart_rate': r'(?:HR|heart rate|pulse)[:\s]*(\d{2,3})',
            'respiratory_rate': r'(?:RR|respiratory rate|respirations?)[:\s]*(\d{1,2})',
            'temperature': r'(?:temp|temperature|T)[:\s]*(\d{2,3}\.?\d?)[\s°]?[CF]?',
            'spo2': r'(?:SpO2|O2 sat|oxygen saturation)[:\s]*(\d{2,3})%?'
        }

    def extract_entities(self, text: str) -> List[MedicalEntity]:
        """
        Extract medical entities from text

        Hybrid approach:
        - ML extraction for SYMPTOM/CONDITION (from i2b2 PROBLEM/TREATMENT)
        - Rule-based extraction for SEVERITY/TEMPORAL/VITAL_SIGN

        Args:
            text: Clinical text to analyze

        Returns:
            List of MedicalEntity objects
        """
        import torch

        entities = []

        # 1. ML-based extraction (if model available)
        if self.use_ml and self.model is not None:
            ml_entities = self._extract_with_ml(text)
            entities.extend(ml_entities)

        # 2. Rule-based extraction for entities not in i2b2
        rule_entities = self._extract_rule_based(text)
        entities.extend(rule_entities)

        # 3. Remove overlaps (keep highest confidence)
        entities = self._remove_overlaps(entities)

        # 4. Sort by position
        entities.sort(key=lambda x: x.start_pos)

        return entities

    def _extract_with_ml(self, text: str) -> List[MedicalEntity]:
        """Extract entities using fine-tuned ClinicalBERT"""
        import torch

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
            padding=True
        )

        offset_mapping = inputs.pop('offset_mapping').squeeze().tolist()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2).squeeze()
            probabilities = torch.softmax(outputs.logits, dim=2).squeeze()

        # Convert to lists if single sequence
        if predictions.dim() == 0:
            predictions = [predictions.item()]
            probabilities = probabilities.unsqueeze(0)
        else:
            predictions = predictions.tolist()
            probabilities = probabilities.cpu().numpy()

        # Convert predictions to entities
        entities = self._predictions_to_entities(
            text, predictions, probabilities, offset_mapping
        )

        return entities

    def _predictions_to_entities(
        self,
        text: str,
        predictions: List[int],
        probabilities,
        offset_mapping: List[Tuple[int, int]]
    ) -> List[MedicalEntity]:
        """Convert BIO predictions to MedicalEntity objects"""
        import numpy as np

        entities = []
        current_entity = None

        for idx, (pred, (start, end)) in enumerate(zip(predictions, offset_mapping)):
            # Skip special tokens
            if start == end:
                continue

            label = self.id2label[pred]
            confidence = probabilities[idx][pred] if len(probabilities.shape) > 1 else probabilities[pred]
            if isinstance(confidence, np.ndarray):
                confidence = float(confidence)

            if label.startswith('B-'):
                # Save previous entity
                if current_entity:
                    entities.append(self._create_medical_entity(text, current_entity))

                # Start new entity
                entity_type = label[2:]  # Remove 'B-'
                current_entity = {
                    'start': start,
                    'end': end,
                    'type': entity_type,
                    'confidences': [confidence]
                }

            elif label.startswith('I-'):
                # Continue current entity
                if current_entity:
                    current_entity['end'] = end
                    current_entity['confidences'].append(confidence)

            else:  # 'O' tag
                # Save previous entity
                if current_entity:
                    entities.append(self._create_medical_entity(text, current_entity))
                    current_entity = None

        # Save last entity
        if current_entity:
            entities.append(self._create_medical_entity(text, current_entity))

        return entities

    def _create_medical_entity(self, text: str, entity_dict: dict) -> MedicalEntity:
        """Create MedicalEntity from entity dictionary"""
        import numpy as np

        # Map i2b2 entity type to project entity type
        i2b2_type = entity_dict['type']
        project_type = self.entity_type_mapping.get(i2b2_type, 'SYMPTOM')

        return MedicalEntity(
            text=text[entity_dict['start']:entity_dict['end']],
            entity_type=project_type,
            start_pos=entity_dict['start'],
            end_pos=entity_dict['end'],
            confidence=float(np.mean(entity_dict['confidences']))
        )

    def _extract_rule_based(self, text: str) -> List[MedicalEntity]:
        """Extract SEVERITY, TEMPORAL, and VITAL_SIGN entities using rules"""
        entities = []
        text_lower = text.lower()

        # SEVERITY extraction
        for term, score in self.severity_terms.items():
            pattern = r'\b' + re.escape(term) + r'\b'
            for match in re.finditer(pattern, text_lower):
                entities.append(MedicalEntity(
                    text=text[match.start():match.end()],
                    entity_type='SEVERITY',
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=score
                ))

        # TEMPORAL extraction
        for term in self.temporal_patterns:
            pattern = r'\b' + re.escape(term) + r'\b'
            for match in re.finditer(pattern, text_lower):
                entities.append(MedicalEntity(
                    text=text[match.start():match.end()],
                    entity_type='TEMPORAL',
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.80
                ))

        # VITAL_SIGN extraction
        for vital_type, pattern in self.vital_sign_patterns.items():
            for match in re.finditer(pattern, text_lower):
                entities.append(MedicalEntity(
                    text=text[match.start():match.end()],
                    entity_type='VITAL_SIGN',
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.95
                ))

        return entities

    def _remove_overlaps(self, entities: List[MedicalEntity]) -> List[MedicalEntity]:
        """Remove overlapping entities, keeping highest confidence"""
        if not entities:
            return entities

        # Sort by position then by confidence (descending)
        entities.sort(key=lambda x: (x.start_pos, -x.confidence))

        non_overlapping = []
        for entity in entities:
            # Check if overlaps with any accepted entity
            overlaps = False
            for accepted in non_overlapping:
                if not (entity.end_pos <= accepted.start_pos or
                        entity.start_pos >= accepted.end_pos):
                    overlaps = True
                    break

            if not overlaps:
                non_overlapping.append(entity)

        return non_overlapping

    def get_summary(self, entities: List[MedicalEntity]) -> Dict:
        """
        Generate structured summary of extracted entities
        Same interface as MedicalNER for compatibility
        """
        summary = {
            'total_entities': len(entities),
            'by_type': {},
            'chief_complaint': None,
            'severity_level': None,
            'temporal_onset': None,
            'anatomy_involved': [],
            'symptoms_list': [],
            'conditions_list': [],
            'vital_signs': {}
        }

        # Count by type
        for entity in entities:
            entity_type = entity.entity_type
            if entity_type not in summary['by_type']:
                summary['by_type'][entity_type] = 0
            summary['by_type'][entity_type] += 1

            # Collect specific information
            if entity_type == 'SYMPTOM':
                summary['symptoms_list'].append(entity.text)
            elif entity_type == 'ANATOMY':
                summary['anatomy_involved'].append(entity.text)
            elif entity_type == 'CONDITION':
                summary['conditions_list'].append(entity.text)
            elif entity_type == 'SEVERITY' and summary['severity_level'] is None:
                summary['severity_level'] = entity.text
            elif entity_type == 'TEMPORAL' and summary['temporal_onset'] is None:
                summary['temporal_onset'] = entity.text

        # Deduplicate lists
        summary['symptoms_list'] = list(set(summary['symptoms_list']))
        summary['anatomy_involved'] = list(set(summary['anatomy_involved']))
        summary['conditions_list'] = list(set(summary['conditions_list']))

        # Infer chief complaint
        if summary['symptoms_list'] and summary['anatomy_involved']:
            severity = summary['severity_level'] or ''
            symptom = summary['symptoms_list'][0]
            anatomy = summary['anatomy_involved'][0]
            summary['chief_complaint'] = f"{severity} {symptom} in {anatomy}".strip()
        elif summary['symptoms_list']:
            summary['chief_complaint'] = summary['symptoms_list'][0]

        return summary

    def generate_clinical_summary(self, entities: List[MedicalEntity], text: str) -> str:
        """
        Generate clinical summary (same interface as MedicalNER)
        """
        summary = self.get_summary(entities)

        clinical_summary = []
        clinical_summary.append("=" * 60)
        clinical_summary.append("EXTRACTED CLINICAL INFORMATION (ML-based NER)")
        clinical_summary.append("=" * 60)

        # Chief Complaint
        if summary['chief_complaint']:
            clinical_summary.append(f"\nChief Complaint: {summary['chief_complaint']}")

        # Symptoms
        if summary['symptoms_list']:
            clinical_summary.append(f"\nSymptoms: {', '.join(summary['symptoms_list'][:5])}")

        # Anatomy Involved
        if summary['anatomy_involved']:
            clinical_summary.append(f"Anatomy Involved: {', '.join(summary['anatomy_involved'][:5])}")

        # Severity
        if summary['severity_level']:
            clinical_summary.append(f"Severity: {summary['severity_level']}")

        # Temporal
        if summary['temporal_onset']:
            clinical_summary.append(f"Onset: {summary['temporal_onset']}")

        # Past Medical History
        if summary['conditions_list']:
            clinical_summary.append(f"Past Medical History: {', '.join(summary['conditions_list'])}")

        # Statistics
        clinical_summary.append(f"\nTotal Entities Extracted: {summary['total_entities']}")
        clinical_summary.append("Breakdown by Type:")
        for entity_type, count in summary['by_type'].items():
            clinical_summary.append(f"  - {entity_type}: {count}")

        clinical_summary.append("=" * 60)

        return "\n".join(clinical_summary)


# Fonction de test
def test_ner():
    """Teste l'extracteur NER"""
    ner = MedicalNER()

    case = """
    Patient is a 55-year-old male presenting with severe crushing chest pain
    radiating to left arm for past 30 minutes. Pain 9/10 severity, sudden onset
    at rest. Associated shortness of breath and diaphoresis. Vital signs:
    BP 90/60, HR 110, RR 24, SpO2 92%. Past medical history significant for
    hypertension, type 2 diabetes, and hyperlipidemia.
    """

    print("="*80)
    print("TEST: MEDICAL NER")
    print("="*80)
    print("\nINPUT TEXT:")
    print(case)

    entities = ner.extract_entities(case)

    print(f"\n📋 ENTITIES EXTRACTED: {len(entities)}")
    print("-"*80)

    # Group by type
    by_type = {}
    for entity in entities:
        if entity.entity_type not in by_type:
            by_type[entity.entity_type] = []
        by_type[entity.entity_type].append(entity)

    for entity_type, ents in by_type.items():
        print(f"\n{entity_type} ({len(ents)}):")
        for ent in ents:
            print(f"  - '{ent.text}' (confidence: {ent.confidence:.2f})")

    # Generate summary
    print("\n" + "="*80)
    print(ner.generate_clinical_summary(entities, case))


if __name__ == "__main__":
    test_ner()
