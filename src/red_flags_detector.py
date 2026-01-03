# -*- coding: utf-8 -*-
"""
Module de Détection des Red Flags Critiques
Sensibilité cible: ≥ 95% (ne pas rater les urgences vitales)
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class RedFlag:
    """Représente un red flag détecté"""
    category: str
    description: str
    severity: str  # 'critical', 'high', 'moderate'
    pattern_matched: str
    confidence: float


class RedFlagsDetector:
    """
    Détecteur de signaux d'alerte critiques dans les présentations médicales
    Basé sur les guidelines cliniques ESI
    """

    def __init__(self):
        """Initialise les patterns de red flags par catégorie"""

        # CARDIOVASCULAIRE - Urgences cardiaques
        self.cardiovascular_patterns = {
            'chest_pain_radiation': {
                'patterns': [
                    r'chest\s+pain.*(?:radiating|radiation|radiates).*(?:arm|jaw|neck|shoulder|back)',
                    r'(?:radiating|radiation).*(?:chest|cardiac).*pain',
                    r'crushing.*chest.*pain',
                    r'pressure.*chest',
                    r'substernal.*pain'
                ],
                'description': 'Chest pain with radiation (cardiac pattern)',
                'severity': 'critical'
            },
            'syncope': {
                'patterns': [
                    r'\b(?:syncope|syncopal|passed out|lost consciousness|blacked out)\b',
                    r'sudden.*loss.*consciousness',
                    r'unresponsive.*episode'
                ],
                'description': 'Syncope or loss of consciousness',
                'severity': 'critical'
            },
            'severe_dyspnea': {
                'patterns': [
                    r'severe.*(?:dyspnea|shortness of breath|difficulty breathing)',
                    r'(?:dyspnea|shortness of breath).*severe',
                    r'cannot.*(?:breathe|catch.*breath)',
                    r'gasping.*air'
                ],
                'description': 'Severe dyspnea or respiratory distress',
                'severity': 'critical'
            },
            'cardiac_arrest': {
                'patterns': [
                    r'\b(?:cardiac arrest|arrested|CPR|resuscitation)\b',
                    r'no.*pulse',
                    r'pulseless',
                    r'v(?:entricular)?\s*fib(?:rillation)?',
                    r'asystole'
                ],
                'description': 'Cardiac arrest or need for resuscitation',
                'severity': 'critical'
            },
            'diaphoresis_cardiac': {
                'patterns': [
                    r'(?:diaphoresis|diaphoretic|sweating|perspiring).*(?:chest pain|cardiac)',
                    r'(?:chest pain|cardiac).*(?:diaphoresis|diaphoretic|sweating)',
                    r'cold.*(?:clammy|sweaty).*skin'
                ],
                'description': 'Diaphoresis with chest pain',
                'severity': 'high'
            }
        }

        # NEUROLOGIQUE - Urgences neurologiques
        self.neurological_patterns = {
            'altered_mental_status': {
                'patterns': [
                    r'\b(?:AMS|altered mental status|confusion|confused|disoriented)\b',
                    r'not.*oriented',
                    r'(?:mental|cognitive).*(?:change|decline|impairment)',
                    r'Glasgow.*(?:score|coma|scale).*(?:[0-9]|low)',
                    r'GCS.*(?:[0-9]|<\s*\d+)'
                ],
                'description': 'Altered mental status or confusion',
                'severity': 'critical'
            },
            'sudden_weakness': {
                'patterns': [
                    r'sudden.*(?:weakness|paralysis|hemiparesis|hemiplegia)',
                    r'(?:weakness|paralysis).*sudden',
                    r'cannot.*move.*(?:arm|leg|limb)',
                    r'facial.*droop',
                    r'slurred.*speech.*sudden'
                ],
                'description': 'Sudden weakness or paralysis (stroke pattern)',
                'severity': 'critical'
            },
            'worst_headache': {
                'patterns': [
                    r'worst.*headache.*(?:life|ever)',
                    r'(?:thunderclap|severe|intense).*headache.*sudden',
                    r'headache.*(?:10/10|severe).*sudden',
                    r'burst.*(?:headache|head)'
                ],
                'description': 'Worst headache of life (SAH pattern)',
                'severity': 'critical'
            },
            'seizure': {
                'patterns': [
                    r'\b(?:seizure|seizing|convulsion|fitting)\b',
                    r'status.*epilepticus',
                    r'tonic.*clonic',
                    r'postictal'
                ],
                'description': 'Seizure activity',
                'severity': 'critical'
            }
        }

        # RESPIRATOIRE - Détresse respiratoire
        self.respiratory_patterns = {
            'stridor': {
                'patterns': [
                    r'\bstridor\b',
                    r'high.*pitch.*breath',
                    r'upper.*airway.*obstruction'
                ],
                'description': 'Stridor (airway obstruction)',
                'severity': 'critical'
            },
            'inability_to_speak': {
                'patterns': [
                    r'cannot.*(?:speak|talk)',
                    r'unable.*(?:speak|talk).*breath',
                    r'too.*(?:short of breath|dyspneic).*speak'
                ],
                'description': 'Inability to speak due to dyspnea',
                'severity': 'critical'
            },
            'cyanosis': {
                'patterns': [
                    r'\bcyanosis\b',
                    r'\bcyanotic\b',
                    r'blue.*(?:lips|face|skin)',
                    r'turning.*blue'
                ],
                'description': 'Cyanosis (severe hypoxia)',
                'severity': 'critical'
            },
            'low_spo2': {
                'patterns': [
                    r'(?:SpO2|oxygen saturation|O2 sat).*(?:<|below|less than).*(?:8[0-9]|7[0-9]|[0-6][0-9])%?',
                    r'(?:hypoxia|hypoxic|desaturation)'
                ],
                'description': 'Severe hypoxia (SpO2 < 88%)',
                'severity': 'critical'
            }
        }

        # TRAUMA - Traumatismes graves
        self.trauma_patterns = {
            'penetrating_injury': {
                'patterns': [
                    r'\b(?:gunshot|GSW|stabbing|stab wound|penetrating)\b',
                    r'knife.*wound',
                    r'shot.*(?:chest|abdomen|head)',
                    r'impalement'
                ],
                'description': 'Penetrating trauma',
                'severity': 'critical'
            },
            'unstable_vitals': {
                'patterns': [
                    r'(?:BP|blood pressure).*(?:<|below).*(?:90|80|70)',
                    r'(?:systolic|SBP).*(?:<|below).*90',
                    r'(?:hypotension|hypotensive|shock)',
                    r'(?:HR|heart rate).*(?:>|above|over).*(?:1[2-9][0-9]|[2-9][0-9]{2})',
                    r'(?:tachycardia|tachycardic).*severe'
                ],
                'description': 'Unstable vital signs (shock pattern)',
                'severity': 'critical'
            },
            'major_trauma': {
                'patterns': [
                    r'(?:fell|fall).*(?:[2-9]|[1-9][0-9]).*(?:feet|meters|stories)',
                    r'high.*speed.*(?:collision|crash|accident)',
                    r'ejected.*vehicle',
                    r'rollover.*accident',
                    r'death.*(?:passenger|occupant)'
                ],
                'description': 'Major trauma mechanism',
                'severity': 'critical'
            },
            'head_trauma': {
                'patterns': [
                    r'head.*(?:trauma|injury).*(?:severe|loss of consciousness)',
                    r'skull.*fracture',
                    r'intracranial.*(?:bleeding|hemorrhage)',
                    r'blown.*pupil'
                ],
                'description': 'Severe head trauma',
                'severity': 'critical'
            }
        }

        # AUTRES - Autres urgences vitales
        self.other_patterns = {
            'uncontrolled_bleeding': {
                'patterns': [
                    r'(?:uncontrolled|massive|severe|profuse).*(?:bleeding|hemorrhage)',
                    r'(?:bleeding|hemorrhage).*(?:uncontrolled|massive|profuse)',
                    r'blood.*(?:spurting|gushing)',
                    r'hematemesis.*(?:large|massive)',
                    r'bright.*red.*blood.*(?:rectum|vomit)'
                ],
                'description': 'Uncontrolled bleeding',
                'severity': 'critical'
            },
            'severe_burns': {
                'patterns': [
                    r'(?:burn|burns).*(?:>|over|more than).*(?:[2-9][0-9]|100)%',
                    r'(?:third|3rd).*degree.*burn',
                    r'full.*thickness.*burn',
                    r'burn.*(?:face|airway|inhalation)'
                ],
                'description': 'Severe burns',
                'severity': 'critical'
            },
            'anaphylaxis': {
                'patterns': [
                    r'\banaphylaxis\b',
                    r'anaphylactic.*(?:shock|reaction)',
                    r'angioedema.*(?:severe|throat|airway)',
                    r'allergic.*reaction.*(?:severe|throat|breathing|swelling)'
                ],
                'description': 'Anaphylaxis signs',
                'severity': 'critical'
            },
            'acute_abdomen': {
                'patterns': [
                    r'(?:rigid|board-like).*abdomen',
                    r'peritoneal.*signs',
                    r'rebound.*tenderness.*severe',
                    r'acute.*abdomen'
                ],
                'description': 'Acute abdomen (surgical emergency)',
                'severity': 'high'
            },
            'overdose': {
                'patterns': [
                    r'\b(?:overdose|OD)\b',
                    r'ingested.*(?:pills|medication|poison)',
                    r'suicide.*attempt.*(?:pills|ingest)',
                    r'altered.*(?:overdose|ingestion)'
                ],
                'description': 'Drug overdose or poisoning',
                'severity': 'high'
            }
        }

        # Regrouper tous les patterns
        self.all_patterns = {
            'Cardiovascular': self.cardiovascular_patterns,
            'Neurological': self.neurological_patterns,
            'Respiratory': self.respiratory_patterns,
            'Trauma': self.trauma_patterns,
            'Other': self.other_patterns
        }

    def detect(self, text: str) -> List[RedFlag]:
        """
        Détecte les red flags dans le texte médical

        Args:
            text: Texte de présentation du patient

        Returns:
            Liste des red flags détectés
        """
        text_lower = text.lower()
        detected_flags = []

        for category, patterns_dict in self.all_patterns.items():
            for flag_name, flag_info in patterns_dict.items():
                for pattern in flag_info['patterns']:
                    matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                    for match in matches:
                        # Calculer la confiance basée sur la spécificité du match
                        confidence = self._calculate_confidence(match, text_lower)

                        red_flag = RedFlag(
                            category=category,
                            description=flag_info['description'],
                            severity=flag_info['severity'],
                            pattern_matched=match.group(0),
                            confidence=confidence
                        )
                        detected_flags.append(red_flag)
                        break  # Un seul match par flag

        # Dédupliquer et trier par sévérité
        detected_flags = self._deduplicate(detected_flags)
        detected_flags.sort(key=lambda x: (
            0 if x.severity == 'critical' else 1 if x.severity == 'high' else 2,
            -x.confidence
        ))

        return detected_flags

    def _calculate_confidence(self, match, text: str) -> float:
        """Calcule un score de confiance pour le match"""
        # Base confidence
        confidence = 0.85

        # Boost si le pattern est long (plus spécifique)
        if len(match.group(0)) > 20:
            confidence += 0.10

        # Boost si mentionné multiple fois
        pattern_count = text.count(match.group(0).lower())
        if pattern_count > 1:
            confidence += 0.05

        return min(confidence, 1.0)

    def _deduplicate(self, flags: List[RedFlag]) -> List[RedFlag]:
        """Élimine les duplicatas en gardant le plus confiant"""
        seen_descriptions = {}
        unique_flags = []

        for flag in flags:
            if flag.description not in seen_descriptions:
                seen_descriptions[flag.description] = flag
                unique_flags.append(flag)
            else:
                # Garder celui avec la confiance la plus élevée
                if flag.confidence > seen_descriptions[flag.description].confidence:
                    unique_flags.remove(seen_descriptions[flag.description])
                    seen_descriptions[flag.description] = flag
                    unique_flags.append(flag)

        return unique_flags

    def get_summary(self, flags: List[RedFlag]) -> Dict:
        """Génère un résumé des red flags détectés"""
        if not flags:
            return {
                'total': 0,
                'critical': 0,
                'high': 0,
                'moderate': 0,
                'categories': [],
                'risk_level': 'LOW'
            }

        summary = {
            'total': len(flags),
            'critical': sum(1 for f in flags if f.severity == 'critical'),
            'high': sum(1 for f in flags if f.severity == 'high'),
            'moderate': sum(1 for f in flags if f.severity == 'moderate'),
            'categories': list(set(f.category for f in flags)),
            'flags_list': [f.description for f in flags]
        }

        # Déterminer le niveau de risque global
        if summary['critical'] > 0:
            summary['risk_level'] = 'CRITICAL'
        elif summary['high'] >= 2:
            summary['risk_level'] = 'HIGH'
        elif summary['high'] > 0:
            summary['risk_level'] = 'MODERATE'
        else:
            summary['risk_level'] = 'LOW'

        return summary


# Fonction de test
def test_red_flags_detector():
    """Teste le détecteur avec des cas cliniques"""
    detector = RedFlagsDetector()

    # Test 1: Cardiac arrest (ESI-1)
    case1 = """
    68-year-old male found unresponsive at home. EMS found patient in cardiac arrest
    with no pulse. CPR initiated. Received defibrillation for ventricular fibrillation.
    Return of spontaneous circulation after 8 minutes. BP 85/50 on norepinephrine,
    HR 110, SpO2 88% on 100% FiO2. GCS 4. History of CAD, MI, HTN, diabetes.
    """

    # Test 2: Acute coronary syndrome (ESI-2)
    case2 = """
    55-year-old male with crushing chest pain radiating to left arm for 30 minutes.
    Pain 9/10, associated with shortness of breath and diaphoresis. Sudden onset at rest.
    History of hypertension and diabetes.
    """

    # Test 3: Stroke (ESI-2)
    case3 = """
    72-year-old female with sudden weakness on right side, facial droop, and slurred speech
    starting 1 hour ago. Unable to move right arm. History of atrial fibrillation.
    """

    print("="*80)
    print("TEST: RED FLAGS DETECTOR")
    print("="*80)

    for i, case in enumerate([case1, case2, case3], 1):
        print(f"\n{'='*80}")
        print(f"CASE {i}:")
        print(f"{'='*80}")
        print(case.strip())

        flags = detector.detect(case)
        summary = detector.get_summary(flags)

        print(f"\n🚨 RED FLAGS DETECTED: {summary['total']}")
        print(f"   Risk Level: {summary['risk_level']}")
        print(f"   Critical: {summary['critical']} | High: {summary['high']} | Moderate: {summary['moderate']}")
        print(f"   Categories: {', '.join(summary['categories'])}")

        if flags:
            print("\n   Detected Flags:")
            for flag in flags:
                severity_icon = "🔴" if flag.severity == "critical" else "🟠" if flag.severity == "high" else "🟡"
                print(f"   {severity_icon} [{flag.category}] {flag.description}")
                print(f"      Pattern: '{flag.pattern_matched}' (confidence: {flag.confidence:.2f})")


if __name__ == "__main__":
    test_red_flags_detector()
