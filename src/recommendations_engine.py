# -*- coding: utf-8 -*-
"""
Moteur de Recommandations d'Examens et Tests
Basé sur les guidelines cliniques et les patterns symptomatiques
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class Recommendation:
    """Représente une recommandation d'examen"""
    exam: str
    priority: str  # 'STAT', 'URGENT', 'ROUTINE'
    rationale: str
    category: str  # 'Lab', 'Imaging', 'Consultation', 'Monitoring'
    order: int  # Ordre de priorité


class RecommendationsEngine:
    """
    Moteur de recommandations basé sur :
    - Symptômes détectés
    - Red flags
    - Niveau ESI
    - Guidelines cliniques standards
    """

    def __init__(self):
        """Initialise les mappings symptômes → examens"""

        # CARDIOVASCULAR RECOMMENDATIONS
        self.cardio_recommendations = {
            'chest_pain': {
                'exams': [
                    ('ECG 12-leads', 'STAT', 'Chest pain evaluation - rule out ACS', 'Monitoring', 1),
                    ('Troponin I/T', 'STAT', 'Cardiac biomarker for myocardial injury', 'Lab', 2),
                    ('Chest X-ray', 'URGENT', 'Rule out pneumothorax, widened mediastinum', 'Imaging', 3),
                    ('D-dimer', 'URGENT', 'If PE suspected (chest pain + dyspnea)', 'Lab', 4),
                    ('Continuous cardiac monitoring', 'STAT', 'Monitor for arrhythmias', 'Monitoring', 1),
                    ('Cardiology consult', 'URGENT', 'If ECG abnormal or high-risk features', 'Consultation', 5)
                ]
            },
            'syncope': {
                'exams': [
                    ('ECG', 'STAT', 'Rule out arrhythmias, conduction disorders', 'Monitoring', 1),
                    ('Orthostatic vital signs', 'STAT', 'Check for orthostatic hypotension', 'Monitoring', 1),
                    ('CBC', 'URGENT', 'Rule out anemia', 'Lab', 3),
                    ('Basic metabolic panel', 'URGENT', 'Electrolytes, glucose', 'Lab', 3),
                    ('Troponin', 'URGENT', 'If cardiac syncope suspected', 'Lab', 3),
                    ('CT head', 'URGENT', 'If trauma or neurological signs', 'Imaging', 4)
                ]
            },
            'palpitations': {
                'exams': [
                    ('ECG', 'URGENT', 'Document rhythm', 'Monitoring', 1),
                    ('Telemetry monitoring', 'URGENT', 'Continuous rhythm monitoring', 'Monitoring', 1),
                    ('Thyroid function tests', 'ROUTINE', 'Rule out hyperthyroidism', 'Lab', 5),
                    ('Electrolytes', 'URGENT', 'Check for imbalances', 'Lab', 3)
                ]
            }
        }

        # NEUROLOGICAL RECOMMENDATIONS
        self.neuro_recommendations = {
            'headache_severe': {
                'exams': [
                    ('CT head without contrast', 'STAT', 'Rule out SAH, ICH, mass', 'Imaging', 1),
                    ('Lumbar puncture', 'STAT', 'If CT negative but SAH suspected', 'Procedure', 2),
                    ('Neurological exam', 'STAT', 'Complete neuro assessment', 'Consultation', 1),
                    ('Blood pressure', 'STAT', 'Hypertensive emergency?', 'Monitoring', 1)
                ]
            },
            'altered_mental_status': {
                'exams': [
                    ('Fingerstick glucose', 'STAT', 'Rule out hypoglycemia', 'Lab', 1),
                    ('Basic metabolic panel', 'STAT', 'Electrolytes, BUN, Cr', 'Lab', 1),
                    ('CBC', 'STAT', 'Infection, anemia', 'Lab', 1),
                    ('CT head', 'STAT', 'Rule out ICH, stroke, mass', 'Imaging', 1),
                    ('Toxicology screen', 'URGENT', 'If overdose suspected', 'Lab', 3),
                    ('Ammonia level', 'URGENT', 'If liver disease history', 'Lab', 3),
                    ('Urinalysis', 'URGENT', 'UTI common cause in elderly', 'Lab', 3)
                ]
            },
            'stroke_symptoms': {
                'exams': [
                    ('CT head without contrast', 'STAT', 'Hemorrhage vs ischemia', 'Imaging', 1),
                    ('Fingerstick glucose', 'STAT', 'Stroke mimic', 'Lab', 1),
                    ('ECG', 'STAT', 'Atrial fibrillation?', 'Monitoring', 1),
                    ('Neurology consult', 'STAT', 'Stroke team activation', 'Consultation', 1),
                    ('PT/INR', 'STAT', 'If on anticoagulation', 'Lab', 2),
                    ('CTA head/neck', 'STAT', 'If thrombectomy candidate', 'Imaging', 2)
                ]
            },
            'seizure': {
                'exams': [
                    ('Fingerstick glucose', 'STAT', 'Common cause', 'Lab', 1),
                    ('Basic metabolic panel', 'STAT', 'Electrolyte abnormalities', 'Lab', 1),
                    ('Anticonvulsant levels', 'URGENT', 'If known epilepsy', 'Lab', 3),
                    ('CT head', 'URGENT', 'First seizure or focal findings', 'Imaging', 2),
                    ('Toxicology screen', 'URGENT', 'Drug-related seizure', 'Lab', 3)
                ]
            }
        }

        # RESPIRATORY RECOMMENDATIONS
        self.respiratory_recommendations = {
            'dyspnea': {
                'exams': [
                    ('Pulse oximetry', 'STAT', 'Assess oxygenation', 'Monitoring', 1),
                    ('Chest X-ray', 'STAT', 'Pneumonia, CHF, pneumothorax', 'Imaging', 1),
                    ('ECG', 'STAT', 'Cardiac vs pulmonary', 'Monitoring', 1),
                    ('BNP or NT-proBNP', 'URGENT', 'CHF evaluation', 'Lab', 3),
                    ('D-dimer', 'URGENT', 'If PE suspected', 'Lab', 3),
                    ('ABG', 'URGENT', 'If severe or hypoxic', 'Lab', 2),
                    ('CT pulmonary angiogram', 'URGENT', 'If high PE suspicion', 'Imaging', 3)
                ]
            },
            'cough_hemoptysis': {
                'exams': [
                    ('Chest X-ray', 'STAT', 'Bleeding source, mass', 'Imaging', 1),
                    ('CBC', 'URGENT', 'Anemia, infection', 'Lab', 2),
                    ('Coagulation studies', 'URGENT', 'PT/PTT/INR', 'Lab', 2),
                    ('CT chest', 'URGENT', 'If massive or recurrent', 'Imaging', 3)
                ]
            }
        }

        # ABDOMINAL RECOMMENDATIONS
        self.abdominal_recommendations = {
            'abdominal_pain': {
                'exams': [
                    ('CBC', 'URGENT', 'Infection, inflammation', 'Lab', 2),
                    ('Basic metabolic panel', 'URGENT', 'Electrolytes', 'Lab', 2),
                    ('Lipase', 'URGENT', 'Pancreatitis', 'Lab', 2),
                    ('Liver function tests', 'URGENT', 'Hepatobiliary pathology', 'Lab', 2),
                    ('Urinalysis', 'URGENT', 'UTI, kidney stone', 'Lab', 2),
                    ('Pregnancy test', 'STAT', 'All females of childbearing age', 'Lab', 1),
                    ('Ultrasound abdomen/pelvis', 'URGENT', 'RUQ pain → gallstones', 'Imaging', 3),
                    ('CT abdomen/pelvis with contrast', 'URGENT', 'Acute abdomen, appendicitis', 'Imaging', 3)
                ]
            },
            'nausea_vomiting': {
                'exams': [
                    ('Basic metabolic panel', 'URGENT', 'Dehydration, electrolytes', 'Lab', 2),
                    ('Pregnancy test', 'STAT', 'Females of childbearing age', 'Lab', 1),
                    ('Urinalysis', 'URGENT', 'UTI, DKA', 'Lab', 3)
                ]
            }
        }

        # TRAUMA RECOMMENDATIONS
        self.trauma_recommendations = {
            'head_trauma': {
                'exams': [
                    ('CT head without contrast', 'STAT', 'ICH, skull fracture', 'Imaging', 1),
                    ('C-spine imaging', 'STAT', 'If mechanism concerning', 'Imaging', 1),
                    ('Neurosurgery consult', 'STAT', 'If ICH or depressed skull fracture', 'Consultation', 1),
                    ('Coagulation studies', 'STAT', 'If on anticoagulation', 'Lab', 2)
                ]
            },
            'trauma_major': {
                'exams': [
                    ('FAST exam', 'STAT', 'Intra-abdominal bleeding', 'Imaging', 1),
                    ('Trauma panel labs', 'STAT', 'CBC, CMP, coags, type & screen', 'Lab', 1),
                    ('CT chest/abdomen/pelvis', 'STAT', 'If stable', 'Imaging', 2),
                    ('Pelvic X-ray', 'STAT', 'If pelvic instability', 'Imaging', 1),
                    ('Trauma surgery consult', 'STAT', 'Activation', 'Consultation', 1)
                ]
            }
        }

        # Combiner tous les patterns
        self.all_patterns = {
            **self.cardio_recommendations,
            **self.neuro_recommendations,
            **self.respiratory_recommendations,
            **self.abdominal_recommendations,
            **self.trauma_recommendations
        }

    def generate_recommendations(self, text: str, esi_level: int, red_flags: List = None) -> List[Recommendation]:
        """
        Génère des recommandations d'examens basées sur le texte et l'ESI

        Args:
            text: Texte de présentation du patient
            esi_level: Niveau ESI prédit
            red_flags: Liste des red flags détectés (optionnel)

        Returns:
            Liste de recommandations triées par priorité
        """
        text_lower = text.lower()
        recommendations = []
        seen_exams = set()  # Pour éviter les duplicatas

        # Détecter les symptômes et générer recommandations
        for condition, info in self.all_patterns.items():
            if self._matches_condition(condition, text_lower):
                for exam, priority, rationale, category, order in info['exams']:
                    # Éviter les duplicatas
                    if exam in seen_exams:
                        continue

                    seen_exams.add(exam)

                    # Ajuster la priorité selon ESI
                    adjusted_priority = self._adjust_priority(priority, esi_level)

                    rec = Recommendation(
                        exam=exam,
                        priority=adjusted_priority,
                        rationale=rationale,
                        category=category,
                        order=order
                    )
                    recommendations.append(rec)

        # Ajouter des recommandations basées sur les red flags
        if red_flags:
            red_flag_recs = self._recommendations_from_red_flags(red_flags, seen_exams)
            recommendations.extend(red_flag_recs)

        # Ajouter recommandations de base selon ESI
        base_recs = self._base_recommendations_by_esi(esi_level, seen_exams)
        recommendations.extend(base_recs)

        # Trier par priorité et ordre
        priority_order = {'STAT': 1, 'URGENT': 2, 'ROUTINE': 3}
        recommendations.sort(key=lambda x: (priority_order[x.priority], x.order))

        return recommendations

    def _matches_condition(self, condition: str, text: str) -> bool:
        """Vérifie si le texte correspond à une condition"""
        # Mapping conditions → keywords
        condition_keywords = {
            'chest_pain': ['chest pain', 'substernal', 'retrosternal', 'cardiac pain', 'angina'],
            'syncope': ['syncope', 'passed out', 'lost consciousness', 'blacked out', 'fainted'],
            'palpitations': ['palpitation', 'heart racing', 'heart pounding', 'irregular heartbeat'],
            'headache_severe': ['severe headache', 'worst headache', 'thunderclap', 'intense headache'],
            'altered_mental_status': ['confusion', 'confused', 'altered mental', 'disoriented', 'AMS'],
            'stroke_symptoms': ['weakness', 'facial droop', 'slurred speech', 'paralysis', 'hemiparesis'],
            'seizure': ['seizure', 'convulsion', 'fitting', 'seizing'],
            'dyspnea': ['shortness of breath', 'dyspnea', 'difficulty breathing', 'sob', 'breathing difficulty'],
            'cough_hemoptysis': ['coughing blood', 'hemoptysis', 'bloody sputum'],
            'abdominal_pain': ['abdominal pain', 'stomach pain', 'belly pain', 'abd pain'],
            'nausea_vomiting': ['nausea', 'vomiting', 'throwing up', 'emesis'],
            'head_trauma': ['head trauma', 'head injury', 'hit head', 'struck head'],
            'trauma_major': ['trauma', 'injured', 'accident', 'crash', 'gsw', 'gunshot', 'stab']
        }

        keywords = condition_keywords.get(condition, [])
        return any(kw in text for kw in keywords)

    def _adjust_priority(self, base_priority: str, esi_level: int) -> str:
        """Ajuste la priorité selon le niveau ESI"""
        if esi_level in [1, 2]:
            # Cas critiques: tout devient STAT ou URGENT
            if base_priority == 'ROUTINE':
                return 'URGENT'
            return base_priority
        elif esi_level == 3:
            return base_priority
        else:  # ESI 4-5
            # Cas moins urgents: STAT→URGENT, URGENT→ROUTINE
            if base_priority == 'STAT':
                return 'URGENT'
            elif base_priority == 'URGENT':
                return 'ROUTINE'
            return base_priority

    def _recommendations_from_red_flags(self, red_flags, seen_exams: set) -> List[Recommendation]:
        """Génère des recommandations spécifiques basées sur les red flags"""
        recommendations = []

        for flag in red_flags:
            # Red flags cardiaques
            if 'chest pain' in flag.description.lower() and 'ECG' not in seen_exams:
                recommendations.append(Recommendation(
                    exam='ECG 12-leads',
                    priority='STAT',
                    rationale=f'Red flag detected: {flag.description}',
                    category='Monitoring',
                    order=1
                ))
                seen_exams.add('ECG 12-leads')

            # Red flags neurologiques
            if any(kw in flag.description.lower() for kw in ['altered mental', 'weakness', 'headache']):
                if 'CT head' not in seen_exams:
                    recommendations.append(Recommendation(
                        exam='CT head without contrast',
                        priority='STAT',
                        rationale=f'Red flag detected: {flag.description}',
                        category='Imaging',
                        order=1
                    ))
                    seen_exams.add('CT head')

        return recommendations

    def _base_recommendations_by_esi(self, esi_level: int, seen_exams: set) -> List[Recommendation]:
        """Recommandations de base selon ESI"""
        recommendations = []

        # ESI 1-2: Labs de base + monitoring
        if esi_level in [1, 2]:
            base_labs = [
                ('CBC', 'STAT', 'Baseline hematology', 'Lab', 2),
                ('Basic metabolic panel', 'STAT', 'Electrolytes, renal function', 'Lab', 2),
                ('Troponin', 'STAT', 'Cardiac marker', 'Lab', 2),
                ('Lactate', 'STAT', 'Tissue perfusion marker', 'Lab', 2),
                ('Type and screen', 'STAT', 'Blood bank preparation', 'Lab', 3),
                ('Continuous monitoring', 'STAT', 'Vitals, telemetry, SpO2', 'Monitoring', 1)
            ]

            for exam, priority, rationale, category, order in base_labs:
                if exam not in seen_exams:
                    recommendations.append(Recommendation(
                        exam=exam,
                        priority=priority,
                        rationale=rationale + f' (ESI-{esi_level})',
                        category=category,
                        order=order
                    ))
                    seen_exams.add(exam)

        return recommendations

    def format_recommendations_list(self, recommendations: List[Recommendation]) -> str:
        """Formate les recommandations en liste textuelle"""
        if not recommendations:
            return "Aucune recommandation spécifique à ce stade."

        output = []
        for i, rec in enumerate(recommendations, 1):
            priority_icon = "🔴" if rec.priority == "STAT" else "🟠" if rec.priority == "URGENT" else "🟢"
            output.append(f"{i}. {priority_icon} **{rec.exam}** ({rec.priority})")
            output.append(f"   - {rec.rationale}")
            output.append("")

        return "\n".join(output)


# Fonction de test
def test_recommendations_engine():
    """Teste le moteur de recommandations"""
    engine = RecommendationsEngine()

    # Test 1: Chest pain ESI-2
    case1 = """
    55-year-old male with crushing chest pain radiating to left arm.
    Shortness of breath and diaphoresis. History of hypertension.
    """

    print("="*80)
    print("TEST: RECOMMENDATIONS ENGINE")
    print("="*80)
    print("\nCASE 1: Chest pain (ESI-2)")
    print("-"*80)
    print(case1.strip())

    recs = engine.generate_recommendations(case1, esi_level=2)
    print(f"\n📋 RECOMMENDATIONS ({len(recs)}):")
    print(engine.format_recommendations_list(recs))


if __name__ == "__main__":
    test_recommendations_engine()
