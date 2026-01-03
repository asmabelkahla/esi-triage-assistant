"""
Module de gestion de l'historique des patients
Sauvegarde et récupération des cas de triage
"""
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class PatientCase:
    """Représente un cas de patient"""
    id: str
    timestamp: str
    description: str
    esi_level: int
    confidence: float
    red_flags_count: int
    source: str  # 'text' ou 'audio'

    # Optionnel
    chief_complaint: Optional[str] = None
    symptoms: Optional[List[str]] = None
    recommended_exams: Optional[List[str]] = None

    def to_dict(self):
        """Convertit en dictionnaire"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        """Crée depuis un dictionnaire"""
        return cls(**data)


class PatientHistory:
    """Gestionnaire d'historique des patients"""

    def __init__(self, history_file: str = "patient_history.json"):
        """
        Initialise le gestionnaire d'historique

        Args:
            history_file: Chemin vers le fichier JSON
        """
        self.history_file = history_file
        self.cases: List[PatientCase] = []
        self.load_history()

    def load_history(self):
        """Charge l'historique depuis le fichier"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.cases = [PatientCase.from_dict(case) for case in data]
            except Exception as e:
                print(f"Erreur lors du chargement de l'historique: {e}")
                self.cases = []
        else:
            self.cases = []

    def save_history(self):
        """Sauvegarde l'historique dans le fichier"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump([case.to_dict() for case in self.cases], f,
                         ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Erreur lors de la sauvegarde: {e}")
            return False

    def add_case(self, description: str, esi_level: int, confidence: float,
                 red_flags_count: int = 0, source: str = 'text',
                 chief_complaint: str = None, symptoms: List[str] = None,
                 recommended_exams: List[str] = None) -> PatientCase:
        """
        Ajoute un nouveau cas à l'historique

        Returns:
            Le cas créé
        """
        now = datetime.now()
        case_id = now.strftime("%Y%m%d_%H%M%S")

        case = PatientCase(
            id=case_id,
            timestamp=now.strftime("%Y-%m-%d %H:%M:%S"),
            description=description,
            esi_level=esi_level,
            confidence=confidence,
            red_flags_count=red_flags_count,
            source=source,
            chief_complaint=chief_complaint,
            symptoms=symptoms,
            recommended_exams=recommended_exams
        )

        self.cases.append(case)
        self.save_history()
        return case

    def get_all_cases(self) -> List[PatientCase]:
        """Retourne tous les cas"""
        return self.cases

    def get_case_by_id(self, case_id: str) -> Optional[PatientCase]:
        """Récupère un cas par son ID"""
        for case in self.cases:
            if case.id == case_id:
                return case
        return None

    def delete_case(self, case_id: str) -> bool:
        """Supprime un cas"""
        for i, case in enumerate(self.cases):
            if case.id == case_id:
                del self.cases[i]
                self.save_history()
                return True
        return False

    def search_cases(self, query: str = None, esi_level: int = None,
                    date_from: str = None, date_to: str = None,
                    source: str = None) -> List[PatientCase]:
        """
        Recherche des cas avec filtres

        Args:
            query: Texte à rechercher dans la description
            esi_level: Niveau ESI
            date_from: Date de début (format: YYYY-MM-DD)
            date_to: Date de fin (format: YYYY-MM-DD)
            source: Type de source ('text' ou 'audio')

        Returns:
            Liste des cas correspondants
        """
        results = self.cases.copy()

        # Filtre par texte
        if query:
            query = query.lower()
            results = [c for c in results if query in c.description.lower()]

        # Filtre par ESI
        if esi_level is not None:
            results = [c for c in results if c.esi_level == esi_level]

        # Filtre par source
        if source:
            results = [c for c in results if c.source == source]

        # Filtre par date
        if date_from:
            results = [c for c in results if c.timestamp >= date_from]
        if date_to:
            results = [c for c in results if c.timestamp <= date_to + " 23:59:59"]

        return results

    def get_statistics(self) -> Dict:
        """
        Calcule des statistiques sur l'historique

        Returns:
            Dictionnaire avec les stats
        """
        if not self.cases:
            return {
                'total_cases': 0,
                'esi_distribution': {},
                'average_confidence': 0,
                'critical_cases': 0,
                'red_flags_total': 0,
                'source_distribution': {}
            }

        # Distribution ESI
        esi_dist = {}
        for i in range(1, 6):
            esi_dist[f'ESI-{i}'] = len([c for c in self.cases if c.esi_level == i])

        # Source distribution
        source_dist = {
            'Texte': len([c for c in self.cases if c.source == 'text']),
            'Audio': len([c for c in self.cases if c.source == 'audio'])
        }

        # Cas critiques (ESI 1-2)
        critical = len([c for c in self.cases if c.esi_level <= 2])

        # Confiance moyenne
        avg_conf = sum(c.confidence for c in self.cases) / len(self.cases)

        # Red flags total
        red_flags_total = sum(c.red_flags_count for c in self.cases)

        return {
            'total_cases': len(self.cases),
            'esi_distribution': esi_dist,
            'average_confidence': avg_conf,
            'critical_cases': critical,
            'red_flags_total': red_flags_total,
            'source_distribution': source_dist
        }

    def export_to_csv(self, filepath: str = "triage_export.csv") -> bool:
        """
        Exporte l'historique en CSV

        Args:
            filepath: Chemin du fichier CSV

        Returns:
            True si succès
        """
        try:
            data = []
            for case in self.cases:
                data.append({
                    'ID': case.id,
                    'Date': case.timestamp,
                    'Description': case.description,
                    'ESI': case.esi_level,
                    'Confiance (%)': f"{case.confidence:.1f}",
                    'Red Flags': case.red_flags_count,
                    'Source': case.source,
                    'Motif': case.chief_complaint or '',
                    'Symptômes': ', '.join(case.symptoms) if case.symptoms else ''
                })

            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            return True
        except Exception as e:
            print(f"Erreur export CSV: {e}")
            return False

    def get_recent_cases(self, n: int = 10) -> List[PatientCase]:
        """Retourne les N cas les plus récents"""
        return sorted(self.cases, key=lambda x: x.timestamp, reverse=True)[:n]

    def clear_all(self) -> bool:
        """Efface tout l'historique"""
        self.cases = []
        return self.save_history()
