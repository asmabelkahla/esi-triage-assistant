# -*- coding: utf-8 -*-
"""
Context Enhancer - Améliore le texte avec des marqueurs contextuels
pour aider le modèle NLP à mieux comprendre
"""

import re
from typing import Tuple

class ContextEnhancer:
    """
    Ajoute des marqueurs contextuels explicites au texte
    pour améliorer la compréhension du modèle NLP
    """

    def __init__(self):
        """Initialise l'enhancer avec les patterns de détection"""

        # Patterns de contexte administratif
        self.admin_patterns = [
            r"(?i)request(?:ing|ed).*?certificate",
            r"(?i)work\s*note",
            r"(?i)sick\s*note",
            r"(?i)medical\s*certificate",
        ]

        # Patterns temporels anciens
        self.old_event_patterns = [
            (r"(?i)(\d+)\s*days?\s*ago", "PAST_EVENT"),
            (r"(?i)(\d+)\s*weeks?\s*ago", "PAST_EVENT"),
            (r"(?i)yesterday", "PAST_EVENT"),
            (r"(?i)last\s*week", "PAST_EVENT"),
        ]

        # Patterns de stabilité
        self.stability_patterns = [
            (r"(?i)vital\s*signs?.*?normal", "STABLE_VITALS"),
            (r"(?i)no.*?deficit", "NO_DEFICIT"),
            (r"(?i)mild.*?(?:pain|discomfort)", "MILD_SYMPTOMS"),
            (r"(?i)conservative\s*management", "NON_URGENT_CARE"),
        ]

    def enhance_text(self, text: str) -> Tuple[str, dict]:
        """
        Ajoute des marqueurs contextuels au texte

        Args:
            text: Texte clinique original

        Returns:
            (enhanced_text, context_info)
        """
        enhanced = text
        context_info = {
            "is_administrative": False,
            "is_old_event": False,
            "is_stable": False,
            "markers_added": []
        }

        # Détecter contexte administratif
        if any(re.search(pattern, text) for pattern in self.admin_patterns):
            enhanced = "[ADMINISTRATIVE_REQUEST] " + enhanced
            context_info["is_administrative"] = True
            context_info["markers_added"].append("ADMINISTRATIVE_REQUEST")

        # Détecter événement ancien
        for pattern, marker in self.old_event_patterns:
            if re.search(pattern, text):
                if "[PAST_EVENT]" not in enhanced:
                    enhanced = "[PAST_EVENT] " + enhanced
                context_info["is_old_event"] = True
                if "PAST_EVENT" not in context_info["markers_added"]:
                    context_info["markers_added"].append("PAST_EVENT")
                break

        # Détecter stabilité (compter les indicateurs)
        stability_count = 0
        for pattern, marker in self.stability_patterns:
            if re.search(pattern, text):
                stability_count += 1

        if stability_count >= 2:
            enhanced = "[STABLE_PATIENT] " + enhanced
            context_info["is_stable"] = True
            context_info["markers_added"].append("STABLE_PATIENT")

        # Combinaison administrative + ancien + stable = NON-URGENT
        if (context_info["is_administrative"] and
            context_info["is_old_event"] and
            context_info["is_stable"]):
            enhanced = "[NON_URGENT_CASE] " + enhanced
            context_info["markers_added"].insert(0, "NON_URGENT_CASE")

        return enhanced, context_info

    def should_downgrade_to_esi5(self, context_info: dict) -> bool:
        """
        Détermine si le cas devrait être ESI-5 basé sur le contexte

        Args:
            context_info: Info de contexte de enhance_text()

        Returns:
            True si devrait être ESI-5
        """
        # Critères stricts pour ESI-5
        if context_info["is_administrative"] and context_info["is_stable"]:
            return True

        if (context_info["is_old_event"] and
            context_info["is_stable"] and
            len(context_info["markers_added"]) >= 3):
            return True

        return False


if __name__ == "__main__":
    # Test
    enhancer = ContextEnhancer()

    # Cas 1: Certificat médical
    text1 = """38-year-old woman requesting medical certificate after minor car accident 2 days ago.
    Mild neck stiffness. All vital signs normal. No neurological deficit.
    Conservative management with NSAIDs."""

    enhanced1, info1 = enhancer.enhance_text(text1)

    print("="*80)
    print("CAS 1: Certificat Médical")
    print("="*80)
    print("\nTEXTE ORIGINAL:")
    print(text1[:150] + "...")
    print("\nTEXTE AMÉLIORÉ:")
    print(enhanced1[:200] + "...")
    print("\nCONTEXTE DÉTECTÉ:")
    for key, value in info1.items():
        print(f"  {key}: {value}")
    print(f"\nDevrait être ESI-5: {enhancer.should_downgrade_to_esi5(info1)}")

    # Cas 2: Urgence réelle
    text2 = """55-year-old male with severe crushing chest pain radiating to left arm for 30 minutes.
    Diaphoretic, pale. BP 160/95, HR 105."""

    enhanced2, info2 = enhancer.enhance_text(text2)

    print("\n" + "="*80)
    print("CAS 2: Urgence Réelle (Douleur Thoracique)")
    print("="*80)
    print("\nTEXTE ORIGINAL:")
    print(text2)
    print("\nTEXTE AMÉLIORÉ:")
    print(enhanced2)
    print("\nCONTEXTE DÉTECTÉ:")
    for key, value in info2.items():
        print(f"  {key}: {value}")
    print(f"\nDevrait être ESI-5: {enhancer.should_downgrade_to_esi5(info2)}")