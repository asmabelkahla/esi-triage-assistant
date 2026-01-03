# -*- coding: utf-8 -*-
"""
ESI Post-Processor - Ajustement des prédictions ESI
Corrige les cas de sur-triage basés sur le contexte clinique
"""

import re
from typing import Dict, Tuple

class ESIPostProcessor:
    """
    Ajuste les prédictions ESI basées sur des règles cliniques contextuelles
    pour éviter le sur-triage dans des cas non-urgents
    """

    def __init__(self):
        """Initialise le post-processeur avec les règles de détection"""

        # Patterns pour détecter les consultations administratives/non-urgentes
        self.administrative_patterns = [
            # Demandes de certificats
            r"(?i)(?:requesting|asking for|needs?|wants?).*?(?:medical )?certificate",
            r"(?i)certificate.*?(?:for|regarding|after)",
            r"(?i)work (?:note|certificate|sick note)",
            r"(?i)doctor'?s? note",

            # Suivi/follow-up non urgent
            r"(?i)follow-?up.*?(?:appointment|visit)",
            r"(?i)routine.*?(?:check-?up|visit|appointment)",
            r"(?i)scheduled.*?(?:appointment|visit)",

            # Consultations de conseil
            r"(?i)advice.*?regarding",
            r"(?i)consultation.*?about",
            r"(?i)questions? about",
        ]

        # Patterns temporels indiquant que l'événement n'est PAS aigu
        self.non_acute_temporal_patterns = [
            r"(?i)(\d+)\s*(?:days?|weeks?|months?)\s*ago",
            r"(?i)(?:yesterday|last week|last month)",
            r"(?i)since\s*(?:yesterday|last\s*\w+)",
            r"(?i)for\s*(?:the\s*)?(?:past|last)\s*\d+\s*(?:days?|weeks?)",
        ]

        # Indicators de cas bénins stables
        self.stable_mild_indicators = [
            r"(?i)vital signs?:?\s*(?:all\s*)?normal",
            r"(?i)no\s*(?:neurological|cardiac|respiratory)\s*(?:deficit|symptoms?)",
            r"(?i)mild\s*(?:discomfort|pain|tenderness)",
            r"(?i)conservative\s*(?:management|treatment)",
            r"(?i)symptomatic\s*treatment\s*only",
            r"(?i)NSAIDS?\s*(?:and|,)",
            r"(?i)able\s*to\s*(?:walk|bear\s*weight|eat|drink)",
            r"(?i)no\s*(?:imaging|labs?|tests?)\s*(?:needed|required|indicated)",
        ]

        # Red flags qui annulent la dégradation vers ESI-5
        # IMPORTANT: Ces patterns doivent être spécifiques pour éviter les faux positifs
        self.override_patterns = [
            # Douleur sévère ACTUELLE (pas historique/diagnositc)
            r"(?i)(?:severe|intense)\s+(?:pain|discomfort)",
            r"(?i)(?:acute|sudden)\s+(?:severe|intense)\s+pain",
            r"(?i)pain.*?(?:severe|intense|unbearable|excruciating)",

            # Perte de conscience
            r"(?i)(?:unconscious|unresponsive|not responding)",
            r"(?i)loss of consciousness|LOC",
            r"(?i)syncope|faint(?:ed|ing)",

            # Hémorragie active
            r"(?i)(?:active|ongoing|profuse|severe)\s*(?:hemorrhage|bleeding)",
            r"(?i)blood loss",
            r"(?i)hemorrhag(?:e|ic)\s*shock",

            # Détresse respiratoire
            r"(?i)respiratory\s*distress",
            r"(?i)(?:severe|acute)\s*(?:dyspnea|SOB|shortness of breath)",
            r"(?i)unable to breathe",
            r"(?i)respiratory\s*failure",

            # Douleur thoracique (syndrome coronarien aigu)
            r"(?i)chest pain.*?(?:radiating|crushing|pressure)",
            r"(?i)(?:suspected|probable|confirmed)\s*(?:MI|STEMI|ACS|myocardial infarction)",
            r"(?i)cardiac\s*arrest",

            # AVC
            r"(?i)(?:acute|suspected)\s*(?:stroke|CVA)",
            r"(?i)(?:hemiparesis|hemiplegia|aphasia|facial droop)",
            r"(?i)(?:sudden|acute)\s*(?:weakness|paralysis|numbness)",

            # Shock/Instabilité
            r"(?i)(?:hemorrhagic|septic|cardiogenic|anaphylactic)\s*shock",
            r"(?i)(?:severe|profound)\s*hypotension",
            r"(?i)hemodynamically\s*unstable",
            r"(?i)BP\s*(?:<|less than|under)\s*(?:80|90)/",

            # Signes vitaux anormaux CRITIQUES
            r"(?i)vital signs?\s*(?:critically|severely)\s*abnormal",
            r"(?i)SpO2\s*(?:<|less than)\s*(?:85|90)%",
        ]

    def _count_pattern_matches(self, text: str, patterns: list) -> int:
        """Compte le nombre de patterns qui matchent dans le texte"""
        count = 0
        for pattern in patterns:
            if re.search(pattern, text):
                count += 1
        return count

    def _has_pattern(self, text: str, patterns: list) -> bool:
        """Vérifie si au moins un pattern match"""
        return any(re.search(pattern, text) for pattern in patterns)

    def adjust_prediction(
        self,
        text: str,
        predicted_esi: int,
        confidence: float,
        probabilities: Dict[int, float]
    ) -> Tuple[int, float, Dict[int, float], str]:
        """
        Ajuste la prédiction ESI si nécessaire

        Args:
            text: Texte clinique
            predicted_esi: ESI prédit par le modèle (1-5)
            confidence: Confiance du modèle (0-100)
            probabilities: Dict {esi: probability}

        Returns:
            (adjusted_esi, adjusted_confidence, adjusted_probabilities, reason)
        """

        # Pas d'ajustement pour ESI-1 (toujours critique)
        if predicted_esi == 1:
            return predicted_esi, confidence, probabilities, ""

        # Pas d'ajustement si déjà ESI-5
        if predicted_esi == 5:
            return predicted_esi, confidence, probabilities, ""

        # Vérifier les red flags qui empêchent la dégradation
        if self._has_pattern(text, self.override_patterns):
            return predicted_esi, confidence, probabilities, ""

        # Compter les indicateurs de non-urgence
        admin_score = self._count_pattern_matches(text, self.administrative_patterns)
        non_acute_score = self._count_pattern_matches(text, self.non_acute_temporal_patterns)
        stable_score = self._count_pattern_matches(text, self.stable_mild_indicators)

        total_non_urgent_score = admin_score + non_acute_score + stable_score

        # Règle 1: Consultation administrative claire
        if admin_score >= 1 and stable_score >= 2:
            # Si demande de certificat + patient stable → ESI-5
            adjusted_esi = 5
            adjusted_confidence = max(probabilities.get(5, 0), 75.0)
            adjusted_probs = probabilities.copy()
            adjusted_probs[5] = adjusted_confidence

            # Réduire les autres probabilités proportionnellement
            remaining = 100 - adjusted_confidence
            for esi in range(1, 5):
                adjusted_probs[esi] = probabilities[esi] * (remaining / sum(probabilities[i] for i in range(1, 5)))

            reason = "Ajusté à ESI-5: demande administrative (certificat médical) avec patient stable"
            return adjusted_esi, adjusted_confidence, adjusted_probs, reason

        # Règle 2: Événement ancien + stable + bénin
        if non_acute_score >= 1 and stable_score >= 3 and predicted_esi >= 3:
            # Événement passé (jours/semaines) + très stable → ESI-5
            adjusted_esi = 5
            adjusted_confidence = max(probabilities.get(5, 0), 70.0)
            adjusted_probs = probabilities.copy()
            adjusted_probs[5] = adjusted_confidence

            remaining = 100 - adjusted_confidence
            for esi in range(1, 5):
                adjusted_probs[esi] = probabilities[esi] * (remaining / sum(probabilities[i] for i in range(1, 5)))

            reason = "Ajusté à ESI-5: événement non-aigu (ancien) avec présentation bénigne stable"
            return adjusted_esi, adjusted_confidence, adjusted_probs, reason

        # Règle 3: Score global très élevé de non-urgence
        if total_non_urgent_score >= 5 and predicted_esi >= 3:
            # Nombreux indicateurs de non-urgence → réduire d'au moins 1 niveau
            adjusted_esi = min(predicted_esi + 1, 5)

            # Si déjà prédit ESI-4 ou ESI-3 avec très haut score → ESI-5
            if predicted_esi >= 3 and total_non_urgent_score >= 6:
                adjusted_esi = 5

            adjusted_confidence = max(probabilities.get(adjusted_esi, 0), 65.0)
            adjusted_probs = probabilities.copy()
            adjusted_probs[adjusted_esi] = adjusted_confidence

            remaining = 100 - adjusted_confidence
            for esi in range(1, 6):
                if esi != adjusted_esi:
                    adjusted_probs[esi] = probabilities[esi] * (remaining / sum(probabilities[i] for i in range(1, 6) if i != adjusted_esi))

            reason = f"Ajusté à ESI-{adjusted_esi}: nombreux indicateurs de non-urgence (score: {total_non_urgent_score})"
            return adjusted_esi, adjusted_confidence, adjusted_probs, reason

        # Règle 4: ESI-3 avec patient très stable → ESI-4
        if predicted_esi == 3 and stable_score >= 4 and non_acute_score >= 1:
            adjusted_esi = 4
            adjusted_confidence = max(probabilities.get(4, 0), 60.0)
            adjusted_probs = probabilities.copy()
            adjusted_probs[4] = adjusted_confidence

            remaining = 100 - adjusted_confidence
            for esi in [1, 2, 3, 5]:
                adjusted_probs[esi] = probabilities[esi] * (remaining / sum(probabilities[i] for i in [1, 2, 3, 5]))

            reason = "Ajusté à ESI-4: ESI-3 prédit mais patient très stable sans urgence"
            return adjusted_esi, adjusted_confidence, adjusted_probs, reason

        # Règle 5: ESI-3 avec indicateurs mineurs → ESI-4
        if predicted_esi == 3:
            minor_keywords = [
                r"minor", r"small", r"superficial", r"mild",
                r"pain [1-4]/10", r"tolerable",
                r"stable.*normal", r"no distress",
                r"simple", r"routine", r"suture removal"
            ]

            minor_score = sum(1 for p in minor_keywords if re.search(p, text, re.IGNORECASE))

            # Si beaucoup d'indicateurs mineurs ET prob ESI-4 > 10%
            if minor_score >= 3 and probabilities.get(4, 0) > 10:
                adjusted_esi = 4
                adjusted_confidence = probabilities[3] + probabilities[4]
                adjusted_probs = probabilities.copy()
                adjusted_probs[4] = adjusted_confidence
                adjusted_probs[3] = 5.0

                # Redistribuer le reste
                remaining = 100 - adjusted_confidence - 5.0
                for esi in [1, 2, 5]:
                    if sum(probabilities[i] for i in [1, 2, 5]) > 0:
                        adjusted_probs[esi] = probabilities[esi] * (remaining / sum(probabilities[i] for i in [1, 2, 5]))
                    else:
                        adjusted_probs[esi] = remaining / 3

                reason = f"Ajusté à ESI-4: cas mineur stable (score indicateurs mineurs: {minor_score})"
                return adjusted_esi, adjusted_confidence, adjusted_probs, reason

        # Pas d'ajustement nécessaire
        return predicted_esi, confidence, probabilities, ""

    def explain_scoring(self, text: str) -> Dict[str, int]:
        """
        Explique les scores détectés (pour debugging)

        Returns:
            Dict avec les différents scores calculés
        """
        return {
            "administrative_score": self._count_pattern_matches(text, self.administrative_patterns),
            "non_acute_temporal_score": self._count_pattern_matches(text, self.non_acute_temporal_patterns),
            "stable_mild_score": self._count_pattern_matches(text, self.stable_mild_indicators),
            "has_red_flag_override": self._has_pattern(text, self.override_patterns),
        }


if __name__ == "__main__":
    # Test du post-processeur
    processor = ESIPostProcessor()

    # Cas test: Certificat médical (devrait être ESI-5)
    test_text = """38-year-old woman requesting medical certificate after minor car accident 2 days ago.
    Mild neck stiffness. All vital signs normal. No neurological deficit.
    Full cervical range of motion. Minor post-traumatic cervicalgia.
    Conservative management with NSAIDs and heat therapy."""

    # Simulation: modèle prédit ESI-3 (erreur)
    predicted = 3
    confidence = 65.0
    probs = {1: 5.0, 2: 15.0, 3: 65.0, 4: 10.0, 5: 5.0}

    print("AVANT POST-PROCESSING:")
    print(f"ESI prédit: {predicted} (confiance: {confidence}%)")
    print(f"Probabilités: {probs}")

    # Appliquer post-processing
    adj_esi, adj_conf, adj_probs, reason = processor.adjust_prediction(
        test_text, predicted, confidence, probs
    )

    print("\nAPRÈS POST-PROCESSING:")
    print(f"ESI ajusté: {adj_esi} (confiance: {adj_conf:.1f}%)")
    print(f"Probabilités ajustées: {adj_probs}")
    print(f"Raison: {reason}")

    print("\nScores détaillés:")
    scores = processor.explain_scoring(test_text)
    for key, value in scores.items():
        print(f"  {key}: {value}")