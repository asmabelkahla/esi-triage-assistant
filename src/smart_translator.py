# -*- coding: utf-8 -*-
"""
🌐 Smart Translator - Traduction Intelligente avec Cache
Traduit automatiquement tout le contenu de l'application
"""

import streamlit as st
from typing import Union, Dict, List, Any
import hashlib
import json

class SmartTranslator:
    """
    Traducteur intelligent qui traduit automatiquement tout contenu
    avec mise en cache pour optimiser les performances.
    """

    def __init__(self):
        """Initialise le traducteur avec cache en session"""
        if 'translation_cache' not in st.session_state:
            st.session_state.translation_cache = {}

        self.cache = st.session_state.translation_cache

        # Langues supportées
        self.supported_languages = {
            'fr': 'Français',
            'en': 'English',
            'ar': 'العربية',
            'es': 'Español',
            'de': 'Deutsch'
        }

    def _get_cache_key(self, text: str, target_lang: str) -> str:
        """Génère une clé de cache unique"""
        content = f"{text}|{target_lang}"
        return hashlib.md5(content.encode()).hexdigest()

    def _translate_with_deepl_api(self, text: str, target_lang: str) -> str:
        """Traduction via DeepL (gratuit jusqu'à 500k caractères/mois)"""
        try:
            import requests

            # Mapping des codes langues
            lang_map = {
                'fr': 'FR',
                'en': 'EN',
                'ar': 'AR',  # Note: DeepL ne supporte pas l'arabe
                'es': 'ES',
                'de': 'DE'
            }

            target = lang_map.get(target_lang, 'EN')

            # Note: Nécessite une clé API DeepL (gratuite)
            # À configurer dans les variables d'environnement
            api_key = st.secrets.get("DEEPL_API_KEY", None)

            if not api_key:
                return self._translate_with_google(text, target_lang)

            url = "https://api-free.deepl.com/v2/translate"
            params = {
                'auth_key': api_key,
                'text': text,
                'target_lang': target
            }

            response = requests.post(url, data=params, timeout=5)
            if response.status_code == 200:
                return response.json()['translations'][0]['text']
            else:
                return self._translate_with_google(text, target_lang)

        except Exception:
            return self._translate_with_google(text, target_lang)

    def _translate_with_google(self, text: str, target_lang: str) -> str:
        """Traduction via Google Translate (deep-translator - gratuit et stable)"""
        try:
            from deep_translator import GoogleTranslator

            # Mapping des codes langues
            lang_map = {
                'fr': 'fr',
                'en': 'en',
                'ar': 'ar',
                'es': 'es',
                'de': 'de'
            }

            target = lang_map.get(target_lang, 'en')

            translator = GoogleTranslator(source='auto', target=target)
            result = translator.translate(text)
            return result

        except Exception as e:
            # Si erreur, retourner le texte original
            return text

    def _translate_with_argos(self, text: str, target_lang: str) -> str:
        """Traduction locale via ArgosTranslate (100% gratuit, hors ligne)"""
        try:
            import argostranslate.package
            import argostranslate.translate

            # Mapping des codes langues
            lang_map = {
                'fr': 'fr',
                'en': 'en',
                'ar': 'ar',
                'es': 'es',
                'de': 'de'
            }

            # Détecter la langue source (on assume français par défaut)
            from_code = 'fr'
            to_code = lang_map.get(target_lang, 'en')

            # Traduire
            translated_text = argostranslate.translate.translate(text, from_code, to_code)
            return translated_text

        except Exception:
            return self._translate_with_google(text, target_lang)

    def translate(self, content: Union[str, List, Dict], target_lang: str, source_lang: str = 'fr') -> Union[str, List, Dict]:
        """
        Traduit intelligemment n'importe quel contenu

        Args:
            content: Texte, liste ou dictionnaire à traduire
            target_lang: Langue cible (fr, en, ar, etc.)
            source_lang: Langue source (défaut: fr)

        Returns:
            Contenu traduit dans le même format
        """
        # Si même langue, retourner tel quel
        if target_lang == source_lang:
            return content

        # Si langue non supportée, retourner original
        if target_lang not in self.supported_languages:
            return content

        # Traduction selon le type
        if isinstance(content, str):
            return self._translate_text(content, target_lang)

        elif isinstance(content, list):
            return [self.translate(item, target_lang, source_lang) for item in content]

        elif isinstance(content, dict):
            return {
                key: self.translate(value, target_lang, source_lang)
                for key, value in content.items()
            }

        else:
            return content

    def _translate_text(self, text: str, target_lang: str) -> str:
        """Traduit un texte avec cache"""
        if not text or text.strip() == '':
            return text

        # Vérifier le cache
        cache_key = self._get_cache_key(text, target_lang)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Traduire
        try:
            # Essayer Google Translate d'abord (plus simple)
            translated = self._translate_with_google(text, target_lang)

            # Mettre en cache
            self.cache[cache_key] = translated

            return translated

        except Exception as e:
            # En cas d'erreur, retourner le texte original
            return text

    def t(self, text: str, target_lang: str = None) -> str:
        """
        Raccourci pour traduire un texte

        Usage:
            translator = SmartTranslator()
            print(translator.t("Bonjour", "en"))  # "Hello"
        """
        if target_lang is None:
            target_lang = st.session_state.get('language', 'fr')

        return self.translate(text, target_lang)


# Fonction helper pour utilisation facile
def auto_translate(content: Any, target_lang: str = None) -> Any:
    """
    Traduit automatiquement n'importe quel contenu

    Usage dans Streamlit:
        st.markdown(auto_translate("Résumé Clinique"))
        st.write(auto_translate(["Symptôme 1", "Symptôme 2"]))
    """
    if target_lang is None:
        target_lang = st.session_state.get('language', 'fr')

    translator = SmartTranslator()
    return translator.translate(content, target_lang)


# Décorateur pour traduire automatiquement les retours de fonction
def translate_output(func):
    """
    Décorateur qui traduit automatiquement le résultat d'une fonction

    Usage:
        @translate_output
        def get_clinical_summary():
            return "Patient présente des symptômes..."
    """
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        target_lang = st.session_state.get('language', 'fr')
        translator = SmartTranslator()
        return translator.translate(result, target_lang)

    return wrapper


if __name__ == "__main__":
    # Tests
    translator = SmartTranslator()

    print("Test 1: Texte simple")
    print(translator.translate("Douleur thoracique intense", "en"))

    print("\nTest 2: Liste")
    symptoms = ["Fièvre", "Toux", "Fatigue"]
    print(translator.translate(symptoms, "en"))

    print("\nTest 3: Dictionnaire")
    data = {
        "motif": "Douleur abdominale",
        "symptomes": ["Nausées", "Vomissements"]
    }
    print(translator.translate(data, "en"))
