"""
Module de traitement audio avec Whisper pour la transcription en temps réel
"""
import os
import tempfile
import warnings
warnings.filterwarnings('ignore')

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: faster-whisper not installed. Audio features will be disabled.")

import torch
import numpy as np


class AudioProcessor:
    """Processeur audio pour la transcription avec Whisper"""

    def __init__(self, model_size="base", device=None, compute_type="int8"):
        """
        Initialise le processeur audio

        Args:
            model_size: Taille du modèle Whisper (tiny, base, small, medium, large)
            device: 'cpu' ou 'cuda'. Si None, détection automatique
            compute_type: Type de calcul ('int8', 'float16', 'float32')
        """
        if not WHISPER_AVAILABLE:
            raise ImportError(
                "faster-whisper n'est pas installé. "
                "Installez-le avec: pip install faster-whisper"
            )

        # Détection automatique du device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_size = model_size

        # Ajuster compute_type selon le device
        if device == "cpu":
            compute_type = "int8"  # Plus rapide sur CPU
        elif compute_type == "int8" and device == "cuda":
            compute_type = "float16"  # int8 pas toujours supporté sur GPU

        print(f"Chargement du modèle Whisper '{model_size}' sur {device} avec {compute_type}...")

        try:
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type
            )
            print("[OK] Modele Whisper charge avec succes")
        except Exception as e:
            print(f"Erreur lors du chargement du modele: {e}")
            raise

    def transcribe_audio(self, audio_file_path, language=None, auto_translate_to='fr'):
        """
        Transcrit un fichier audio en texte avec traduction automatique

        Args:
            audio_file_path: Chemin vers le fichier audio
            language: Code langue (en, fr, ar, etc.) ou None pour détection auto
            auto_translate_to: Langue cible pour traduction automatique (défaut: 'fr')
                             Si None, pas de traduction

        Returns:
            dict: {
                'text': transcription complète,
                'text_translated': traduction en langue cible (si activé),
                'segments': liste des segments avec timestamps,
                'language': langue détectée,
                'language_probability': probabilité de la langue détectée
            }
        """
        try:
            # Transcription avec détection automatique de langue
            segments, info = self.model.transcribe(
                audio_file_path,
                language=language,  # None = détection auto
                beam_size=5,
                vad_filter=True,  # Voice Activity Detection
                vad_parameters=dict(
                    min_silence_duration_ms=1500  # Augmenté de 500ms à 1500ms pour éviter les coupures
                )
            )

            # Récupérer tous les segments
            segments_list = []
            full_text = []

            for segment in segments:
                segments_list.append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip()
                })
                full_text.append(segment.text.strip())

            transcribed_text = ' '.join(full_text)

            result = {
                'text': transcribed_text,
                'text_original': transcribed_text,
                'segments': segments_list,
                'language': info.language,
                'language_probability': info.language_probability
            }

            # Traduction automatique si langue détectée != langue cible
            if auto_translate_to and info.language != auto_translate_to:
                try:
                    # Importer le traducteur intelligent
                    from smart_translator import SmartTranslator
                    translator = SmartTranslator()

                    # Traduire le texte
                    translated_text = translator.translate(
                        transcribed_text,
                        target_lang=auto_translate_to,
                        source_lang=info.language
                    )

                    result['text_translated'] = translated_text
                    result['text'] = translated_text  # Utiliser la traduction par défaut
                    result['translation_applied'] = True

                    # Éviter les erreurs d'encodage sur Windows
                    try:
                        print(f"✅ Audio traduit: {info.language} → {auto_translate_to}")
                    except UnicodeEncodeError:
                        print(f"[OK] Audio traduit: {info.language} -> {auto_translate_to}")

                except Exception as e:
                    # Éviter les erreurs d'encodage sur Windows
                    try:
                        print(f"⚠️ Traduction échouée: {e}. Utilisation du texte original.")
                    except UnicodeEncodeError:
                        print(f"[WARNING] Traduction echouee: {e}. Utilisation du texte original.")
                    result['text_translated'] = transcribed_text
                    result['translation_applied'] = False
            else:
                result['text_translated'] = transcribed_text
                result['translation_applied'] = False

            return result

        except Exception as e:
            print(f"Erreur lors de la transcription: {e}")
            return {
                'text': '',
                'text_original': '',
                'text_translated': '',
                'segments': [],
                'language': 'unknown',
                'error': str(e)
            }

    def transcribe_audio_data(self, audio_data, sample_rate=16000, language="fr"):
        """
        Transcrit des données audio depuis un array numpy

        Args:
            audio_data: numpy array avec les données audio
            sample_rate: Taux d'échantillonnage
            language: Code langue

        Returns:
            dict: Résultat de la transcription
        """
        # Créer un fichier temporaire
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Sauvegarder l'audio dans un fichier temporaire
            import soundfile as sf
            sf.write(tmp_path, audio_data, sample_rate)

            # Transcrire
            result = self.transcribe_audio(tmp_path, language=language)

            return result

        finally:
            # Nettoyer le fichier temporaire
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def transcribe_audio_bytes(self, audio_bytes, language="en"):
        """
        Transcrit des données audio depuis des bytes (format WAV)

        Args:
            audio_bytes: Bytes audio (format WAV)
            language: Code langue (en, fr, etc.)

        Returns:
            dict: Résultat de la transcription
        """
        # Créer un fichier temporaire
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        try:
            # Transcrire
            result = self.transcribe_audio(tmp_path, language=language)
            return result

        finally:
            # Nettoyer le fichier temporaire
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @staticmethod
    def get_available_models():
        """Retourne la liste des modèles Whisper disponibles"""
        return ["tiny", "base", "small", "medium", "large-v2", "large-v3"]

    @staticmethod
    def estimate_model_size(model_name):
        """Estime la taille et la performance d'un modèle"""
        sizes = {
            "tiny": {"size": "~75 MB", "speed": "~32x", "accuracy": "Basique"},
            "base": {"size": "~145 MB", "speed": "~16x", "accuracy": "Bonne"},
            "small": {"size": "~465 MB", "speed": "~6x", "accuracy": "Très bonne"},
            "medium": {"size": "~1.5 GB", "speed": "~2x", "accuracy": "Excellente"},
            "large-v2": {"size": "~3 GB", "speed": "~1x", "accuracy": "Maximale"},
            "large-v3": {"size": "~3 GB", "speed": "~1x", "accuracy": "Maximale+"}
        }
        return sizes.get(model_name, {"size": "Unknown", "speed": "Unknown", "accuracy": "Unknown"})


def test_audio_processor():
    """Fonction de test pour vérifier l'installation"""
    print("Test du processeur audio...")

    if not WHISPER_AVAILABLE:
        print("[ERROR] faster-whisper non disponible")
        return False

    try:
        # Créer un processeur avec le plus petit modèle
        processor = AudioProcessor(model_size="tiny")
        print("[OK] AudioProcessor initialise avec succes")

        # Afficher les modèles disponibles
        print("\nModèles Whisper disponibles:")
        for model in AudioProcessor.get_available_models():
            info = AudioProcessor.estimate_model_size(model)
            print(f"  - {model}: {info['size']} (vitesse: {info['speed']}, précision: {info['accuracy']})")

        return True

    except Exception as e:
        print(f"[ERROR] Erreur: {e}")
        return False


if __name__ == "__main__":
    test_audio_processor()
