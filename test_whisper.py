"""
Test rapide du modèle Whisper pour le débogage
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_whisper_loading():
    """Test le chargement du modèle Whisper"""
    print("=" * 60)
    print("TEST 1: Chargement du modèle Whisper")
    print("=" * 60)

    try:
        from audio_processor import AudioProcessor
        print("✅ Module audio_processor importé avec succès")

        print("\n🔄 Chargement du modèle Whisper 'base'...")
        print("   (Cela peut prendre 30-60 secondes la première fois)")

        processor = AudioProcessor(model_size="base")
        print("✅ Modèle Whisper chargé avec succès!")

        return processor

    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_basic_transcription(processor):
    """Test une transcription de base (nécessite un fichier audio)"""
    print("\n" + "=" * 60)
    print("TEST 2: Test de transcription (optionnel)")
    print("=" * 60)

    # Vérifier si un fichier de test existe
    test_audio_file = "test_audio.wav"

    if not os.path.exists(test_audio_file):
        print(f"⏭️  Fichier de test '{test_audio_file}' non trouvé")
        print("   Pour tester la transcription, créez un fichier audio de test")
        return

    print(f"\n🎧 Transcription du fichier: {test_audio_file}")

    try:
        result = processor.transcribe_audio(
            test_audio_file,
            language=None,  # Détection automatique
            auto_translate_to='fr'
        )

        print("\n✅ Résultat de la transcription:")
        print(f"   - Langue détectée: {result.get('language', 'N/A')}")
        print(f"   - Probabilité: {result.get('language_probability', 0):.2%}")
        print(f"   - Texte: {result.get('text', 'N/A')[:100]}...")
        print(f"   - Traduction appliquée: {result.get('translation_applied', False)}")

        if result.get('translation_applied'):
            print(f"   - Texte original: {result.get('text_original', 'N/A')[:100]}...")

    except Exception as e:
        print(f"❌ Erreur lors de la transcription: {e}")
        import traceback
        traceback.print_exc()

def test_translator():
    """Test le module de traduction"""
    print("\n" + "=" * 60)
    print("TEST 3: Module de traduction")
    print("=" * 60)

    try:
        from smart_translator import SmartTranslator
        print("✅ Module smart_translator importé avec succès")

        translator = SmartTranslator()
        print("✅ SmartTranslator initialisé")

        # Test de traduction simple
        print("\n🌐 Test de traduction FR → EN:")
        text_fr = "Douleur thoracique intense"
        text_en = translator.translate(text_fr, target_lang='en')
        print(f"   FR: {text_fr}")
        print(f"   EN: {text_en}")

        # Test de traduction EN → FR
        print("\n🌐 Test de traduction EN → FR:")
        text_en2 = "Severe chest pain"
        text_fr2 = translator.translate(text_en2, target_lang='fr')
        print(f"   EN: {text_en2}")
        print(f"   FR: {text_fr2}")

        print("\n✅ Module de traduction fonctionne correctement!")

    except Exception as e:
        print(f"❌ Erreur avec le module de traduction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("\n🔧 TEST DU SYSTÈME DE TRANSCRIPTION AUDIO\n")

    # Test 1: Chargement du modèle
    processor = test_whisper_loading()

    if processor:
        # Test 2: Transcription (optionnel)
        test_basic_transcription(processor)

    # Test 3: Traduction
    test_translator()

    print("\n" + "=" * 60)
    print("✅ TESTS TERMINÉS")
    print("=" * 60)
    print("\nSi tous les tests passent, le système audio devrait fonctionner.")
    print("Sinon, vérifiez les erreurs ci-dessus.\n")
