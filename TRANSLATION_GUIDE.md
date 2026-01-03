# 🌐 Guide de Traduction Intelligente - Medical Triage Assistant

## 📋 Vue d'ensemble

Le système de traduction intelligente permet une **traduction automatique et dynamique** de tout le contenu de l'application, y compris:

- ✅ Interface utilisateur (titres, boutons, labels)
- ✅ Contenu dynamique (résumés cliniques, alertes, examens)
- ✅ Résultats d'analyse (raisonnement IA, indicateurs clés)
- ✅ **Audio multilingue** (transcription + traduction automatique)

## 🎯 Fonctionnalités Principales

### 1. Traduction de l'Interface (Statique)
**Fichier:** `app.py` (dictionnaire `TRANSLATIONS`)

Langues supportées:
- 🇫🇷 **Français** (défaut)
- 🇬🇧 **Anglais**
- 🇸🇦 **Arabe**

### 2. Traduction du Contenu Dynamique (Intelligent)
**Module:** `src/smart_translator.py`

Traduit automatiquement:
- Signaux d'alerte et descriptions
- Résumés cliniques (motifs, symptômes, sévérité)
- Examens recommandés
- Raisonnement clinique IA (patterns, red flags, evidence, indicators)

### 3. Traduction Audio (Speech-to-Text Multilingue)
**Module:** `src/audio_processor.py`

Fonctionnalités:
- 🎤 **Détection automatique de langue** (Whisper)
- 🌍 **Traduction automatique** vers français pour analyse ESI
- 📊 Affichage de la langue détectée et du texte original

**Exemple:**
```
Patient parle en arabe → Whisper détecte "ar" → Traduction en français → Analyse ESI
```

## 🛠️ Installation

### 1. Installer les dépendances
```bash
pip install -r requirements.txt
```

Dépendances clés pour la traduction:
- `googletrans==4.0.0rc1` - Google Translate (gratuit, en ligne)
- `requests>=2.31.0` - Pour DeepL API (optionnel)
- `argostranslate>=1.8.0` - Traduction locale hors ligne (optionnel)

### 2. Configuration (Optionnelle)

#### Option A: Google Translate (Par défaut, gratuit)
✅ Aucune configuration requise
✅ Fonctionne immédiatement

#### Option B: DeepL API (Meilleure qualité)
1. Créer un compte gratuit sur [DeepL](https://www.deepl.com/pro-api)
2. Obtenir une clé API (500k caractères/mois gratuits)
3. Ajouter dans `.streamlit/secrets.toml`:
```toml
DEEPL_API_KEY = "votre-clé-api-ici"
```

#### Option C: ArgosTranslate (Local, hors ligne)
```bash
pip install argostranslate
python -m argostranslate.package install-from-path fr en ar
```

## 📖 Utilisation

### Pour les Utilisateurs

#### 1. Changer la langue de l'interface
- Dans l'interface, utilisez le sélecteur de langue en haut à droite
- Sélectionnez: Français 🇫🇷 / English 🇬🇧 / العربية 🇸🇦

#### 2. Utiliser l'audio multilingue
1. Aller dans l'onglet **🎤 Audio**
2. Enregistrer un message (en n'importe quelle langue)
3. Cliquer sur **"🔄 Transcrire et Analyser"**
4. Le système détecte automatiquement la langue et traduit en français pour l'analyse ESI

**Langues audio supportées par Whisper:**
- Français, Anglais, Arabe, Espagnol, Allemand, Italien, Portugais, Russe, Chinois, Japonais, et 90+ autres langues

### Pour les Développeurs

#### 1. Utiliser la fonction `tr()` pour traduire du contenu
```python
from app import tr

# Traduire un texte simple
texte_traduit = tr("Douleur thoracique intense")

# Traduire vers une langue spécifique
texte_en_anglais = tr("Résumé Clinique", target_lang="en")

# Traduire une liste
symptomes = ["Fièvre", "Toux", "Fatigue"]
symptomes_traduits = [tr(s) for s in symptomes]
```

#### 2. Utiliser la classe `SmartTranslator`
```python
from smart_translator import SmartTranslator, auto_translate

translator = SmartTranslator()

# Traduire un texte
result = translator.translate("Bonjour", "en")  # → "Hello"

# Traduire une liste
liste = translator.translate(["Symptôme 1", "Symptôme 2"], "ar")

# Traduire un dictionnaire
data = {"motif": "Douleur abdominale", "niveau": "Urgent"}
data_traduite = translator.translate(data, "en")
```

#### 3. Décorateur pour traduire automatiquement
```python
from smart_translator import translate_output

@translate_output
def get_clinical_summary():
    return "Patient présente des symptômes de..."

# La fonction retourne automatiquement la traduction selon la langue de session
summary = get_clinical_summary()
```

## 🎨 Architecture Technique

### 1. Cache de Traduction
- Les traductions sont mises en cache dans `st.session_state.translation_cache`
- Clé de cache: `MD5(texte + langue_cible)`
- Améliore les performances en évitant les traductions répétées

### 2. Stratégie de Fallback
```
1. Essayer Google Translate (rapide, gratuit)
   ↓ Si échec
2. Essayer DeepL API (si clé configurée)
   ↓ Si échec
3. Essayer ArgosTranslate (local)
   ↓ Si échec
4. Retourner le texte original
```

### 3. Flux Audio → Traduction
```
Audio enregistré
    ↓
Whisper: Transcription + Détection de langue
    ↓
Si langue ≠ français:
    ↓
SmartTranslator: Traduction automatique
    ↓
Texte en français pour analyse ESI
```

## 🧪 Tests

### Test manuel rapide
```python
# Dans un terminal Python
from src.smart_translator import SmartTranslator

translator = SmartTranslator()

# Test 1: Texte simple
print(translator.translate("Douleur thoracique", "en"))
# → "Chest pain"

# Test 2: Liste
symptoms = ["Fièvre", "Toux", "Fatigue"]
print(translator.translate(symptoms, "ar"))
# → ["حمى", "سعال", "إعياء"]

# Test 3: Dictionnaire
data = {"motif": "Urgence", "niveau": "Critique"}
print(translator.translate(data, "en"))
# → {"motif": "Emergency", "niveau": "Critical"}
```

### Test audio multilingue
1. Lancer l'application: `streamlit run app.py`
2. Aller dans l'onglet **🎤 Audio**
3. Enregistrer un message en arabe ou anglais
4. Vérifier que la langue est détectée et traduite en français

## 🔧 Dépannage

### Problème: `ImportError: No module named 'googletrans'`
**Solution:**
```bash
pip install googletrans==4.0.0rc1
```

### Problème: Traduction lente ou timeout
**Solutions:**
1. Installer DeepL API (plus rapide et fiable)
2. Installer ArgosTranslate pour traduction locale:
```bash
pip install argostranslate
```

### Problème: Traduction audio ne fonctionne pas
**Vérifications:**
1. `faster-whisper` est installé
2. Le module `smart_translator.py` est dans `src/`
3. Les dépendances de traduction sont installées

### Problème: Erreur "AttributeError: 'NoneType' object has no attribute 'group'"
**Cause:** Version incompatible de googletrans

**Solution:**
```bash
pip uninstall googletrans
pip install googletrans==4.0.0rc1
```

## 📈 Performances

### Temps de traduction moyens
- **Google Translate:** 0.2-0.5s par texte
- **DeepL API:** 0.1-0.3s par texte
- **ArgosTranslate:** 0.5-1.5s par texte (local)

### Cache
- Premier accès: traduction complète
- Accès suivants: instantané (cache)

## 🚀 Améliorations Futures

- [ ] Support de plus de langues (espagnol, allemand, etc.)
- [ ] Traduction des PDFs générés
- [ ] Détection automatique de la langue du texte saisi
- [ ] Interface de sélection de langue dans l'onglet audio
- [ ] Traduction des graphiques et visualisations
- [ ] Support de modèles de traduction locaux plus avancés (NLLB, M2M100)

## 📚 Ressources

- [Google Translate API](https://py-googletrans.readthedocs.io/)
- [DeepL API](https://www.deepl.com/docs-api)
- [ArgosTranslate](https://github.com/argosopentech/argos-translate)
- [Whisper Multilingual](https://github.com/openai/whisper)

## 📝 Licence

Même licence que le projet principal.

---

**Développé avec ❤️ pour améliorer l'accessibilité médicale multilingue**
