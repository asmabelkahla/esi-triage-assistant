# 🚀 Guide de Démarrage Rapide - Assistant ESI v4.0

## ⚡ Lancement Rapide

### Option 1: Script de Lancement (Recommandé)

Double-cliquez sur `run_app.bat` ou exécutez dans le terminal:

```bash
run_app.bat
```

L'application vérifie automatiquement les dépendances et lance l'interface!

### Option 2: Lancement Manuel

```bash
# 1. Activer l'environnement
conda activate esi_training

# 2. Lancer l'application
streamlit run app.py
```

## 🌐 Accès à l'Interface

Une fois lancée, l'application est accessible sur:

- **Local**: http://localhost:8501
- **Réseau**: http://192.168.1.16:8501

## 📱 Utilisation de l'Interface

### 1️⃣ Analyse Texte

1. Entrez une description du patient (ex: "Homme 55 ans, douleur thoracique intense")
2. Cliquez sur **"Analyser"**
3. Consultez:
   - Niveau ESI prédit
   - Confiance du modèle
   - Signaux d'alerte
   - Examens recommandés
   - Raisonnement clinique IA

### 2️⃣ Analyse Audio (Nouveau! 🎤)

1. Allez dans l'onglet **"🎤 Audio"**
2. Attendez le chargement du modèle Whisper
3. Cliquez sur le microphone pour enregistrer (3-10 secondes minimum)
4. Cliquez à nouveau pour arrêter
5. Cliquez sur **"🔄 Transcrire et Analyser"**
6. Le système:
   - Transcrit automatiquement votre voix
   - Détecte la langue (FR/EN/AR/etc.)
   - Traduit en français si nécessaire
   - Analyse le cas médical

### 3️⃣ Changer de Langue

- En haut à droite, sélectionnez votre langue:
  - **Français 🇫🇷**
  - **English 🇬🇧**
  - **العربية 🇸🇦**

Toute l'interface et les résultats sont traduits automatiquement!

## 🎯 Exemples d'Utilisation

### Exemple 1: Patient Critique (ESI-1)

**Texte:**
```
Patient inconscient, pas de pouls, pas de respiration
```

**Résultat attendu:**
- ESI-1 (Immédiate)
- Délai: 0 min
- Alertes critiques multiples

### Exemple 2: Patient Urgent (ESI-2)

**Audio (en français):**
> "Homme de 60 ans avec douleur thoracique irradiant vers le bras gauche depuis 30 minutes"

**Résultat attendu:**
- ESI-2 (Très urgente)
- Délai: ≤10 min
- Examens: ECG STAT, Troponine

### Exemple 3: Patient Multilingue (Nouveau!)

**Audio (en arabe):**
> "رجل عمره 45 سنة يعاني من صداع شديد"

**Processus:**
1. Whisper détecte: Arabe (ar)
2. Traduction auto: "Homme 45 ans souffrant de maux de tête sévères"
3. Analyse ESI en français
4. Résultats affichés dans votre langue choisie

## 🔧 Résolution de Problèmes

### L'audio ne fonctionne pas

**Problème:** "Library cublas64_12.dll is not found"

**Solution:** Le système utilise automatiquement le CPU au lieu du GPU. C'est normal et fonctionnel.

**Problème:** L'enregistrement tourne sans résultat

**Solutions:**
1. Vérifiez que vous parlez clairement pendant 3-10 secondes minimum
2. Autorisez l'accès au microphone dans votre navigateur
3. Vérifiez la console pour les messages d'erreur

### Erreur d'encodage

**Problème:** "charmap codec can't encode characters"

**Solution:** Déjà corrigé dans la v4.0. Si le problème persiste, lancez via `run_app.bat` qui configure l'encodage UTF-8.

### Modèle Whisper lent

**Cause:** Le modèle utilise le CPU (pas de GPU CUDA disponible)

**Performance attendue:**
- Temps de chargement initial: 30-60 secondes (une seule fois)
- Transcription: 3-10 secondes pour 30 secondes d'audio

**Pour accélérer (optionnel):**
Installez CUDA Toolkit si vous avez une carte NVIDIA GPU.

### Traduction ne fonctionne pas

**Problème:** Le contenu n'est pas traduit

**Solution:**
```bash
pip install deep-translator
```

Vérifiez ensuite que vous avez une connexion Internet (Google Translate nécessaire).

## 📊 Performance & Limitations

### Temps de Réponse

- **Analyse texte**: < 1 seconde
- **Transcription audio**: 3-10 secondes (CPU) ou 0.5-2 secondes (GPU)
- **Traduction**: 0.2-0.5 secondes (cache après première utilisation)

### Limitations Connues

1. **Audio**: Nécessite une parole claire (pas de fond sonore important)
2. **Traduction**: Nécessite connexion Internet pour Google Translate
3. **Langues audio**: Whisper supporte 90+ langues, mais la qualité varie
4. **Taille audio**: Recommandé 3-30 secondes

## 🆘 Besoin d'Aide?

- **Documentation complète**: Voir [TRANSLATION_GUIDE.md](TRANSLATION_GUIDE.md)
- **Issues**: Signalez les bugs sur GitHub
- **Questions**: Consultez le [README.md](README.md)

## 🎓 Conseils d'Utilisation

### Pour de Meilleurs Résultats

1. **Texte**: Soyez spécifique (âge, symptômes, durée, intensité)
2. **Audio**: Parlez clairement et distinctement pendant au moins 5 secondes
3. **Multilingue**: Le système traduit automatiquement, parlez dans votre langue naturelle
4. **Historique**: Utilisez l'historique de session pour comparer plusieurs cas

### Workflow Recommandé

```
1. Patient arrive
   ↓
2. Enregistrement audio rapide (description du cas)
   ↓
3. Transcription + Traduction automatique
   ↓
4. Analyse ESI instantanée
   ↓
5. Consultation des recommandations
   ↓
6. Export PDF du rapport (optionnel)
```

## 🌟 Fonctionnalités Avancées

- **Cache de traduction**: Les traductions sont mises en cache pour des réponses instantanées
- **Multi-langues simultanées**: Chaque utilisateur peut avoir sa propre langue d'interface
- **Détection automatique**: Whisper détecte automatiquement la langue parlée
- **Fallback intelligent**: Si la traduction échoue, le texte original est utilisé

---

**Version**: 4.0 | **Dernière mise à jour**: 2026-01-03

Bon triage! 🏥✨
