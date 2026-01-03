# 🚀 Guide de Déploiement - Streamlit Cloud

## 📋 Prérequis

1. ✅ Compte GitHub
2. ✅ Compte Streamlit Cloud (gratuit sur [share.streamlit.io](https://share.streamlit.io))
3. ✅ Repository Git configuré

## 🎯 Étapes de Déploiement

### 1. Préparer le Repository Git

```bash
# Initialiser le repository (si pas déjà fait)
git init

# Ajouter tous les fichiers (le .gitignore filtre automatiquement)
git add .

# Vérifier les fichiers qui seront commités
git status

# Créer le premier commit
git commit -m "Initial commit - ESI Triage Assistant v4.0 avec traduction multilingue"

# Ajouter le remote GitHub
git remote add origin https://github.com/VOTRE_USERNAME/medical-triage-assistant.git

# Push vers GitHub
git push -u origin main
```

### 2. Vérifier les Fichiers Requis

Assurez-vous que ces fichiers sont bien commités:

#### ✅ Fichiers Essentiels

- `app.py` - Application principale
- `requirements.txt` - Dépendances Python
- `README.md` - Documentation
- `src/` - Tous les modules source
- `model/` - Modèle entraîné (IMPORTANT!)

#### ⚠️ Vérification du Modèle

```bash
# Vérifier la taille du modèle
du -sh model/

# Si > 1GB, utiliser Git LFS
git lfs install
git lfs track "model/**/*.safetensors"
git add .gitattributes
git commit -m "Add Git LFS for model files"
```

### 3. Optimiser pour Streamlit Cloud

#### Limite de Taille

Streamlit Cloud a une limite de **1GB** pour le repository.

**Solutions si modèle trop gros:**

1. **Utiliser Git LFS** (recommandé)
2. **Héberger le modèle ailleurs** (Hugging Face Hub)
3. **Compresser le modèle** (quantization)

#### Créer `.streamlit/config.toml`

```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
```

#### Ajouter `packages.txt` (si nécessaire)

Pour les dépendances système:

```bash
# packages.txt
ffmpeg
libsndfile1
```

### 4. Déployer sur Streamlit Cloud

#### Étape A: Se Connecter

1. Aller sur [share.streamlit.io](https://share.streamlit.io)
2. Se connecter avec GitHub
3. Autoriser l'accès au repository

#### Étape B: Créer une Nouvelle App

1. Cliquer sur **"New app"**
2. Sélectionner votre repository: `VOTRE_USERNAME/medical-triage-assistant`
3. Branch: `main`
4. Main file path: `app.py`
5. Cliquer sur **"Deploy!"**

#### Étape C: Configuration (Optionnel)

**Advanced settings:**
- **Python version**: 3.9 ou 3.11
- **Secrets**: Pour les clés API (voir ci-dessous)

### 5. Configurer les Secrets (Optionnel)

Pour les clés API de traduction:

Dans Streamlit Cloud → App settings → Secrets:

```toml
# .streamlit/secrets.toml
DEEPL_API_KEY = "votre-clé-deepl-ici"
```

**Note:** Les secrets ne sont JAMAIS committés dans Git!

### 6. Surveiller le Déploiement

#### Logs de Déploiement

Streamlit Cloud affiche les logs en temps réel:

```
⚙️ Preparing system...
⚙️ Spinning up manager process...
⚙️ Provisioning machine...
⚙️ Installing Python dependencies...
⚙️ Starting application...
🎉 Your app is ready!
```

#### Temps de Déploiement

- **Installation des dépendances**: 2-5 minutes
- **Téléchargement du modèle**: 1-3 minutes (première fois)
- **Premier chargement Whisper**: 30-60 secondes
- **Total**: ~5-10 minutes

## ⚡ Optimisations

### 1. Réduire le Temps de Démarrage

```python
# Dans app.py
@st.cache_resource
def load_model():
    """Cache le modèle pour éviter de le recharger"""
    return AutoModelForSequenceClassification.from_pretrained("model/")

@st.cache_resource
def load_whisper():
    """Cache Whisper"""
    return AudioProcessor(model_size="base", device="cpu")
```

### 2. Gérer les Ressources

Streamlit Cloud gratuit:
- **RAM**: 1GB
- **CPU**: Limité
- **GPU**: Non disponible (gratuit)

**Optimisations:**

```python
# Utiliser CPU explicitement
AudioProcessor(model_size="base", device="cpu", compute_type="int8")

# Limiter les threads PyTorch
torch.set_num_threads(2)
```

### 3. Ajouter un Health Check

```python
# Dans app.py
if st.sidebar.button("🏥 Health Check"):
    st.sidebar.success("✅ App Running")
    st.sidebar.info(f"Model: Loaded")
    st.sidebar.info(f"Whisper: {'Loaded' if 'audio_proc' in st.session_state else 'Not loaded'}")
```

## 🐛 Dépannage

### Problème 1: Modèle Non Trouvé

**Erreur:** `FileNotFoundError: model/config.json not found`

**Solution:**

```bash
# Vérifier que le modèle est commité
git ls-files model/

# Si vide, ajouter le modèle
git add model/
git commit -m "Add trained model"
git push
```

### Problème 2: Out of Memory

**Erreur:** `MemoryError` ou app qui crash

**Solutions:**

1. Réduire la taille du modèle Whisper:
```python
AudioProcessor(model_size="tiny")  # Au lieu de "base"
```

2. Désactiver certains modules:
```python
MODULES_OK = False  # Désactive NER, red flags, etc.
```

### Problème 3: Dépendances Manquantes

**Erreur:** `ModuleNotFoundError: No module named 'X'`

**Solution:**

```bash
# Vérifier requirements.txt
cat requirements.txt

# Ajouter la dépendance manquante
echo "package-name>=version" >> requirements.txt

# Commit et push
git add requirements.txt
git commit -m "Add missing dependency"
git push
```

### Problème 4: Timeout au Démarrage

**Erreur:** App ne démarre pas après 10 minutes

**Solutions:**

1. Vérifier les logs Streamlit Cloud
2. Réduire le nombre de dépendances
3. Utiliser un modèle plus petit

## 📊 Monitoring

### Métriques Streamlit Cloud

- **Viewers**: Nombre d'utilisateurs actifs
- **CPU/RAM**: Utilisation des ressources
- **Logs**: Erreurs et warnings

### Analytics Personnalisés

```python
# Dans app.py
if 'session_count' not in st.session_state:
    st.session_state.session_count = 0

st.session_state.session_count += 1

# Afficher dans sidebar
st.sidebar.metric("Sessions", st.session_state.session_count)
```

## 🔄 Mise à Jour

Pour mettre à jour l'app déployée:

```bash
# Faire vos modifications
git add .
git commit -m "Update: description des changements"
git push

# Streamlit Cloud redéploie automatiquement!
```

## 🌐 URL de l'Application

Votre app sera accessible sur:

```
https://VOTRE_USERNAME-medical-triage-assistant-app-HASH.streamlit.app
```

Exemple:
```
https://john-medical-triage-assistant-app-abc123.streamlit.app
```

## 💡 Conseils

1. **Tester localement d'abord**: `streamlit run app.py`
2. **Commits fréquents**: Pour faciliter le rollback
3. **Branches**: Utiliser une branche `dev` pour tester
4. **Logs**: Surveiller les logs Streamlit Cloud régulièrement
5. **Cache**: Utiliser `@st.cache_resource` pour les modèles
6. **Feedback**: Ajouter un formulaire de feedback dans l'app

## 📚 Ressources

- [Documentation Streamlit Cloud](https://docs.streamlit.io/streamlit-community-cloud)
- [Git LFS](https://git-lfs.github.com/)
- [Hugging Face Hub](https://huggingface.co/docs/hub/index) (alternative pour héberger le modèle)

## ✅ Checklist de Déploiement

Avant de déployer:

- [ ] `.gitignore` configuré
- [ ] `requirements.txt` à jour
- [ ] Modèle dans `model/` (< 1GB ou LFS)
- [ ] Tests locaux passés
- [ ] README.md mis à jour
- [ ] Secrets configurés (si API keys)
- [ ] Commit et push vers GitHub
- [ ] App créée sur Streamlit Cloud
- [ ] Premier déploiement réussi
- [ ] Test de l'URL publique
- [ ] Monitoring activé

---

**Bon déploiement!** 🚀

Si vous rencontrez des problèmes, consultez les logs Streamlit Cloud ou ouvrez une issue sur GitHub.
