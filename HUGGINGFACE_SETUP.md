# 🤗 Déploiement du Modèle sur Hugging Face Hub

Ce guide explique comment héberger votre modèle ESI sur Hugging Face Hub pour un déploiement gratuit sur Streamlit Cloud.

## 📋 Pourquoi Hugging Face Hub?

- ✅ **Gratuit et illimité** pour les modèles publics
- ✅ **Pas de limite de taille** (contrairement à Git: 1GB)
- ✅ **Téléchargement automatique** au démarrage de l'app
- ✅ **Versioning intégré** du modèle
- ✅ **CDN mondial** pour chargement rapide

## 🚀 Étapes d'Upload (Déjà effectuées)

### 1. Authentification
```bash
pip install huggingface_hub
python login_hf.py
```

### 2. Upload du Modèle
```bash
python upload_to_huggingface.py
```

Le script a uploadé le modèle vers: **https://huggingface.co/yallou/esi-clinical-triage**

## ⚙️ Configuration de l'Application

### Option 1: Variable d'Environnement (Recommandé pour Production)

Sur Streamlit Cloud, ajoutez dans les **Secrets**:
```toml
# .streamlit/secrets.toml (sur Streamlit Cloud)
HF_MODEL_NAME = "yallou/esi-clinical-triage"
```

### Option 2: Variable d'Environnement Locale

Pour tester localement:
```bash
# Windows
set HF_MODEL_NAME=yallou/esi-clinical-triage
streamlit run app.py

# Linux/Mac
export HF_MODEL_NAME=yallou/esi-clinical-triage
streamlit run app.py
```

### Option 3: Fallback Local (Développement)

Si `HF_MODEL_NAME` n'est pas défini, l'app cherchera le modèle dans `model/final_model/` (local).

## 📦 Flux de Chargement du Modèle

```python
# app.py - Fonction charger_modele()

1. Vérifie si HF_MODEL_NAME est défini
   ├─ OUI → Télécharge depuis Hugging Face (production)
   └─ NON → Cherche model/final_model/ (développement)
                ├─ Existe → Charge le modèle local
                └─ N'existe pas → Charge ClinicalBERT de base (fallback)
```

## 🌐 Déploiement sur Streamlit Cloud

### 1. Pousser le Code sur GitHub

```bash
# Ajouter tous les fichiers (sans le modèle local grâce au .gitignore)
git add .
git commit -m "Add Hugging Face model integration"
git push origin main
```

**Important**: Le dossier `model/` (4GB) est maintenant ignoré dans `.gitignore`, donc ne sera PAS poussé sur GitHub.

### 2. Configurer Streamlit Cloud

1. Allez sur [https://share.streamlit.io/](https://share.streamlit.io/)
2. Connectez votre repository GitHub
3. Dans **Advanced settings > Secrets**, ajoutez:

```toml
HF_MODEL_NAME = "yallou/esi-clinical-triage"
```

4. Déployez!

### 3. Premier Démarrage

Au premier démarrage, Streamlit Cloud va:
- ✅ Installer les dépendances (`requirements.txt`)
- ✅ Télécharger le modèle depuis Hugging Face (~433MB)
- ✅ Mettre en cache le modèle (`@st.cache_resource`)

**Temps estimé**: 2-3 minutes pour le premier démarrage, puis instantané grâce au cache.

## 📊 Avantages de cette Architecture

| Aspect | Avant (Git) | Après (Hugging Face) |
|--------|-------------|----------------------|
| **Taille repo** | 8.3GB ❌ | ~50MB ✅ |
| **Limite Streamlit** | Dépasse 1GB ❌ | Sous 1GB ✅ |
| **Temps upload** | Très long | Rapide |
| **Versioning modèle** | Difficile | Natif HF ✅ |
| **Partage modèle** | Impossible | Public HF ✅ |

## 🔄 Mise à Jour du Modèle

Pour mettre à jour le modèle après un nouvel entraînement:

```bash
# 1. Réentraîner le modèle (train.py)
python train.py

# 2. Re-uploader vers Hugging Face
python upload_to_huggingface.py

# 3. Redémarrer l'app Streamlit
# Le cache se rafraîchira automatiquement
```

## 🔒 Modèle Privé (Optionnel)

Si vous voulez garder le modèle privé:

### 1. Modifier le Repository en Privé
Sur Hugging Face → Settings → Make Private

### 2. Créer un Token Read
[https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → New Token → Type: **Read**

### 3. Ajouter le Token dans Streamlit Secrets
```toml
# .streamlit/secrets.toml
HF_MODEL_NAME = "yallou/esi-clinical-triage"
HF_TOKEN = "hf_xxxxxxxxxxxxx"
```

### 4. Modifier app.py
```python
# Dans la fonction charger_modele()
HF_TOKEN = os.getenv("HF_TOKEN", None)

model = AutoModelForSequenceClassification.from_pretrained(
    HF_MODEL_NAME,
    use_auth_token=HF_TOKEN  # ← Ajouter cette ligne
)
```

## 📚 Ressources

- **Modèle Hugging Face**: https://huggingface.co/yallou/esi-clinical-triage
- **Documentation HF Hub**: https://huggingface.co/docs/hub/index
- **Streamlit Cloud Docs**: https://docs.streamlit.io/streamlit-community-cloud

## 🆘 Dépannage

### Erreur: "Model not found"
- Vérifiez que `HF_MODEL_NAME` est correctement défini
- Vérifiez que le modèle existe sur Hugging Face
- Vérifiez votre connexion internet

### Erreur: "Token expired"
- Re-générez un token sur Hugging Face
- Re-connectez-vous avec `python login_hf.py`

### L'app est lente au démarrage
- Normal au premier lancement (télécharge le modèle)
- Les lancements suivants sont rapides grâce au cache

---

✅ **Votre modèle est maintenant hébergé sur Hugging Face!**
✅ **Prêt pour un déploiement gratuit sur Streamlit Cloud!**
