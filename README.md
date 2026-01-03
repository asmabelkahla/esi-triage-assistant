# 🏥 Assistant de Triage Médical ESI - Multilingue & IA

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Whisper](https://img.shields.io/badge/Whisper-OpenAI-green.svg)](https://github.com/openai/whisper)
[![Translation](https://img.shields.io/badge/Translation-Multilingual-orange.svg)](https://github.com/nidhaloff/deep-translator)

**Système de triage médical intelligent avec IA multilingue et transcription audio**

**Précision:** 85% | **Version:** 4.0 | **Langues:** FR/EN/AR

---

## 📋 Description

Classification automatique des patients en 5 niveaux d'urgence ESI (Emergency Severity Index) utilisant le modèle **ClinicalBERT** fine-tuné, avec support multilingue et transcription audio intelligente.

### 🌟 Nouvelles Fonctionnalités v4.0

- **🌐 Interface Multilingue**: Français, Anglais, Arabe
- **🎤 Transcription Audio**: Whisper AI pour la reconnaissance vocale
- **🔄 Traduction Automatique**: Détection et traduction automatique des langues
- **🧠 Traduction Intelligente**: Tout le contenu dynamique traduit automatiquement
- **📊 Analyse Multilingue**: Analysez des patients parlant n'importe quelle langue

### Niveaux ESI

| Niveau | Urgence | Délai | Exemple |
|--------|---------|-------|---------|
| 🔴 ESI-1 | Immédiate | 0 min | Arrêt cardiaque |
| 🟠 ESI-2 | Très urgente | ≤10 min | Douleur thoracique |
| 🟡 ESI-3 | Urgente | 30-60 min | Fracture |
| 🟢 ESI-4 | Semi-urgente | 1-2h | Entorse |
| 🔵 ESI-5 | Non-urgente | >2h | Rhume |

---

## 📁 Structure du Projet

```
medical_triage_assistant/
│
├── app.py                  # Interface Streamlit
├── train.py                # Entraînement du modèle
├── preprocessing.py        # Préparation des données
│
├── src/                    # Modules Python
│   ├── config.py
│   ├── esi_post_processor.py
│   ├── explainability.py
│   ├── recommendations_engine.py
│   ├── red_flags_detector.py
│   ├── context_enhancer.py
│   ├── ner_extractor.py
│   ├── audio_processor.py      # 🆕 Transcription audio Whisper
│   ├── smart_translator.py     # 🆕 Traduction intelligente
│   └── patient_history.py
│
├── data/                   # Datasets
│   ├── custom_training_data.csv
│   └── esi_data.csv
│
├── model/                  # Modèle fine-tuné
│   └── checkpoint-52/      # Modèle final
│
├── requirements.txt        # Dépendances
└── README.md              # Ce fichier
```

---

## 🚀 Installation

```bash
# Cloner le projet
git clone https://github.com/asmabelkahla/medical_triage_assistant.git
cd medical_triage_assistant

# Créer environnement
conda create -n esi python=3.9
conda activate esi

# Installer dépendances
pip install -r requirements.txt
```

---

## 💻 Utilisation

### 1. Interface Streamlit

**Lancer l'application:**
```bash
# Windows
run_app.bat

# Linux/macOS
streamlit run app.py
```

**Accès:** http://localhost:8501

**Fonctionnalités:**
- 📝 Saisie texte ou 🎤 audio (Whisper)
- 🤖 Prédiction ESI automatique
- 📊 Visualisation probabilités
- 📄 Export PDF rapport
- 🌍 Multilingue (FR/EN/AR)

---

### 2. Entraînement du Modèle

**Préparer les données:**
```bash
python preprocessing.py
```

**Entraîner:**
```bash
# Windows
train.bat

# Linux/macOS
python train.py
```

**Paramètres dans `train.py`:**
- `num_epochs`: Nombre d'époques (défaut: 5)
- `learning_rate`: Taux d'apprentissage (défaut: 1e-5)
- `batch_size`: Taille des batchs (défaut: 8)

---

## 📊 Performance

| Métrique | Valeur |
|----------|--------|
| **Accuracy** | **85%** |
| Precision | 0.85 |
| Recall | 0.84 |
| F1-Score | 0.83 |

**Par classe:**
- ESI-1: F1 = 0.90
- ESI-2: F1 = 0.87
- ESI-3: F1 = 0.82
- ESI-4: F1 = 0.78
- ESI-5: F1 = 0.80

---

## 🔧 Technologies

- **PyTorch** 2.0.1
- **Transformers** 4.35.0 (Hugging Face)
- **ClinicalBERT** (modèle médical)
- **Streamlit** 1.28.1
- **Faster-Whisper** 0.10.0 (audio)
- **Scikit-learn** 1.3.0

---

## 📖 Datasets

**Custom dataset:** 150 cas (30 par niveau ESI)
**MIMIC-IV:** 2000 cas médicaux

**Format CSV:**
```csv
text,esi_label
"Patient 55 ans, douleur thoracique intense...",2
"Enfant 5 ans, fièvre légère...",5
```

---

## ⚠️ Avertissement

Cette application est un **outil d'aide à la décision**. Les prédictions doivent être **validées par un professionnel de santé**.

---

## 📄 Licence

MIT License

---

## 🙏 Remerciements

- **Hugging Face** - Transformers & Model Hub
- **MIT** - ClinicalBERT pre-trained model
- **MTSamples** - Medical transcription dataset for training
- **OpenAI** - Whisper speech recognition model
- **Streamlit** - Interface framework
- **Deep Translator** - Multilingual translation engine

---

**Version:** 4.0
**Date:** Janvier 2026
**Statut:** ✅ Production Ready - Multilingue & Audio IA
