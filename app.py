# -*- coding: utf-8 -*-
"""
⚕️ Interface Médicale Professionnelle - Triage ESI
Version optimisée pour le personnel médical
"""

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import sys
import os
from datetime import datetime
from io import BytesIO

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from audio_recorder_streamlit import audio_recorder
    from audio_processor import AudioProcessor
    AUDIO_OK = True
except Exception as e:
    AUDIO_OK = False
    print(f"⚠️ Audio features disabled: {str(e)}")

try:
    from ner_extractor import MedicalNER
    from red_flags_detector import RedFlagsDetector
    from recommendations_engine import RecommendationsEngine
    from explainability import ExplainabilityEngine
    from esi_post_processor import ESIPostProcessor
    from context_enhancer import ContextEnhancer
    from smart_translator import SmartTranslator, auto_translate
    MODULES_OK = True
    TRANSLATOR_OK = True
except Exception as e:
    MODULES_OK = False
    TRANSLATOR_OK = False
    print(f"⚠️ Advanced modules disabled: {str(e)}")

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    PDF_OK = True
except Exception as e:
    PDF_OK = False
    print(f"⚠️ PDF generation disabled: {str(e)}")

# Configuration
st.set_page_config(
    page_title="Triage ESI - Assistant Médical",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Style CSS créatif - Dashboard médical moderne
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    /* Global Styles */
    * {
        font-family: 'Poppins', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        min-height: 100vh;
    }

    /* Glassmorphism header avec fond coloré élégant */
    .compact-header {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 100%);
        backdrop-filter: blur(10px);
        padding: 1.5rem 2rem;
        border-radius: 20px;
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.3);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    .compact-header h1 {
        font-size: 2rem;
        font-weight: 700;
        color: white !important;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 1rem;
        letter-spacing: -0.5px;
        text-shadow: 0 3px 10px rgba(0,0,0,0.3);
    }

    /* KPI Cards Créatifs avec Glassmorphism */
    .kpi-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.18);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--kpi-color-start), var(--kpi-color-end));
    }

    .kpi-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 12px 48px rgba(31, 38, 135, 0.25);
    }

    .kpi-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        filter: drop-shadow(0 4px 8px rgba(0,0,0,0.1));
    }

    .kpi-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.5rem 0 0.25rem 0;
        background: linear-gradient(135deg, var(--kpi-color-start), var(--kpi-color-end));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .kpi-label {
        font-size: 0.75rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    .kpi-trend {
        font-size: 0.7rem;
        color: #10b981;
        margin-top: 0.5rem;
        font-weight: 500;
    }

    /* History table */
    .history-table {
        font-size: 0.85rem;
    }

    .history-table th {
        background: #f3f4f6;
        padding: 0.5rem;
        font-weight: 600;
        color: #374151;
    }

    .history-table td {
        padding: 0.5rem;
        border-bottom: 1px solid #e5e7eb;
    }

    /* ESI result avec animation */
    .esi-result {
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
        border: none;
        position: relative;
        overflow: hidden;
        animation: slideIn 0.5s ease-out;
    }

    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .esi-result::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(255,255,255,0.7));
        z-index: 0;
    }

    .esi-result > * {
        position: relative;
        z-index: 1;
    }

    .esi-1-bg {
        background: linear-gradient(135deg, #fee2e2 0%, #fca5a5 100%);
        color: #7f1d1d;
    }
    .esi-2-bg {
        background: linear-gradient(135deg, #fed7aa 0%, #fb923c 100%);
        color: #7c2d12;
    }
    .esi-3-bg {
        background: linear-gradient(135deg, #fef3c7 0%, #fbbf24 100%);
        color: #78350f;
    }
    .esi-4-bg {
        background: linear-gradient(135deg, #d1fae5 0%, #34d399 100%);
        color: #065f46;
    }
    .esi-5-bg {
        background: linear-gradient(135deg, #dbeafe 0%, #60a5fa 100%);
        color: #1e3a8a;
    }

    /* Boutons avec gradients */
    .stButton>button {
        border-radius: 12px;
        font-weight: 600;
        height: 3rem;
        font-size: 1rem;
        border: none;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }

    .stButton>button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.4);
    }

    .stButton>button:not([kind="primary"]) {
        background: rgba(255, 255, 255, 0.9);
        color: #667eea;
        border: 2px solid #667eea;
    }

    /* Text Area avec glassmorphism */
    .stTextArea textarea {
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        font-size: 0.95rem;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
    }

    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.2);
    }

    /* Sections avec style */
    h3, h4 {
        font-weight: 700;
        font-size: 1.1rem;
        color: white;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Cards conteneur */
    .content-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }

    /* Hide Streamlit elements */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Logo */
    .medical-logo {
        font-size: 4rem;
        color: #6366f1;
        text-align: center;
        margin: 1rem 0;
        filter: drop-shadow(0 4px 8px rgba(99, 102, 241, 0.3));
    }

    /* Dashboard metrics */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 4px solid;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1;
        margin: 0.5rem 0;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }

    /* Barres de progression modernes */
    .probability-bar {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        height: 10px;
        overflow: hidden;
        margin: 0.5rem 0;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }

    .probability-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
        background: linear-gradient(90deg, var(--bar-color), var(--bar-color-light));
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }

    /* Progress animation */
    @keyframes progressLoad {
        0% { width: 0; }
    }

    .probability-fill {
        animation: progressLoad 0.8s ease-out;
    }

    /* Enhanced alerts */
    .alert-banner {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 6px solid;
        font-weight: 600;
        background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 100%);
        backdrop-filter: blur(10px);
    }

    .critical-alert {
        border-left-color: #dc2626;
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        color: #7f1d1d;
    }

    .warning-alert {
        border-left-color: #f59e0b;
        background: linear-gradient(135deg, #fefce8 0%, #fef3c7 100%);
        color: #78350f;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def charger_modele():
    """Charge le modèle ESI depuis Hugging Face ou local"""

    # Configuration du modèle - Modifier ici pour utiliser votre modèle Hugging Face
    # Format: "votre-username/esi-clinical-triage"
    # Laisser None pour utiliser le modèle local
    HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", None)  # Ex: "username/esi-clinical-triage"
    LOCAL_MODEL_PATH = "model/final_model"

    try:
        if HF_MODEL_NAME:
            # Charger depuis Hugging Face Hub
            print(f"📥 Chargement du modèle depuis Hugging Face: {HF_MODEL_NAME}")
            model = AutoModelForSequenceClassification.from_pretrained(
                HF_MODEL_NAME,
                trust_remote_code=False
            )
            tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
            print(f"✅ Modèle chargé depuis Hugging Face")
        else:
            # Fallback: charger depuis le dossier local
            print(f"📂 Chargement du modèle local: {LOCAL_MODEL_PATH}")
            if os.path.exists(LOCAL_MODEL_PATH):
                model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH)
                tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
                print(f"✅ Modèle local chargé")
            else:
                # Fallback ultime: ClinicalBERT de base (non fine-tuné)
                print("⚠️ Modèle local introuvable, utilisation de ClinicalBERT de base")
                model = AutoModelForSequenceClassification.from_pretrained(
                    'emilyalsentzer/Bio_ClinicalBERT',
                    num_labels=5
                )
                tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

        model.eval()
        return model, tokenizer

    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        # En cas d'erreur, charger le modèle de base
        print("🔄 Chargement du modèle ClinicalBERT de base...")
        model = AutoModelForSequenceClassification.from_pretrained(
            'emilyalsentzer/Bio_ClinicalBERT',
            num_labels=5
        )
        tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        model.eval()
        return model, tokenizer

def predire_esi(texte, model, tokenizer):
    """Prédiction du niveau ESI"""
    inputs = tokenizer(texte, return_tensors='pt', truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)[0]

    esi_predit = int(torch.argmax(probs).item()) + 1
    confiance = float(probs[esi_predit-1]) * 100

    return esi_predit, confiance

def analyser_patient(texte, model, tokenizer):
    """Analyse complète du patient"""
    # Prédiction ESI initiale
    inputs = tokenizer(texte, return_tensors='pt', truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)[0]

    probabilities = {i+1: float(probs[i]) * 100 for i in range(5)}
    esi_initial = int(torch.argmax(probs).item()) + 1
    confiance_initial = probabilities[esi_initial]

    # POST-PROCESSING: Ajustement contextuel
    esi = esi_initial
    confiance = confiance_initial
    post_processing_reason = ""

    # ⚠️ POST-PROCESSOR DÉSACTIVÉ TEMPORAIREMENT POUR DIAGNOSTIC
    # Raison: Le post-processor peut sur-ajuster et causer des prédictions incorrectes
    # Pour réactiver, décommentez le bloc ci-dessous

    USE_POST_PROCESSOR = True  # ← Changez à True pour réactiver

    if USE_POST_PROCESSOR and MODULES_OK:
        try:
            post_processor = ESIPostProcessor()
            esi, confiance, probabilities, post_processing_reason = post_processor.adjust_prediction(
                texte, esi_initial, confiance_initial, probabilities
            )
        except:
            pass

    resultats = {
        'esi': esi,
        'confiance': confiance,
        'esi_initial': esi_initial if post_processing_reason else None,
        'post_processing_reason': post_processing_reason,
        'probabilites_detaillees': probabilities  # Pour diagnostic
    }

    if not MODULES_OK:
        return resultats

    try:
        # NER
        ner = MedicalNER()
        entites = ner.extract_entities(texte)
        resultats['resume'] = ner.get_summary(entites)

        # Red Flags
        detecteur = RedFlagsDetector()
        flags = detecteur.detect(texte)
        resultats['alertes'] = detecteur.get_summary(flags)
        resultats['flags_liste'] = flags

        # Recommandations (utiliser ESI ajusté)
        reco = RecommendationsEngine()
        resultats['examens'] = reco.generate_recommendations(texte, esi, flags)

        # Explicabilité (utiliser ESI ajusté)
        explic = ExplainabilityEngine()
        resultats['raisonnement'] = explic.generate_explanation(
            text=texte, predicted_esi=esi, confidence=confiance,
            probabilities=probabilities,
            red_flags=flags, entities=entites
        )
    except:
        pass

    return resultats

def generer_pdf_rapport(texte_patient, resultats):
    """Génère un rapport PDF médical professionnel"""
    if not PDF_OK:
        return None

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                           rightMargin=2*cm, leftMargin=2*cm,
                           topMargin=2*cm, bottomMargin=2*cm)

    # Styles
    styles = getSampleStyleSheet()
    story = []

    # Style sobre pour le titre
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        alignment=0,  # Left aligned
        fontName='Helvetica-Bold'
    )

    # Style sobre pour les sections
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=8,
        spaceBefore=15,
        fontName='Helvetica-Bold',
        borderWidth=1,
        borderColor=colors.HexColor('#bdc3c7'),
        borderPadding=5,
        backColor=colors.HexColor('#ecf0f1')
    )

    # En-tête sobre
    story.append(Paragraph("RAPPORT DE TRIAGE - URGENCES", title_style))
    story.append(Spacer(1, 0.3*cm))

    # Date et heure dans un tableau sobre
    now = datetime.now()
    date_str = now.strftime("%d/%m/%Y à %H:%M:%S")
    info_header = [
        ['Date d\'évaluation:', date_str],
        ['Système:', 'Emergency Severity Index (ESI)']
    ]

    header_table = Table(info_header, colWidths=[4*cm, 12*cm])
    header_table.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#7f8c8d')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
    ]))

    story.append(header_table)
    story.append(Spacer(1, 0.6*cm))

    # Ligne de séparation sobre
    story.append(Paragraph("<para borderWidth='1' borderColor='#bdc3c7'><br/></para>", styles['Normal']))
    story.append(Spacer(1, 0.4*cm))

    # Résultat ESI
    esi = resultats['esi']
    confiance = resultats['confiance']

    infos_esi = {
        1: {"nom": "ESI-1 (Immédiate)", "delai": "0 min"},
        2: {"nom": "ESI-2 (Très urgente)", "delai": "≤10 min"},
        3: {"nom": "ESI-3 (Urgente)", "delai": "30-60 min"},
        4: {"nom": "ESI-4 (Semi-urgente)", "delai": "1-2h"},
        5: {"nom": "ESI-5 (Non-urgente)", "delai": ">2h"}
    }

    info = infos_esi[esi]

    story.append(Paragraph("1. RÉSULTAT DU TRIAGE", heading_style))
    story.append(Spacer(1, 0.2*cm))

    # Tableau ESI sobre
    esi_data = [
        ['Niveau de triage:', info['nom']],
        ['Délai maximum d\'évaluation:', info['delai']],
        ['Confiance du système:', f"{confiance:.1f}%"]
    ]

    esi_table = Table(esi_data, colWidths=[6*cm, 10*cm])
    esi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
    ]))

    story.append(esi_table)
    story.append(Spacer(1, 0.5*cm))

    # Description du patient
    story.append(Paragraph("2. PRÉSENTATION CLINIQUE", heading_style))
    story.append(Spacer(1, 0.2*cm))

    desc_style = ParagraphStyle(
        'DescStyle',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#2c3e50')
    )
    story.append(Paragraph(texte_patient, desc_style))
    story.append(Spacer(1, 0.5*cm))

    # Alertes
    if resultats.get('alertes') and resultats['alertes']['total'] > 0:
        story.append(Paragraph("3. SIGNAUX D'ALERTE", heading_style))
        story.append(Spacer(1, 0.2*cm))
        alertes = resultats['alertes']

        alert_data = [
            ['Nombre de signaux détectés:', str(alertes['total'])],
            ['Niveau de risque:', alertes['risk_level']]
        ]

        alert_table = Table(alert_data, colWidths=[6*cm, 10*cm])
        alert_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
        ]))

        story.append(alert_table)

        # Liste des flags
        if resultats.get('flags_liste'):
            story.append(Spacer(1, 0.3*cm))
            list_style = ParagraphStyle(
                'ListStyle',
                parent=styles['Normal'],
                fontSize=9,
                leftIndent=10,
                textColor=colors.HexColor('#34495e')
            )
            for flag in resultats['flags_liste'][:10]:
                story.append(Paragraph(f"• {flag.description}", list_style))

        story.append(Spacer(1, 0.5*cm))

    # Résumé clinique
    section_num = 4 if resultats.get('alertes') and resultats['alertes']['total'] > 0 else 3

    if resultats.get('resume'):
        story.append(Paragraph(f"{section_num}. RÉSUMÉ CLINIQUE", heading_style))
        story.append(Spacer(1, 0.2*cm))
        resume = resultats['resume']

        resume_style = ParagraphStyle(
            'ResumeStyle',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#2c3e50')
        )

        if resume.get('chief_complaint'):
            story.append(Paragraph(f"<b>Motif de consultation:</b> {resume['chief_complaint']}", resume_style))

        if resume.get('symptoms_list'):
            symptoms = ', '.join(resume['symptoms_list'][:6])
            story.append(Paragraph(f"<b>Symptômes:</b> {symptoms}", resume_style))

        if resume.get('severity_level'):
            story.append(Paragraph(f"<b>Sévérité:</b> {resume['severity_level']}", resume_style))

        if resume.get('temporal_onset'):
            story.append(Paragraph(f"<b>Début:</b> {resume['temporal_onset']}", resume_style))

        story.append(Spacer(1, 0.5*cm))
        section_num += 1

    # Examens recommandés
    if resultats.get('examens'):
        story.append(Paragraph(f"{section_num}. EXAMENS RECOMMANDÉS", heading_style))
        story.append(Spacer(1, 0.2*cm))
        examens = resultats['examens']

        exam_style = ParagraphStyle(
            'ExamStyle',
            parent=styles['Normal'],
            fontSize=9,
            leftIndent=10,
            textColor=colors.HexColor('#34495e')
        )

        stat = [e for e in examens if e.priority == 'STAT']
        urgent = [e for e in examens if e.priority == 'URGENT']
        routine = [e for e in examens if e.priority == 'ROUTINE']

        if stat:
            story.append(Paragraph("<b>Priorité STAT (immédiate):</b>", styles['Normal']))
            for e in stat:
                story.append(Paragraph(f"• {e.exam}", exam_style))
            story.append(Spacer(1, 0.2*cm))

        if urgent:
            story.append(Paragraph("<b>Priorité URGENT:</b>", styles['Normal']))
            for e in urgent:
                story.append(Paragraph(f"• {e.exam}", exam_style))
            story.append(Spacer(1, 0.2*cm))

        if routine:
            story.append(Paragraph("<b>Routine:</b>", styles['Normal']))
            for e in routine:
                story.append(Paragraph(f"• {e.exam}", exam_style))

        story.append(Spacer(1, 0.5*cm))

    # Raisonnement clinique
    if resultats.get('raisonnement'):
        story.append(Paragraph(f"{section_num}. JUSTIFICATION CLINIQUE", heading_style))
        story.append(Spacer(1, 0.2*cm))

        reasoning_style = ParagraphStyle(
            'ReasoningStyle',
            parent=styles['Normal'],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor('#34495e'),
            leftIndent=5
        )

        raisonnement = resultats['raisonnement']

        # Pattern clinique principal
        if raisonnement.clinical_pattern:
            story.append(Paragraph(f"<b>Justification:</b> {raisonnement.clinical_pattern}", reasoning_style))

        # Indicateurs clés
        if raisonnement.key_indicators:
            story.append(Spacer(1, 0.2*cm))
            story.append(Paragraph("<b>Indicateurs détectés:</b>", reasoning_style))
            for ind in raisonnement.key_indicators[:5]:
                story.append(Paragraph(f"• {ind}", reasoning_style))

        story.append(Spacer(1, 0.5*cm))

    # Footer sobre
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("<para borderWidth='0.5' borderColor='#bdc3c7'><br/></para>", styles['Normal']))
    story.append(Spacer(1, 0.3*cm))

    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#7f8c8d'),
        alignment=1
    )
    story.append(Paragraph("Système d'aide à la décision médicale - ESI Triage Assistant", footer_style))
    story.append(Paragraph("Ce rapport est un outil d'aide. Le jugement clinique reste primordial.", footer_style))
    story.append(Spacer(1, 0.1*cm))
    story.append(Paragraph(f"Document généré automatiquement le {date_str}", footer_style))

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def tr(text, target_lang=None):
    """
    Helper de traduction automatique pour le contenu dynamique

    Args:
        text: Texte ou structure à traduire
        target_lang: Langue cible (défaut: langue session)

    Returns:
        Texte traduit
    """
    if not TRANSLATOR_OK:
        return text

    if target_lang is None:
        target_lang = st.session_state.get('language', 'fr')

    # Si français, pas de traduction
    if target_lang == 'fr':
        return text

    try:
        return auto_translate(text, target_lang)
    except:
        return text


def afficher_resultat_esi(esi, confiance, t):
    """Affiche le résultat ESI - Version compacte"""
    infos = {
        1: {"delai": "0 min", "classe": "esi-1-bg", "icone": "🔴"},
        2: {"delai": "≤10 min", "classe": "esi-2-bg", "icone": "🟠"},
        3: {"delai": "30-60 min", "classe": "esi-3-bg", "icone": "🟡"},
        4: {"delai": "1-2h", "classe": "esi-4-bg", "icone": "🟢"},
        5: {"delai": ">2h", "classe": "esi-5-bg", "icone": "🔵"}
    }

    info = infos[esi]

    st.markdown(f"""
    <div class="esi-result {info['classe']}">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div style="font-size: 0.75rem; font-weight: 500; opacity: 0.8; margin-bottom: 0.25rem;">{t['esi_level']}</div>
                <div style="font-size: 1.5rem; font-weight: 700; margin: 0;">
                    {info['icone']} ESI-{esi} - {t['esi_names'][esi]}
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.75rem; font-weight: 500; opacity: 0.8; margin-bottom: 0.25rem;">{t['confidence']}</div>
                <div style="font-size: 1.5rem; font-weight: 700;">{confiance:.0f}%</div>
            </div>
        </div>
        <div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid rgba(0,0,0,0.1);
             font-size: 0.85rem; font-weight: 500;">
            ⏱️ {t['max_delay']}: {info['delai']}
        </div>
    </div>
    """, unsafe_allow_html=True)

def afficher_dashboard_probabilites(probabilites, esi_predit, t):
    """Affiche un dashboard créatif des probabilités"""
    st.markdown(f"#### 📊 {t['probabilities']}")

    st.markdown('<div class="content-card">', unsafe_allow_html=True)

    infos = {
        1: {"couleur": "#dc2626", "icone": "🔴", "light": "#fca5a5"},
        2: {"couleur": "#ea580c", "icone": "🟠", "light": "#fb923c"},
        3: {"couleur": "#f59e0b", "icone": "🟡", "light": "#fbbf24"},
        4: {"couleur": "#10b981", "icone": "🟢", "light": "#6ee7b7"},
        5: {"couleur": "#3b82f6", "icone": "🔵", "light": "#93c5fd"}
    }

    for esi_level in range(1, 6):
        prob = probabilites.get(esi_level, 0)
        info = infos[esi_level]
        is_predicted = (esi_level == esi_predit)

        # Style spécial pour la prédiction
        glow = f"box-shadow: 0 0 20px rgba({int(info['couleur'][1:3], 16)}, {int(info['couleur'][3:5], 16)}, {int(info['couleur'][5:7], 16)}, 0.4);" if is_predicted else ""
        scale = "transform: scale(1.03);" if is_predicted else ""

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.85) 100%);
             padding: 1rem; border-radius: 12px; margin-bottom: 1rem;
             border: 2px solid {'#667eea' if is_predicted else '#e5e7eb'};
             {glow} {scale} transition: all 0.3s ease;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <span style="font-size: 1.5rem; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));">{info['icone']}</span>
                    <div>
                        <div style="font-weight: 700; font-size: 0.95rem; color: #1f2937;">ESI-{esi_level}</div>
                        <div style="font-size: 0.75rem; color: #6b7280;">{t['esi_names'][esi_level]}</div>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-weight: 800; font-size: 1.8rem;
                         background: linear-gradient(135deg, {info['couleur']} 0%, {info['light']} 100%);
                         -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                        {prob:.1f}%
                    </div>
                </div>
            </div>
            <div class="probability-bar">
                <div class="probability-fill" style="--bar-color: {info['couleur']}; --bar-color-light: {info['light']}; width: {prob}%;"></div>
            </div>
            {f'<div style="text-align: center; margin-top: 0.75rem; padding: 0.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 6px; font-weight: 700; font-size: 0.8rem; letter-spacing: 1px;">{t["predicted_level"]}</div>' if is_predicted else ''}
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Dictionnaire de traductions
TRANSLATIONS = {
    'fr': {
        'title': '⚕️ Système de Triage ESI',
        'subtitle': 'Assistant Médical Intelligent',
        'total_cases': 'Total Cas Analysés',
        'critical_cases': 'Cas Critiques (ESI ≤2)',
        'avg_confidence': 'Confiance Moyenne',
        'most_frequent': 'Niveau le Plus Fréquent',
        'session_running': '↗ Session en cours',
        'high_reliability': '✓ Fiabilité élevée',
        'of_total': 'du total',
        'cases': 'cas',
        'history': 'Historique Session',
        'distribution': 'Distribution ESI',
        'no_cases': 'Aucun cas analysé',
        'showing': 'Affichage des 10 derniers sur',
        'patient_desc': 'Description du Patient',
        'analyze': 'Analyser',
        'reset': 'Réinitialiser',
        'placeholder': 'Ex: Homme 55 ans, douleur thoracique intense...',
        'analyzing': 'Analyse en cours...',
        'esi_level': 'NIVEAU ESI',
        'confidence': 'CONFIANCE',
        'confidence_text': 'Confiance',
        'max_delay': 'Délai maximum',
        'probabilities': 'Analyse des Probabilités',
        'predicted_level': '✓ NIVEAU PRÉDIT',
        'esi_names': {
            1: 'IMMÉDIATE',
            2: 'TRÈS URGENTE',
            3: 'URGENTE',
            4: 'SEMI-URGENTE',
            5: 'NON-URGENTE'
        },
        'language': '🌐 Langue',
        'tab_text': '📝 Texte',
        'tab_audio': '🎤 Audio'
    },
    'en': {
        'title': '⚕️ ESI Triage System',
        'subtitle': 'Intelligent Medical Assistant',
        'total_cases': 'Total Cases Analyzed',
        'critical_cases': 'Critical Cases (ESI ≤2)',
        'avg_confidence': 'Average Confidence',
        'most_frequent': 'Most Frequent Level',
        'session_running': '↗ Session ongoing',
        'high_reliability': '✓ High reliability',
        'of_total': 'of total',
        'cases': 'cases',
        'history': 'Session History',
        'distribution': 'ESI Distribution',
        'no_cases': 'No cases analyzed',
        'showing': 'Showing last 10 out of',
        'patient_desc': 'Patient Description',
        'analyze': 'Analyze',
        'reset': 'Reset',
        'placeholder': 'Ex: 55-year-old male, severe chest pain...',
        'analyzing': 'Analyzing...',
        'esi_level': 'ESI LEVEL',
        'confidence': 'CONFIDENCE',
        'confidence_text': 'Confidence',
        'max_delay': 'Maximum delay',
        'probabilities': 'Probability Analysis',
        'predicted_level': '✓ PREDICTED LEVEL',
        'esi_names': {
            1: 'IMMEDIATE',
            2: 'VERY URGENT',
            3: 'URGENT',
            4: 'SEMI-URGENT',
            5: 'NON-URGENT'
        },
        'language': '🌐 Language',
        'tab_text': '📝 Text',
        'tab_audio': '🎤 Audio'
    },
    'ar': {
        'title': '⚕️ نظام فرز ESI',
        'subtitle': 'مساعد طبي ذكي',
        'total_cases': 'إجمالي الحالات المحللة',
        'critical_cases': 'حالات حرجة (ESI ≤2)',
        'avg_confidence': 'متوسط الثقة',
        'most_frequent': 'المستوى الأكثر تكرارا',
        'session_running': '↗ جلسة جارية',
        'high_reliability': '✓ موثوقية عالية',
        'of_total': 'من الإجمالي',
        'cases': 'حالات',
        'history': 'سجل الجلسة',
        'distribution': 'توزيع ESI',
        'no_cases': 'لا توجد حالات محللة',
        'showing': 'عرض آخر 10 من',
        'patient_desc': 'وصف المريض',
        'analyze': 'تحليل',
        'reset': 'إعادة تعيين',
        'placeholder': 'مثال: رجل 55 سنة، ألم شديد في الصدر...',
        'analyzing': 'جاري التحليل...',
        'esi_level': 'مستوى ESI',
        'confidence': 'الثقة',
        'confidence_text': 'الثقة',
        'max_delay': 'أقصى تأخير',
        'probabilities': 'تحليل الاحتمالات',
        'predicted_level': '✓ المستوى المتوقع',
        'esi_names': {
            1: 'فوري',
            2: 'عاجل جدا',
            3: 'عاجل',
            4: 'شبه عاجل',
            5: 'غير عاجل'
        },
        'language': '🌐 اللغة',
        'tab_text': '📝 نص',
        'tab_audio': '🎤 صوت'
    }
}

def main():
    # Initialize session state
    if 'historique' not in st.session_state:
        st.session_state.historique = []
    if 'language' not in st.session_state:
        st.session_state.language = 'fr'

    # Charger traductions
    t = TRANSLATIONS[st.session_state.language]

    # Sélecteur de langue dans le header
    lang_col1, lang_col2 = st.columns([5, 1])

    with lang_col1:
        st.markdown(f'<div class="compact-header"><h1 style="color: white !important;">{t["title"]}</h1></div>', unsafe_allow_html=True)

    with lang_col2:
        st.markdown('<div style="margin-top: 1.5rem;">', unsafe_allow_html=True)
        lang_display = {
            'fr': 'Français 🇫🇷',
            'en': 'English 🇬🇧',
            'ar': 'العربية 🇸🇦'
        }

        current_index = list(lang_display.keys()).index(st.session_state.language)

        selected_lang = st.selectbox(
            'Langue',
            options=list(lang_display.values()),
            index=current_index,
            key='lang_selector'
        )

        # Mettre à jour la langue
        for code, display in lang_display.items():
            if display == selected_lang:
                if st.session_state.language != code:
                    st.session_state.language = code
                    st.rerun()
                break

        st.markdown('</div>', unsafe_allow_html=True)

    # Charger modèle
    try:
        model, tokenizer = charger_modele()
    except Exception as e:
        st.error(f"❌ Erreur: {e}")
        st.stop()

    # Display feature status (diagnostic)
    with st.expander("🔧 Status des fonctionnalités", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            status_pdf = "✅ Activée" if PDF_OK else "❌ Désactivée"
            st.write(f"**PDF Export:** {status_pdf}")
        with col2:
            status_audio = "✅ Activée" if AUDIO_OK else "❌ Désactivée"
            st.write(f"**Audio/Whisper:** {status_audio}")
        with col3:
            status_modules = "✅ Activés" if MODULES_OK else "❌ Désactivés"
            st.write(f"**Modules avancés:** {status_modules}")

    # KPIs en haut - 4 colonnes
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    total_cas = len(st.session_state.historique)
    esi_critiques = sum(1 for h in st.session_state.historique if h['esi'] <= 2)
    confiance_moy = sum(h['confiance'] for h in st.session_state.historique) / total_cas if total_cas > 0 else 0

    with kpi1:
        st.markdown(f"""
        <div class="kpi-card" style="--kpi-color-start: #667eea; --kpi-color-end: #764ba2;">
            <div class="kpi-icon">📊</div>
            <div class="kpi-value">{total_cas}</div>
            <div class="kpi-label">{t['total_cases']}</div>
            <div class="kpi-trend">{t['session_running']}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        pct_critique = (esi_critiques / total_cas * 100) if total_cas > 0 else 0
        st.markdown(f"""
        <div class="kpi-card" style="--kpi-color-start: #f093fb; --kpi-color-end: #f5576c;">
            <div class="kpi-icon">🚨</div>
            <div class="kpi-value">{esi_critiques}</div>
            <div class="kpi-label">{t['critical_cases']}</div>
            <div class="kpi-trend">{pct_critique:.0f}% {t['of_total']}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="kpi-card" style="--kpi-color-start: #4facfe; --kpi-color-end: #00f2fe;">
            <div class="kpi-icon">🎯</div>
            <div class="kpi-value">{confiance_moy:.0f}%</div>
            <div class="kpi-label">{t['avg_confidence']}</div>
            <div class="kpi-trend">{t['high_reliability']}</div>
        </div>
        """, unsafe_allow_html=True)

    esi_counts = {}
    for i in range(1, 6):
        esi_counts[i] = sum(1 for h in st.session_state.historique if h['esi'] == i)

    with kpi4:
        most_common_esi = max(esi_counts, key=esi_counts.get) if total_cas > 0 else '-'
        most_common_count = esi_counts[most_common_esi] if total_cas > 0 else 0
        esi_icons = {1: "🔴", 2: "🟠", 3: "🟡", 4: "🟢", 5: "🔵", '-': "➖"}
        st.markdown(f"""
        <div class="kpi-card" style="--kpi-color-start: #fa709a; --kpi-color-end: #fee140;">
            <div class="kpi-icon">{esi_icons.get(most_common_esi, "📈")}</div>
            <div class="kpi-value">ESI-{most_common_esi}</div>
            <div class="kpi-label">{t['most_frequent']}</div>
            <div class="kpi-trend">{most_common_count} {t['cases']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Layout principal - 2 colonnes
    col_main, col_side = st.columns([2, 1])

    with col_side:
        # Historique dans la colonne de droite
        st.markdown(f"### 📋 {t['history']}")

        st.markdown('<div class="content-card">', unsafe_allow_html=True)

        if st.session_state.historique:
            for idx, h in enumerate(reversed(st.session_state.historique[-10:])):
                esi_colors = {1: "#dc2626", 2: "#ea580c", 3: "#f59e0b", 4: "#10b981", 5: "#3b82f6"}
                esi_icons = {1: "🔴", 2: "🟠", 3: "🟡", 4: "🟢", 5: "🔵"}

                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 100%);
                     padding: 0.75rem; border-radius: 10px; margin-bottom: 0.75rem;
                     border-left: 4px solid {esi_colors[h['esi']]};
                     box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                     transition: transform 0.2s ease;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="font-size: 0.7rem; font-weight: 600; color: #9ca3af;
                             background: #f3f4f6; padding: 0.2rem 0.5rem; border-radius: 4px;">
                            #{len(st.session_state.historique) - idx}
                        </div>
                        <div style="font-size: 0.7rem; color: #6b7280; font-weight: 500;">
                            🕐 {h['timestamp']}
                        </div>
                    </div>
                    <div style="margin-top: 0.5rem; font-weight: 700; font-size: 1.1rem; color: {esi_colors[h['esi']]};">
                        {esi_icons[h['esi']]} ESI-{h['esi']}
                    </div>
                    <div style="font-size: 0.8rem; color: #374151; margin-top: 0.25rem;">
                        {t['confidence_text']}: <span style="font-weight: 600; color: {esi_colors[h['esi']]};">{h['confiance']:.0f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if len(st.session_state.historique) > 10:
                st.caption(f"📌 {t['showing']} {len(st.session_state.historique)}")
        else:
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem; color: white; opacity: 0.7;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">📊</div>
                <div>{t['no_cases']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Distribution ESI avec graphique circulaire amélioré
        if total_cas > 0:
            st.markdown(f"### 📊 {t['distribution']}")

            st.markdown('<div class="content-card">', unsafe_allow_html=True)

            for esi in range(1, 6):
                count = esi_counts[esi]
                pct = (count / total_cas * 100) if total_cas > 0 else 0
                esi_colors = {1: "#dc2626", 2: "#ea580c", 3: "#f59e0b", 4: "#10b981", 5: "#3b82f6"}
                esi_icons = {1: "🔴", 2: "🟠", 3: "🟡", 4: "🟢", 5: "🔵"}
                color = esi_colors[esi]
                color_light = {1: "#fca5a5", 2: "#fb923c", 3: "#fbbf24", 4: "#6ee7b7", 5: "#93c5fd"}[esi]

                st.markdown(f"""
                <div style="margin-bottom: 1rem; padding: 0.75rem; background: rgba(255,255,255,0.5);
                     border-radius: 10px; backdrop-filter: blur(5px);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <span style="font-size: 1.2rem;">{esi_icons[esi]}</span>
                            <span style="font-weight: 700; font-size: 0.95rem;">ESI-{esi}</span>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-weight: 700; font-size: 1.1rem; color: {color};">{count}</div>
                            <div style="font-size: 0.7rem; color: #6b7280;">{pct:.0f}%</div>
                        </div>
                    </div>
                    <div class="probability-bar">
                        <div class="probability-fill" style="--bar-color: {color}; --bar-color-light: {color_light}; width: {pct}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    # Tabs dans la colonne principale
    with col_main:
        # Tabs
        if AUDIO_OK:
            tab1, tab2 = st.tabs([t['tab_text'], t['tab_audio']])
        else:
            tab1 = st.container()

    # TAB TEXTE - dans col_main
    with (tab1 if AUDIO_OK else tab1):
        st.markdown(f"#### 📝 {t['patient_desc']}")

        texte_patient = st.text_area(
            "",
            height=120,
            placeholder=t['placeholder'],
            label_visibility="collapsed"
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            bouton_analyser = st.button(f"🔍 {t['analyze']}", type="primary", use_container_width=True)
        with col2:
            if st.button(f"🗑️ {t['reset']}", use_container_width=True):
                st.session_state.historique = []
                st.rerun()

        if bouton_analyser and texte_patient.strip():
            with st.spinner(t['analyzing']):
                resultats = analyser_patient(texte_patient, model, tokenizer)

            # Ajouter à l'historique
            st.session_state.historique.append({
                'esi': resultats['esi'],
                'confiance': resultats['confiance'],
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'description_courte': texte_patient[:50] + "..." if len(texte_patient) > 50 else texte_patient
            })

            st.markdown("---")

            # Résultat ESI
            afficher_resultat_esi(resultats['esi'], resultats['confiance'], t)

            # Affichage du post-processing si ajustement effectué
            if resultats.get('post_processing_reason'):
                st.info(f"ℹ️ **Ajustement contextuel:** {resultats['post_processing_reason']} (ESI initial: {resultats['esi_initial']})")

            # Dashboard des probabilités
            st.markdown("---")
            probs = resultats.get('probabilites_detaillees', {})
            afficher_dashboard_probabilites(probs, resultats['esi'], t)

            # Bouton téléchargement PDF
            if PDF_OK:
                col_pdf1, col_pdf2, col_pdf3 = st.columns([1, 2, 1])
                with col_pdf2:
                    pdf_buffer = generer_pdf_rapport(texte_patient, resultats)
                    if pdf_buffer:
                        now = datetime.now()
                        filename = f"Rapport_Triage_ESI_{now.strftime('%Y%m%d_%H%M%S')}.pdf"
                        st.download_button(
                            label="📄 Télécharger le Rapport PDF",
                            data=pdf_buffer,
                            file_name=filename,
                            mime="application/pdf",
                            use_container_width=True,
                            type="secondary"
                        )
            else:
                st.warning("⚠️ Module PDF non disponible. Installez reportlab: pip install reportlab")

            st.markdown("---")

            # Alertes (avec traduction automatique)
            if resultats.get('alertes'):
                alertes = resultats['alertes']
                if alertes['total'] > 0:
                    titre_alertes = tr("Signaux d'Alerte")
                    st.markdown(f'<div style="display: flex; align-items: center; gap: 0.8rem; margin-top: 2rem;"><span style="font-size: 2rem;">⚠️</span><h3 style="margin: 0;">{titre_alertes}</h3></div>', unsafe_allow_html=True)
                    if alertes['risk_level'] == 'CRITICAL':
                        st.error(f"🚨 **{tr(str(alertes['total']) + ' signal(aux) CRITIQUE(S)')}**")
                    else:
                        msg_alerte = tr(str(alertes['total']) + " signal(aux) d'alerte")
                        st.warning(f"⚠️ {msg_alerte}")

                    # Liste flags (traduite)
                    if resultats.get('flags_liste'):
                        for flag in resultats['flags_liste'][:5]:
                            st.markdown(f"- **{tr(flag.description)}**")
                else:
                    msg_ok = tr("Aucun signal d'alerte critique")
                    st.success(f"✅ {msg_ok}")

            # Résumé clinique (avec traduction automatique)
            if resultats.get('resume'):
                st.markdown("---")
                st.markdown(f'<div style="display: flex; align-items: center; gap: 0.8rem;"><span style="font-size: 2rem;">📋</span><h3 style="margin: 0;">{tr("Résumé Clinique")}</h3></div>', unsafe_allow_html=True)
                resume = resultats['resume']

                col1, col2 = st.columns(2)
                with col1:
                    if resume.get('chief_complaint'):
                        st.markdown(f"**{tr('Motif')}:** {tr(resume['chief_complaint'])}")
                    if resume.get('symptoms_list'):
                        symptoms_tr = ', '.join([tr(s) for s in resume['symptoms_list'][:4]])
                        st.markdown(f"**{tr('Symptômes')}:** {symptoms_tr}")
                with col2:
                    if resume.get('severity_level'):
                        st.markdown(f"**{tr('Sévérité')}:** {tr(resume['severity_level'])}")
                    if resume.get('temporal_onset'):
                        st.markdown(f"**{tr('Début')}:** {tr(resume['temporal_onset'])}")

            # Examens recommandés (avec traduction automatique)
            if resultats.get('examens'):
                st.markdown("---")
                st.markdown(f'<div style="display: flex; align-items: center; gap: 0.8rem;"><span style="font-size: 2rem;">🔬</span><h3 style="margin: 0;">{tr("Examens Recommandés")}</h3></div>', unsafe_allow_html=True)

                examens = resultats['examens']
                stat = [e for e in examens if e.priority == 'STAT']
                urgent = [e for e in examens if e.priority == 'URGENT']

                if stat:
                    st.markdown(f"**🔴 {tr('IMMÉDIAT')}:**")
                    for e in stat:
                        st.markdown(f"- {tr(e.exam)}")

                if urgent:
                    st.markdown(f"**🟠 {tr('URGENT')}:**")
                    for e in urgent:
                        st.markdown(f"- {tr(e.exam)}")

            # Raisonnement (optionnel) - avec traduction automatique
            if resultats.get('raisonnement'):
                st.markdown("---")
                with st.expander(f"💡 **{tr('Raisonnement Clinique IA')}**", expanded=False):
                    raisonnement = resultats['raisonnement']

                    # PREDICTED ESI LEVEL
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #a8dadc 0%, #457b9d 100%);
                                padding: 1.2rem;
                                border-radius: 10px;
                                margin-bottom: 1rem;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <h4 style="margin: 0 0 0.5rem 0; color: white;">🎯 {tr('PREDICTED ESI LEVEL')}</h4>
                        <p style="margin: 0; color: white; font-size: 1.1rem;">
                            <strong>ESI-{raisonnement.predicted_esi}</strong> ({tr('Confiance')}: {raisonnement.confidence:.1%})
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # PRIMARY JUSTIFICATION (Pattern clinique)
                    if raisonnement.clinical_pattern:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #ffd6a5 0%, #fdab80 100%);
                                    padding: 1.2rem;
                                    border-radius: 10px;
                                    margin-bottom: 1rem;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <h4 style="margin: 0 0 0.5rem 0; color: #1f2937;">📌 {tr('PRIMARY JUSTIFICATION')}</h4>
                            <p style="margin: 0; color: #1f2937; font-size: 1rem;">
                                {tr(raisonnement.clinical_pattern)}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    # CRITICAL RED FLAGS
                    if raisonnement.red_flags:
                        flags_html = "<br>".join([f"• {tr(flag)}" for flag in raisonnement.red_flags])
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #ffadad 0%, #ff6b6b 100%);
                                    padding: 1.2rem;
                                    border-radius: 10px;
                                    margin-bottom: 1rem;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <h4 style="margin: 0 0 0.5rem 0; color: white;">⚠️ {tr('CRITICAL RED FLAGS')}</h4>
                            <p style="margin: 0; color: white; font-size: 1rem; line-height: 1.8;">
                                {flags_html}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    # SUPPORTING EVIDENCE
                    if raisonnement.supporting_evidence:
                        evidence_html = "<br>".join([f"• {tr(ev)}" for ev in raisonnement.supporting_evidence])
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #caffbf 0%, #9bf6ff 100%);
                                    padding: 1.2rem;
                                    border-radius: 10px;
                                    margin-bottom: 1rem;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <h4 style="margin: 0 0 0.5rem 0; color: #1f2937;">📋 {tr('SUPPORTING EVIDENCE')}</h4>
                            <p style="margin: 0; color: #1f2937; font-size: 1rem; line-height: 1.8;">
                                {evidence_html}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    # KEY INDICATORS
                    if raisonnement.key_indicators:
                        indicators_html = "<br>".join([f"• {tr(ind)}" for ind in raisonnement.key_indicators])
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #dda1f9 0%, #c084fc 100%);
                                    padding: 1.2rem;
                                    border-radius: 10px;
                                    margin-bottom: 1rem;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <h4 style="margin: 0 0 0.5rem 0; color: white;">🔑 {tr('KEY INDICATORS')}</h4>
                            <p style="margin: 0; color: white; font-size: 1rem; line-height: 1.8;">
                                {indicators_html}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    # CLINICAL RATIONALE (extrait du reasoning_text)
                    reasoning_text = raisonnement.reasoning_text
                    if "💭 CLINICAL RATIONALE:" in reasoning_text:
                        rationale_section = reasoning_text.split("💭 CLINICAL RATIONALE:")[1].split("\n\n")[0].strip()
                        rationale_text = rationale_section.replace("   ", "").strip()
                        if rationale_text:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
                                        padding: 1.2rem;
                                        border-radius: 10px;
                                        margin-bottom: 1rem;
                                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                <h4 style="margin: 0 0 0.5rem 0; color: #1f2937;">💭 CLINICAL RATIONALE</h4>
                                <p style="margin: 0; color: #1f2937; font-size: 1rem; line-height: 1.6;">
                                    {rationale_text}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                    # DIFFERENTIAL CONSIDERATION
                    if "🤔 DIFFERENTIAL CONSIDERATION:" in reasoning_text:
                        diff_section = reasoning_text.split("🤔 DIFFERENTIAL CONSIDERATION:")[1].split("\n\n")[0].strip()
                        diff_text = diff_section.replace("   ", "").strip()
                        if diff_text:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
                                        padding: 1.2rem;
                                        border-radius: 10px;
                                        margin-bottom: 1rem;
                                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                <h4 style="margin: 0 0 0.5rem 0; color: #1f2937;">🤔 DIFFERENTIAL CONSIDERATION</h4>
                                <p style="margin: 0; color: #1f2937; font-size: 1rem; line-height: 1.6;">
                                    {diff_text}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                    # TIME-SENSITIVE CONDITION
                    if "⏱️  TIME-SENSITIVE CONDITION:" in reasoning_text and raisonnement.predicted_esi <= 2:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #ff7675 0%, #d63031 100%);
                                    padding: 1.2rem;
                                    border-radius: 10px;
                                    margin-bottom: 0.5rem;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                    border: 2px solid #c0392b;">
                            <h4 style="margin: 0 0 0.5rem 0; color: white;">⏱️ TIME-SENSITIVE CONDITION</h4>
                            <p style="margin: 0; color: white; font-size: 1.1rem; font-weight: bold;">
                                ⚡ Immediate evaluation required. Delay may result in adverse outcomes.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

        elif bouton_analyser:
            st.warning("⚠️ Veuillez entrer une description")

    # TAB AUDIO
    if AUDIO_OK:
        with tab2:
            st.markdown('<div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;"><span style="font-size: 2.5rem;">⚕️</span><h2 style="margin: 0; color: #1f2937;">Enregistrement Audio</h2></div>', unsafe_allow_html=True)

            # Initialiser le processeur audio avec message de chargement
            if 'audio_proc' not in st.session_state:
                with st.spinner("🔄 Chargement du modèle Whisper (première fois uniquement, peut prendre 30-60 secondes)..."):
                    try:
                        # Forcer l'utilisation du CPU pour éviter les problèmes CUDA
                        st.session_state.audio_proc = AudioProcessor(
                            model_size="base",
                            device="cpu",
                            compute_type="int8"
                        )
                        st.success("✅ Modèle Whisper chargé avec succès (CPU)!")
                    except Exception as e:
                        st.error(f"❌ Erreur lors du chargement du modèle Whisper: {e}")
                        st.stop()

            # Instructions améliorées dans un container moderne
            st.markdown("""
            <div class="audio-container">
                <div style="text-align: center; margin-bottom: 1.5rem;">
                    <h3 style="color: #1f2937; margin-bottom: 0.5rem;">🎙️ Enregistrement Vocal</h3>
                    <p style="color: #6b7280; font-size: 1rem;">Décrivez l'état du patient de manière claire et concise</p>
                </div>
                <div style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); padding: 1rem; border-radius: 8px; border-left: 4px solid #3b82f6; margin-bottom: 1rem;">
                    <strong style="color: #1e3a8a;">📌 Instructions:</strong><br>
                    <span style="color: #1e40af;">• Cliquez sur le microphone pour démarrer l'enregistrement</span><br>
                    <span style="color: #1e40af;">• Cliquez à nouveau pour arrêter</span><br>
                    <span style="color: #1e40af;">• L'enregistrement sera automatiquement transcrit en français</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            audio_bytes = audio_recorder(
                text="",
                recording_color="#dc2626",
                neutral_color="#6366f1",
                icon_name="microphone",
                icon_size="6x",
                energy_threshold=(-1.0, 1.0),
                pause_threshold=300.0,
                sample_rate=16000
            )

            if audio_bytes:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #f0fdf4 0%, #d1fae5 100%);
                     padding: 1rem; border-radius: 8px; border-left: 4px solid #10b981;
                     margin: 1rem 0; text-align: center;">
                    <strong style="color: #14532d;">✅ Enregistrement réussi</strong>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### 🎧 Lecture de l'enregistrement")
                st.audio(audio_bytes, format="audio/wav")

                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
                    f.write(audio_bytes)
                    audio_path = f.name

                if st.button("🔄 Transcrire et Analyser", type="primary", use_container_width=True):
                    # Message de progression
                    progress_text = st.empty()
                    progress_text.info("🎧 Étape 1/3: Transcription audio avec Whisper...")

                    try:
                        # Détection automatique de langue + traduction vers français pour le modèle
                        result = st.session_state.audio_proc.transcribe_audio(
                            audio_path,
                            language=None,  # Détection automatique
                            auto_translate_to='fr'  # Traduction automatique vers français
                        )

                        if 'error' in result:
                            progress_text.empty()
                            st.error(f"❌ Erreur de transcription: {result.get('error', 'Erreur inconnue')}")
                            st.stop()

                        progress_text.info("🌐 Étape 2/3: Traduction automatique...")
                        import time
                        time.sleep(0.5)  # Petit délai pour que l'utilisateur voie le message

                        progress_text.empty()

                    except Exception as e:
                        progress_text.empty()
                        st.error(f"❌ Erreur lors de la transcription: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                        st.stop()

                    if result.get('text', '').strip():
                        # Afficher message de succès avec langue détectée
                        detected_lang = result.get('language', 'unknown')
                        was_translated = result.get('translation_applied', False)

                        if was_translated:
                            st.success(f"✅ Audio en {detected_lang.upper()} → traduit en français automatiquement")
                        else:
                            st.success(f"✅ Transcription terminée ({detected_lang.upper()})")

                        # Afficher les infos de debug
                        with st.expander("🔍 Détails de transcription", expanded=False):
                            st.write(f"**Langue détectée:** {result.get('language', 'N/A')} ({result.get('language_probability', 0):.2%})")
                            if was_translated:
                                st.write(f"**Texte original ({detected_lang}):** {result.get('text_original', 'N/A')}")
                                st.write(f"**Texte traduit (fr):** {result.get('text_translated', 'N/A')}")
                            st.write(f"**Nombre de segments:** {len(result.get('segments', []))}")
                            if result.get('segments'):
                                st.write("**Segments détaillés:**")
                                for i, seg in enumerate(result['segments'][:10], 1):
                                    st.write(f"{i}. [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")

                        transcription = st.text_area("Texte (modifiable):", result['text'], height=120)

                        with st.spinner("⏳ Analyse..."):
                            resultats = analyser_patient(transcription, model, tokenizer)

                        st.markdown("---")
                        afficher_resultat_esi(resultats['esi'], resultats['confiance'], t)

                        # Affichage du post-processing si ajustement effectué
                        if resultats.get('post_processing_reason'):
                            st.info(f"ℹ️ **Ajustement contextuel:** {resultats['post_processing_reason']} (ESI initial: {resultats['esi_initial']})")

                        # Dashboard des probabilités
                        st.markdown("---")
                        probs = resultats.get('probabilites_detaillees', {})
                        afficher_dashboard_probabilites(probs, resultats['esi'], t)

                        # Bouton téléchargement PDF
                        if PDF_OK:
                            col_pdf1, col_pdf2, col_pdf3 = st.columns([1, 2, 1])
                            with col_pdf2:
                                pdf_buffer = generer_pdf_rapport(transcription, resultats)
                                if pdf_buffer:
                                    now = datetime.now()
                                    filename = f"Rapport_Triage_ESI_{now.strftime('%Y%m%d_%H%M%S')}.pdf"
                                    st.download_button(
                                        label="📄 Télécharger le Rapport PDF",
                                        data=pdf_buffer,
                                        file_name=filename,
                                        mime="application/pdf",
                                        use_container_width=True,
                                        type="secondary"
                                    )
                        else:
                            st.warning("⚠️ Module PDF non disponible. Installez reportlab: pip install reportlab")

                        st.markdown("---")

                        # Alertes
                        if resultats.get('alertes'):
                            alertes = resultats['alertes']
                            if alertes['total'] > 0:
                                st.markdown('<div style="display: flex; align-items: center; gap: 0.8rem; margin-top: 2rem;"><span style="font-size: 2rem;">⚠️</span><h3 style="margin: 0;">Signaux d\'Alerte</h3></div>', unsafe_allow_html=True)
                                if alertes['risk_level'] == 'CRITICAL':
                                    st.error(f"🚨 **{alertes['total']} signal(aux) CRITIQUE(S)**")
                                else:
                                    st.warning(f"⚠️ {alertes['total']} signal(aux) d'alerte")

                                # Liste flags
                                if resultats.get('flags_liste'):
                                    for flag in resultats['flags_liste'][:5]:
                                        st.markdown(f"- **{flag.description}**")
                            else:
                                st.success("✅ Aucun signal d'alerte critique")

                        # Résumé clinique
                        if resultats.get('resume'):
                            st.markdown("---")
                            st.markdown('<div style="display: flex; align-items: center; gap: 0.8rem;"><span style="font-size: 2rem;">📋</span><h3 style="margin: 0;">Résumé Clinique</h3></div>', unsafe_allow_html=True)
                            resume = resultats['resume']

                            col1, col2 = st.columns(2)
                            with col1:
                                if resume.get('chief_complaint'):
                                    st.markdown(f"**Motif:** {resume['chief_complaint']}")
                                if resume.get('symptoms_list'):
                                    st.markdown(f"**Symptômes:** {', '.join(resume['symptoms_list'][:4])}")
                            with col2:
                                if resume.get('severity_level'):
                                    st.markdown(f"**Sévérité:** {resume['severity_level']}")
                                if resume.get('temporal_onset'):
                                    st.markdown(f"**Début:** {resume['temporal_onset']}")

                        # Examens recommandés
                        if resultats.get('examens'):
                            st.markdown("---")
                            st.markdown('<div style="display: flex; align-items: center; gap: 0.8rem;"><span style="font-size: 2rem;">🔬</span><h3 style="margin: 0;">Examens Recommandés</h3></div>', unsafe_allow_html=True)

                            examens = resultats['examens']
                            stat = [e for e in examens if e.priority == 'STAT']
                            urgent = [e for e in examens if e.priority == 'URGENT']

                            if stat:
                                st.markdown("**🔴 IMMÉDIAT:**")
                                for e in stat:
                                    st.markdown(f"- {e.exam}")

                            if urgent:
                                st.markdown("**🟠 URGENT:**")
                                for e in urgent:
                                    st.markdown(f"- {e.exam}")

                        # Raisonnement (optionnel)
                        if resultats.get('raisonnement'):
                            st.markdown("---")
                            with st.expander("💡 **Raisonnement Clinique IA**", expanded=False):
                                raisonnement = resultats['raisonnement']

                                # PREDICTED ESI LEVEL
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #a8dadc 0%, #457b9d 100%);
                                            padding: 1.2rem;
                                            border-radius: 10px;
                                            margin-bottom: 1rem;
                                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                    <h4 style="margin: 0 0 0.5rem 0; color: white;">🎯 PREDICTED ESI LEVEL</h4>
                                    <p style="margin: 0; color: white; font-size: 1.1rem;">
                                        <strong>ESI-{raisonnement.predicted_esi}</strong> (Confiance: {raisonnement.confidence:.1%})
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)

                                # PRIMARY JUSTIFICATION (Pattern clinique)
                                if raisonnement.clinical_pattern:
                                    st.markdown(f"""
                                    <div style="background: linear-gradient(135deg, #ffd6a5 0%, #fdab80 100%);
                                                padding: 1.2rem;
                                                border-radius: 10px;
                                                margin-bottom: 1rem;
                                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                        <h4 style="margin: 0 0 0.5rem 0; color: #1f2937;">📌 PRIMARY JUSTIFICATION</h4>
                                        <p style="margin: 0; color: #1f2937; font-size: 1rem;">
                                            {raisonnement.clinical_pattern}
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)

                                # CRITICAL RED FLAGS
                                if raisonnement.red_flags:
                                    flags_html = "<br>".join([f"• {flag}" for flag in raisonnement.red_flags])
                                    st.markdown(f"""
                                    <div style="background: linear-gradient(135deg, #ffadad 0%, #ff6b6b 100%);
                                                padding: 1.2rem;
                                                border-radius: 10px;
                                                margin-bottom: 1rem;
                                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                        <h4 style="margin: 0 0 0.5rem 0; color: white;">⚠️ CRITICAL RED FLAGS</h4>
                                        <p style="margin: 0; color: white; font-size: 1rem; line-height: 1.8;">
                                            {flags_html}
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)

                                # SUPPORTING EVIDENCE
                                if raisonnement.supporting_evidence:
                                    evidence_html = "<br>".join([f"• {ev}" for ev in raisonnement.supporting_evidence])
                                    st.markdown(f"""
                                    <div style="background: linear-gradient(135deg, #caffbf 0%, #9bf6ff 100%);
                                                padding: 1.2rem;
                                                border-radius: 10px;
                                                margin-bottom: 1rem;
                                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                        <h4 style="margin: 0 0 0.5rem 0; color: #1f2937;">📋 SUPPORTING EVIDENCE</h4>
                                        <p style="margin: 0; color: #1f2937; font-size: 1rem; line-height: 1.8;">
                                            {evidence_html}
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)

                                # KEY INDICATORS
                                if raisonnement.key_indicators:
                                    indicators_html = "<br>".join([f"• {ind}" for ind in raisonnement.key_indicators])
                                    st.markdown(f"""
                                    <div style="background: linear-gradient(135deg, #dda1f9 0%, #c084fc 100%);
                                                padding: 1.2rem;
                                                border-radius: 10px;
                                                margin-bottom: 1rem;
                                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                        <h4 style="margin: 0 0 0.5rem 0; color: white;">🔑 KEY INDICATORS</h4>
                                        <p style="margin: 0; color: white; font-size: 1rem; line-height: 1.8;">
                                            {indicators_html}
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)

                                # CLINICAL RATIONALE (extrait du reasoning_text)
                                reasoning_text = raisonnement.reasoning_text
                                if "💭 CLINICAL RATIONALE:" in reasoning_text:
                                    rationale_section = reasoning_text.split("💭 CLINICAL RATIONALE:")[1].split("\n\n")[0].strip()
                                    rationale_text = rationale_section.replace("   ", "").strip()
                                    if rationale_text:
                                        st.markdown(f"""
                                        <div style="background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
                                                    padding: 1.2rem;
                                                    border-radius: 10px;
                                                    margin-bottom: 1rem;
                                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                            <h4 style="margin: 0 0 0.5rem 0; color: #1f2937;">💭 CLINICAL RATIONALE</h4>
                                            <p style="margin: 0; color: #1f2937; font-size: 1rem; line-height: 1.6;">
                                                {rationale_text}
                                            </p>
                                        </div>
                                        """, unsafe_allow_html=True)

                                # DIFFERENTIAL CONSIDERATION
                                if "🤔 DIFFERENTIAL CONSIDERATION:" in reasoning_text:
                                    diff_section = reasoning_text.split("🤔 DIFFERENTIAL CONSIDERATION:")[1].split("\n\n")[0].strip()
                                    diff_text = diff_section.replace("   ", "").strip()
                                    if diff_text:
                                        st.markdown(f"""
                                        <div style="background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
                                                    padding: 1.2rem;
                                                    border-radius: 10px;
                                                    margin-bottom: 1rem;
                                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                            <h4 style="margin: 0 0 0.5rem 0; color: #1f2937;">🤔 DIFFERENTIAL CONSIDERATION</h4>
                                            <p style="margin: 0; color: #1f2937; font-size: 1rem; line-height: 1.6;">
                                                {diff_text}
                                            </p>
                                        </div>
                                        """, unsafe_allow_html=True)

                                # TIME-SENSITIVE CONDITION
                                if "⏱️  TIME-SENSITIVE CONDITION:" in reasoning_text and raisonnement.predicted_esi <= 2:
                                    st.markdown(f"""
                                    <div style="background: linear-gradient(135deg, #ff7675 0%, #d63031 100%);
                                                padding: 1.2rem;
                                                border-radius: 10px;
                                                margin-bottom: 0.5rem;
                                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                                border: 2px solid #c0392b;">
                                        <h4 style="margin: 0 0 0.5rem 0; color: white;">⏱️ TIME-SENSITIVE CONDITION</h4>
                                        <p style="margin: 0; color: white; font-size: 1.1rem; font-weight: bold;">
                                            ⚡ Immediate evaluation required. Delay may result in adverse outcomes.
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                    else:
                        st.error("❌ Aucun texte détecté dans l'enregistrement")
                        st.warning("⚠️ Vérifiez que l'enregistrement contient de la parole claire. Réessayez avec un enregistrement plus long ou plus clair.")

                try:
                    os.unlink(audio_path)
                except:
                    pass

    # Footer compact
    st.markdown("---")
    st.caption("⚕️ Projet Académique - Le jugement clinique reste primordial | Version 2.0")

if __name__ == "__main__":
    main()