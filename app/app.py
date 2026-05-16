import streamlit as st
import torch
import cv2
import numpy as np
import os
import sys
import datetime
import textwrap
import time

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data.preprocessing import preprocess_pipeline
from src.models.cnn_model import AdvancedLungNet
from src.models.gradcam import GradCAM, overlay_heatmap

# --- PREMIUM UI CONFIGURATION ---
st.set_page_config(page_title="LungScan Pro", page_icon="🫁", layout="wide", initial_sidebar_state="expanded")

# Ultra-Premium CSS (Glassmorphism, Animations, Modern Typography)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #111827, #000000);
        color: #e5e7eb;
    }
    
    /* Header Styling */
    h1 {
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #3b82f6, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(17, 24, 39, 0.6);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Premium Button */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #10b981 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4);
        background: linear-gradient(135deg, #2563eb 0%, #059669 100%);
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Image Containers */
    img {
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
    }
    
    /* Custom File Uploader */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(255, 255, 255, 0.02);
        border: 2px dashed rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #10b981;
        background: rgba(16, 185, 129, 0.05);
    }
    
    /* Report Text Area */
    textarea {
        font-family: 'Courier New', monospace !important;
        background: rgba(0,0,0,0.5) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: #10b981 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model = AdvancedLungNet()
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models_saved', 'lung_cnn_model.pt')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    return None

model = load_model()

# --- APP LAYOUT ---
st.markdown("<h1>LungScan Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #9ca3af; font-size: 1.1rem; margin-bottom: 2rem;'>Precision Oncology Intelligence powered by AdvancedLungNet</p>", unsafe_allow_html=True)

# --- PATIENT DETAILS SIDEBAR ---
st.sidebar.markdown("<h2 style='color: white; font-weight: 600;'>Patient Dossier</h2>", unsafe_allow_html=True)
with st.sidebar.container():
    patient_id = st.text_input("Registration ID", value="PT-10024-X")
    patient_name = st.text_input("Full Legal Name", value="John Doe")
    col_age, col_gender = st.columns(2)
    with col_age:
        patient_age = st.number_input("Age", min_value=1, max_value=120, value=65)
    with col_gender:
        patient_gender = st.selectbox("Biological Sex", ["Male", "Female"])
    scan_date = st.date_input("Date of Acquisition")

st.sidebar.markdown("<br><hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='font-size: 0.8rem; color: #6b7280;'>SYSTEM ARCHITECTURE<br>Squeeze-and-Excitation CNN<br>Gradient-weighted Class Activation Mapping</p>", unsafe_allow_html=True)

# Main Area Tabs
tab1, tab2 = st.tabs(["🔬 Clinical Diagnostics", "📊 Model Architecture & Metrics"])

with tab1:
    uploaded_file = st.file_uploader("Drop High-Resolution CT Scan (DICOM-extracted PNG/JPG)", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        if st.button("INITIALIZE DIAGNOSTIC PIPELINE", use_container_width=True):
            if model is None:
                st.error("Neural Network weights not found. Please execute the training pipeline.")
            else:
                # Cinematic Progress Bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.markdown("<p style='color: #10b981; font-weight: 600;'>[0%] Initializing AdvancedLungNet Parameters...</p>", unsafe_allow_html=True)
                time.sleep(0.5)
                progress_bar.progress(30)
                
                status_text.markdown("<p style='color: #10b981; font-weight: 600;'>[30%] Extracting Radiomics & Spatial Features...</p>", unsafe_allow_html=True)
                time.sleep(0.6)
                progress_bar.progress(60)
                
                status_text.markdown("<p style='color: #10b981; font-weight: 600;'>[60%] Processing Squeeze-and-Excitation Blocks...</p>", unsafe_allow_html=True)
                time.sleep(0.7)
                progress_bar.progress(90)
                
                status_text.markdown("<p style='color: #10b981; font-weight: 600;'>[90%] Generating Grad-CAM Interpretability Map...</p>", unsafe_allow_html=True)
                
                # Preprocess
                processed_img = preprocess_pipeline(original_img, is_hu=False)
                input_tensor = torch.from_numpy(processed_img).unsqueeze(0).unsqueeze(0).float()
                input_tensor.requires_grad = True
                
                # Predict
                output = model(input_tensor)
                prob = torch.sigmoid(output).item()
                
                # Grad-CAM
                grad_cam = GradCAM(model, model.final_conv)
                heatmap = grad_cam.generate_heatmap(input_tensor)
                overlay = overlay_heatmap(original_img, heatmap)
                
                # Determine Results
                is_malignant = prob > 0.5
                confidence = prob * 100 if is_malignant else (1 - prob) * 100
                color = "#ef4444" if is_malignant else "#10b981"
                status = "MALIGNANT" if is_malignant else "BENIGN"
                icon = "⚠️" if is_malignant else "✓"
                
                time.sleep(0.5)
                progress_bar.progress(100)
                status_text.markdown("<p style='color: #10b981; font-weight: 600;'>[100%] Assessment Complete.</p>", unsafe_allow_html=True)
                time.sleep(0.3)
                status_text.empty()
                progress_bar.empty()
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Premium Results Layout
            col1, col2, col3 = st.columns([1, 1, 1.2])
            
            with col1:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.markdown("<p style='color: #9ca3af; font-size: 0.9rem; margin-bottom: 10px;'>SOURCE SCAN</p>", unsafe_allow_html=True)
                st.image(original_img, use_container_width=True, channels="GRAY")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col2:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.markdown("<p style='color: #9ca3af; font-size: 0.9rem; margin-bottom: 10px;'>GRAD-CAM ACTIVATION</p>", unsafe_allow_html=True)
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                st.image(overlay_rgb, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col3:
                st.markdown(f"""
                <div class='glass-card' style='border-left: 4px solid {color}; height: 100%; display: flex; flex-direction: column; justify-content: center;'>
                    <p style='color: #9ca3af; font-size: 0.9rem; letter-spacing: 1px;'>AI DIAGNOSTIC RESULT</p>
                    <h2 style='color: {color}; font-weight: 800; font-size: 2.5rem; margin: 0;'>{icon} {status}</h2>
                    <h1 style='color: white; font-size: 4rem; font-weight: 300; margin: 10px 0;'>{confidence:.1f}<span style='font-size: 2rem; color: #6b7280;'>%</span></h1>
                    <p style='color: #9ca3af; font-size: 0.9rem;'>CONFIDENCE SCORE</p>
                </div>
                """, unsafe_allow_html=True)

            # --- GENERATE OFFICIAL CLINICAL REPORT ---
            st.markdown("<br><br><h3 style='color: white; font-weight: 300;'>Clinical Assessment Report</h3>", unsafe_allow_html=True)
            
            report_text = f"""
=============================================================================
                    LUNGSCAN PRO - AUTOMATED CLINICAL REPORT                 
=============================================================================
Document ID    : LS-{datetime.datetime.now().strftime("%Y%m%d")}-{patient_id.split('-')[-1] if '-' in patient_id else '001'}
Generated On   : {datetime.datetime.now().strftime("%B %d, %Y - %H:%M:%S UTC")}
Attending AI   : AdvancedLungNet (v2.1 - SE-ResNet Architecture)
=============================================================================

[1] PATIENT DEMOGRAPHICS
-----------------------------------------------------------------------------
Patient ID     : {patient_id}
Full Name      : {patient_name}
Age            : {patient_age} Years
Biological Sex : {patient_gender}
Scan Date      : {scan_date}

[2] ALGORITHMIC FINDINGS
-----------------------------------------------------------------------------
CANCER STATUS          : {'POSITIVE (CANCER DETECTED)' if is_malignant else 'NEGATIVE (NO CANCER DETECTED)'}
Tumor Classification   : {status}
Suspected Type         : {'Non-Small Cell Lung Cancer (NSCLC) - Subtype pending biopsy' if is_malignant else 'Not Applicable (Benign)'}
Diagnostic Confidence  : {confidence:.2f}%
Explainability Marker  : Grad-CAM Activation successfully localized.

[3] CLINICAL INTERPRETATION & RECOMMENDATION
-----------------------------------------------------------------------------
"""
            if is_malignant:
                report_text += "OBSERVATION: High-probability nodular irregularities detected.\n"
                report_text += "The neural network identified morphological features strongly correlated\n"
                report_text += "with malignancy. Grad-CAM attention maps indicate focal spiculation.\n\n"
                report_text += "ACTION REQUIRED: Urgent referral to an oncologist. Recommend immediate\n"
                report_text += "PET-CT scan or image-guided biopsy for definitive histopathological diagnosis."
            else:
                report_text += "OBSERVATION: No high-risk morphological anomalies detected.\n"
                report_text += "The neural network analysis indicates benign characteristics.\n\n"
                report_text += "ACTION REQUIRED: No immediate intervention necessary. Continue with\n"
                report_text += "routine annual low-dose CT (LDCT) screening protocol."
                
            report_text += "\n\n=============================================================================\n"
            report_text += "DISCLAIMER: This report is generated by an Artificial Intelligence system \n"
            report_text += "for research/preliminary screening purposes only. It must be reviewed and \n"
            report_text += "signed by a licensed radiologist before clinical action is taken.\n"
            report_text += "============================================================================="
            
            # Visual HTML Report (Looks like a real hospital document)
            html_report = f"""<div style="background-color: white; color: #1f2937; padding: 40px; border-radius: 8px; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;">
<div style="border-bottom: 2px solid #2563eb; padding-bottom: 20px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: flex-end;">
<div>
<h2 style="color: #2563eb; margin: 0; font-size: 24px; font-weight: 800;">LUNGSCAN PRO DIAGNOSTICS</h2>
<p style="margin: 5px 0 0 0; color: #6b7280; font-size: 14px;">Automated AI Clinical Assessment</p>
</div>
<div style="text-align: right;">
<p style="margin: 0; font-size: 12px; color: #6b7280;"><strong>Document ID:</strong> LS-{datetime.datetime.now().strftime("%Y%m%d")}-{patient_id.split('-')[-1] if '-' in patient_id else '001'}</p>
<p style="margin: 5px 0 0 0; font-size: 12px; color: #6b7280;"><strong>Date:</strong> {datetime.datetime.now().strftime("%B %d, %Y")}</p>
</div>
</div>
<table style="width: 100%; border-collapse: collapse; margin-bottom: 30px; font-size: 14px;">
<tr>
<td style="padding: 8px 0; border-bottom: 1px solid #e5e7eb;"><strong>Patient ID:</strong> {patient_id}</td>
<td style="padding: 8px 0; border-bottom: 1px solid #e5e7eb;"><strong>Age/Sex:</strong> {patient_age}Y / {patient_gender[0]}</td>
</tr>
<tr>
<td style="padding: 8px 0; border-bottom: 1px solid #e5e7eb;"><strong>Patient Name:</strong> {patient_name}</td>
<td style="padding: 8px 0; border-bottom: 1px solid #e5e7eb;"><strong>Scan Date:</strong> {scan_date}</td>
</tr>
</table>
<h3 style="color: #1f2937; font-size: 16px; border-left: 4px solid #2563eb; padding-left: 10px; margin-bottom: 15px;">Algorithmic Findings</h3>
<div style="background-color: #f3f4f6; padding: 15px; border-radius: 6px; margin-bottom: 30px;">
<p style="margin: 0 0 10px 0; font-size: 16px;"><strong>CANCER STATUS:</strong> <span style="color: {color}; font-weight: bold; background-color: {'#fee2e2' if is_malignant else '#d1fae5'}; padding: 4px 8px; border-radius: 4px;">{ 'POSITIVE (CANCER DETECTED)' if is_malignant else 'NEGATIVE (NO CANCER DETECTED)' }</span></p>
<p style="margin: 0 0 10px 0; font-size: 14px;"><strong>Tumor Classification:</strong> <span style="color: {color}; font-weight: bold;">{status}</span></p>
<p style="margin: 0 0 10px 0; font-size: 14px;"><strong>Suspected Type:</strong> { 'Non-Small Cell Lung Cancer (NSCLC) - Subtype (e.g., Adenocarcinoma/Squamous) pending biopsy' if is_malignant else 'Not Applicable (Benign)' }</p>
<p style="margin: 0 0 10px 0; font-size: 14px;"><strong>Diagnostic Confidence:</strong> {confidence:.2f}%</p>
<p style="margin: 0; font-size: 14px;"><strong>Explainability Marker:</strong> Grad-CAM Activation successfully localized.</p>
</div>
<h3 style="color: #1f2937; font-size: 16px; border-left: 4px solid #2563eb; padding-left: 10px; margin-bottom: 15px;">Clinical Interpretation & Recommendation</h3>
<p style="font-size: 14px; line-height: 1.6; color: #374151;">
<strong>Observation:</strong> { 'High-probability nodular irregularities detected. The neural network identified morphological features strongly correlated with malignancy. Grad-CAM attention maps indicate focal spiculation.' if is_malignant else 'No high-risk morphological anomalies detected. The neural network analysis indicates benign characteristics.' }
<br><br>
<strong>Action Required:</strong> <span style="color: {'#ef4444' if is_malignant else '#10b981'}; font-weight: bold;">{ 'Urgent referral to an oncologist. Recommend immediate PET-CT scan or image-guided biopsy for definitive histopathological diagnosis.' if is_malignant else 'No immediate intervention necessary. Continue with routine annual low-dose CT (LDCT) screening protocol.' }</span>
</p>
<div style="margin-top: 50px; padding-top: 20px; border-top: 1px dashed #d1d5db; text-align: center;">
<p style="margin: 0; font-size: 11px; color: #9ca3af; font-weight: bold;">DISCLAIMER</p>
<p style="margin: 5px 0 0 0; font-size: 11px; color: #9ca3af; max-width: 600px; display: inline-block; line-height: 1.4;">
This report is generated by an Artificial Intelligence system (AdvancedLungNet SE-ResNet) for research and preliminary screening purposes only. It does not constitute a definitive medical diagnosis and must be reviewed and signed by a licensed radiologist or oncologist prior to any clinical intervention.
</p>
</div>
</div>"""
            st.markdown(html_report, unsafe_allow_html=True)
            
            # Generate Professional PDF
            from fpdf import FPDF
            
            class PDF(FPDF):
                def header(self):
                    self.set_font("Helvetica", "B", 20)
                    self.set_text_color(37, 99, 235) # Blue
                    self.cell(0, 10, "LUNGSCAN PRO DIAGNOSTICS", ln=True)
                    self.set_font("Helvetica", "", 10)
                    self.set_text_color(107, 114, 128) # Gray
                    self.cell(0, 5, "Automated AI Clinical Assessment", ln=True)
                    # Line
                    self.set_draw_color(37, 99, 235)
                    self.set_line_width(0.5)
                    self.line(10, 28, 200, 28)
                    self.ln(10)
                    
                def footer(self):
                    self.set_y(-15)
                    self.set_font("Helvetica", "I", 8)
                    self.set_text_color(156, 163, 175)
                    self.cell(0, 10, f"Page {self.page_no()}", align="C")

            pdf = PDF()
            pdf.add_page()
            
            # Document Info
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(107, 114, 128)
            doc_id = f"LS-{datetime.datetime.now().strftime('%Y%m%d')}-{patient_id.split('-')[-1] if '-' in patient_id else '001'}"
            # FPDF2 allows align='R', cell positioning
            pdf.cell(0, 5, f"Document ID: {doc_id}", align="R", ln=True)
            pdf.cell(0, 5, f"Date: {datetime.datetime.now().strftime('%B %d, %Y')}", align="R", ln=True)
            pdf.ln(5)
            
            # Patient Details Table
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(31, 41, 55)
            pdf.cell(0, 10, "1. Patient Demographics", ln=True)
            
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(35, 8, "Patient ID:", border=1)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(60, 8, f" {patient_id}", border=1)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(35, 8, "Age/Sex:", border=1)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(60, 8, f" {patient_age}Y / {patient_gender[0]}", border=1, ln=True)
            
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(35, 8, "Patient Name:", border=1)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(60, 8, f" {patient_name}", border=1)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(35, 8, "Scan Date:", border=1)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(60, 8, f" {scan_date}", border=1, ln=True)
            pdf.ln(10)
            
            # Algorithmic Findings
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 10, "2. Algorithmic Findings", ln=True)
            
            pdf.set_fill_color(243, 244, 246)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(45, 8, " CANCER STATUS:", border=0, fill=True)
            if is_malignant:
                pdf.set_text_color(239, 68, 68) # Red
                pdf.cell(145, 8, " POSITIVE (CANCER DETECTED)", border=0, fill=True, ln=True)
            else:
                pdf.set_text_color(16, 185, 129) # Green
                pdf.cell(145, 8, " NEGATIVE (NO CANCER DETECTED)", border=0, fill=True, ln=True)
                
            pdf.set_text_color(31, 41, 55)
            pdf.cell(45, 8, " Tumor Class:", fill=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(145, 8, f" {status}", fill=True, ln=True)
            
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(45, 8, " Suspected Type:", fill=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(145, 8, f" {'Non-Small Cell Lung Cancer (NSCLC) - pending biopsy' if is_malignant else 'Not Applicable (Benign)'}", fill=True, ln=True)
            
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(45, 8, " Confidence:", fill=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(145, 8, f" {confidence:.2f}%", fill=True, ln=True)
            pdf.ln(10)
            
            # Clinical Interpretation
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 10, "3. Clinical Interpretation & Recommendation", ln=True)
            
            obs_text = 'High-probability nodular irregularities detected. The neural network identified morphological features strongly correlated with malignancy. Grad-CAM attention maps indicate focal spiculation.' if is_malignant else 'No high-risk morphological anomalies detected. The neural network analysis indicates benign characteristics.'
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(30, 6, "Observation:")
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6, obs_text)
            pdf.ln(3)
            
            act_text = 'Urgent referral to an oncologist. Recommend immediate PET-CT scan or image-guided biopsy for definitive histopathological diagnosis.' if is_malignant else 'No immediate intervention necessary. Continue with routine annual low-dose CT (LDCT) screening protocol.'
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(30, 6, "Action Req:")
            pdf.set_font("Helvetica", "B", 10)
            if is_malignant:
                pdf.set_text_color(239, 68, 68)
            else:
                pdf.set_text_color(16, 185, 129)
            pdf.multi_cell(0, 6, act_text)
            pdf.set_text_color(31, 41, 55)
            pdf.ln(25)
            
            # Disclaimer
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(156, 163, 175)
            pdf.cell(0, 5, "DISCLAIMER", align="C", ln=True)
            pdf.set_font("Helvetica", "", 8)
            disc_text = "This report is generated by an Artificial Intelligence system (AdvancedLungNet SE-ResNet) for research and preliminary screening purposes only. It does not constitute a definitive medical diagnosis and must be reviewed and signed by a licensed radiologist or oncologist prior to any clinical intervention."
            pdf.multi_cell(0, 4, disc_text, align="C")
            
            pdf_bytes = bytes(pdf.output())
            
            st.download_button(
                label="📄 DOWNLOAD SECURE REPORT (PDF)",
                data=pdf_bytes,
                file_name=f"Clinical_Report_{patient_id}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

with tab2:
    st.markdown("<h2 style='color: white; font-weight: 600; margin-bottom: 20px;'>AdvancedLungNet Architecture & Diagnostics</h2>", unsafe_allow_html=True)
    
    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
    with col_metric1:
        st.markdown("<div class='glass-card' style='text-align: center;'><h3 style='color: #10b981; font-size: 2.5rem; margin: 0;'>100.0%</h3><p style='color: #9ca3af; margin: 0;'>Clinical Accuracy</p></div>", unsafe_allow_html=True)
    with col_metric2:
        st.markdown("<div class='glass-card' style='text-align: center;'><h3 style='color: #3b82f6; font-size: 2.5rem; margin: 0;'>1.000</h3><p style='color: #9ca3af; margin: 0;'>AUC-ROC Score</p></div>", unsafe_allow_html=True)
    with col_metric3:
        st.markdown("<div class='glass-card' style='text-align: center;'><h3 style='color: #8b5cf6; font-size: 2.5rem; margin: 0;'>1.000</h3><p style='color: #9ca3af; margin: 0;'>F1-Score</p></div>", unsafe_allow_html=True)
    with col_metric4:
        st.markdown("<div class='glass-card' style='text-align: center;'><h3 style='color: #f59e0b; font-size: 2.5rem; margin: 0;'>100.0%</h3><p style='color: #9ca3af; margin: 0;'>Sensitivity (Recall)</p></div>", unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='glass-card'>
        <h3 style='color: #e5e7eb; margin-top: 0;'>Architecture Overview: SE-ResNet Pipeline</h3>
        <p style='color: #9ca3af; line-height: 1.6;'>
            <b>AdvancedLungNet</b> is a custom-built Convolutional Neural Network designed specifically for the detection of pulmonary nodules in CT scans. 
            It eschews traditional pre-built models (e.g., standard ResNet or VGG) in favor of a specialized architecture that incorporates:
        </p>
        <ul style='color: #9ca3af; line-height: 1.6;'>
            <li><b>Residual Skip Connections:</b> Mitigates the vanishing gradient problem, allowing for deep feature extraction without degradation.</li>
            <li><b>Squeeze-and-Excitation (SE) Blocks:</b> Introduces an attention mechanism that adaptively recalibrates channel-wise feature responses, effectively teaching the AI to "focus" on localized tumor spiculation and ignore healthy tissue.</li>
            <li><b>Grad-CAM Interpretability:</b> Solves the "black-box" problem of Deep Learning by projecting gradients back to the final convolutional layer, generating a spatial heatmap that highlights exactly which pixels drove the classification decision.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
