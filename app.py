import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import matplotlib.pyplot as plt
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DeFacto Forensics",
    page_icon="🕵️‍♀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
    <style>
    * {
        margin: 0;
        padding: 0;
    }
    
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0d1b2a 100%);
        min-height: 100vh;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0d1b2a 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1419, #1a2332);
        border-right: 1px solid rgba(120, 200, 255, 0.2);
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #1e90ff, #00bfff);
        color: white;
        font-weight: 600;
        border-radius: 12px;
        border: none;
        padding: 14px 28px;
        font-size: 0.95rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(30, 144, 255, 0.35);
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #00bfff, #1e90ff);
        box-shadow: 0 8px 30px rgba(30, 144, 255, 0.5);
        transform: translateY(-2px);
    }
    
    h1 {
        color: #e0f7ff;
        text-align: center;
        font-family: 'Segoe UI', 'Trebuchet MS', sans-serif;
        font-size: 3.8rem;
        font-weight: 800;
        text-shadow: 0 2px 20px rgba(30, 144, 255, 0.3);
        margin-bottom: 15px;
        letter-spacing: 1px;
    }
    
    h2 {
        color: #00d9ff;
        font-size: 2rem;
        font-weight: 700;
        text-shadow: 0 2px 8px rgba(0, 217, 255, 0.2);
        border-bottom: 1px solid rgba(0, 217, 255, 0.2);
        padding-bottom: 12px;
    }
    
    h3 {
        color: #b0e0ff;
        font-size: 1.35rem;
        font-weight: 600;
        margin: 20px 0 15px 0;
    }
    
    .subtitle {
        color: #90c8ff;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 35px;
        font-weight: 300;
        letter-spacing: 0.8px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 144, 255, 0.08), rgba(0, 191, 255, 0.08));
        padding: 28px 24px;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(30, 144, 255, 0.12);
        text-align: center;
        border: 1px solid rgba(30, 144, 255, 0.2);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
    }
    
    .metric-card:hover {
        border-color: rgba(30, 144, 255, 0.4);
        box-shadow: 0 12px 48px rgba(30, 144, 255, 0.25);
        transform: translateY(-6px);
        background: linear-gradient(135deg, rgba(30, 144, 255, 0.15), rgba(0, 191, 255, 0.15));
    }
    
    .metric-value {
        font-size: 2.6rem;
        font-weight: 800;
        color: #00ff9f;
        text-shadow: 0 2px 12px rgba(0, 255, 159, 0.3);
        margin: 12px 0;
        letter-spacing: -1px;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #90c8ff;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }
    
    .probability-container {
        background: linear-gradient(135deg, rgba(30, 144, 255, 0.05), rgba(0, 191, 255, 0.05));
        padding: 22px;
        border-radius: 14px;
        border: 1px solid rgba(30, 144, 255, 0.15);
        backdrop-filter: blur(8px);
    }
    
    .upload-section {
        background: linear-gradient(135deg, rgba(30, 144, 255, 0.1), rgba(0, 191, 255, 0.1));
        padding: 35px;
        border-radius: 16px;
        border: 2px dashed rgba(30, 144, 255, 0.35);
        text-align: center;
        margin: 20px 0;
        backdrop-filter: blur(8px);
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(30, 144, 255, 0.08), rgba(0, 191, 255, 0.08));
        border-left: 3px solid #1e90ff;
        border-radius: 12px;
    }
    
    .info-text {
        color: #90c8ff;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(30, 144, 255, 0.2), transparent);
        margin: 35px 0;
    }
    
    .footer-text {
        text-align: center;
        color: #90c8ff;
        font-size: 0.85rem;
        margin-top: 45px;
        padding-top: 25px;
        border-top: 1px solid rgba(30, 144, 255, 0.15);
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    .ready-state {
        text-align: center;
        padding: 100px 40px;
        background: linear-gradient(135deg, rgba(30, 144, 255, 0.08), rgba(0, 191, 255, 0.08));
        border-radius: 20px;
        border: 1px solid rgba(30, 144, 255, 0.25);
        backdrop-filter: blur(10px);
    }
    
    .ready-state h2 {
        font-size: 2.8rem;
        color: #00d9ff;
        margin-bottom: 20px;
        text-shadow: 0 2px 15px rgba(0, 217, 255, 0.2);
    }
    
    .ready-state p {
        color: #90c8ff;
        font-size: 1.05rem;
        margin: 12px 0;
        line-height: 1.6;
    }
    
    [data-testid="stTabs"] [role="tablist"] button {
        color: #90c8ff;
        font-weight: 600;
    }
    
    [data-testid="stTabs"] [role="tablist"] button[aria-selected="true"] {
        color: #00d9ff;
        border-bottom-color: #1e90ff;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256
CLASS_NAMES = ["Splicing", "Copy-Move", "Inpainting", "Face Manipulation"]

# --- MODEL LOADING ---
@st.cache_resource
def load_models():
    """
    Loads both the classification and segmentation models.
    Replace path strings with your actual .pth file paths.
    """
    # 1. Load Classification Model (EfficientNet-B3)
    try:
        cls_model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=4)
        # UNCOMMENT THE LINE BELOW AND ADD YOUR PATH
        cls_model.load_state_dict(torch.load("trained_models\unet_efficientnet_b0_epoch4.pth", map_location=DEVICE))
        cls_model.to(DEVICE)
        cls_model.eval()
    except FileNotFoundError:
        cls_model = None
        st.sidebar.warning("⚠️ Classification model file not found. Using initialized weights.")

    # 2. Load Segmentation Model (U-Net + EfficientNet-B0)
    try:
        seg_model = smp.Unet(
            encoder_name="efficientnet-b0",
            encoder_weights=None,
            in_channels=3,
            classes=1
        )
        # UNCOMMENT THE LINE BELOW AND ADD YOUR PATH
        # seg_model.load_state_dict(torch.load("unet_efficientnet_b0_epoch4.pth", map_location=DEVICE))
        seg_model.to(DEVICE)
        seg_model.eval()
    except FileNotFoundError:
        seg_model = None
        st.sidebar.warning("⚠️ Segmentation model file not found. Using initialized weights.")

    return cls_model, seg_model

cls_model, seg_model = load_models()

# --- PREPROCESSING ---
def get_preprocessing(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def preprocess_image(image_pil):
    """
    Converts PIL image to Tensor for model inference.
    """
    image_np = np.array(image_pil)
    # Ensure RGB
    if image_np.shape[-1] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    
    transform = get_preprocessing(IMG_SIZE)
    augmented = transform(image=image_np)
    image_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
    return image_np, image_tensor

# --- INFERENCE ---
def run_inference(image_tensor):
    # Classification
    cls_pred_idx = 0
    cls_probs = torch.tensor([0.25, 0.25, 0.25, 0.25]) # Default dummy
    
    if cls_model:
        with torch.no_grad():
            cls_out = cls_model(image_tensor)
            cls_probs = torch.softmax(cls_out, dim=1)[0]
            cls_pred_idx = torch.argmax(cls_probs).item()

    # Segmentation
    mask_pred = np.zeros((IMG_SIZE, IMG_SIZE))
    
    if seg_model:
        with torch.no_grad():
            seg_out = seg_model(image_tensor)
            # Sigmoid for binary probability
            seg_out = torch.sigmoid(seg_out) 
            mask_pred = seg_out.squeeze().cpu().numpy()
            
    return cls_pred_idx, cls_probs, mask_pred

# --- UI LAYOUT ---
st.title("🕵️‍♀️ DeFacto: Image Forensics Suite")
st.markdown('<p class="subtitle">Advanced Manipulation Detection & Segmentation</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📋 Upload Image")
    
    # Upload section with enhanced styling
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📁 Select Image", type=["jpg", "png", "tif", "jpeg"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### ℹ️ About")
    st.markdown('<div class="info-text" style="padding: 15px;">', unsafe_allow_html=True)
    st.info("🔎 DeFacto uses advanced AI models to detect image manipulation including splicing, copy-move forgery, inpainting, and face manipulation.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Set default mask threshold without showing slider
    mask_threshold = 0.5

if uploaded_file is not None:
    # Read Image
    image_pil = Image.open(uploaded_file).convert('RGB')
    
    # Run Analysis
    with st.spinner('🔍 Analyzing image for forgeries...'):
        original_np, image_tensor = preprocess_image(image_pil)
        cls_idx, cls_probs, mask_pred = run_inference(image_tensor)
        
        # Resize mask back to original image size for display
        mask_resized = cv2.resize(mask_pred, (original_np.shape[1], original_np.shape[0]))
        mask_binary = (mask_resized > mask_threshold).astype(np.float32)

    # --- RESULTS DISPLAY ---
    st.markdown("---")
    st.markdown("### 📊 Detection Results")
    
    # Primary Result Card
    st.markdown(f"""
    <div class="metric-card" style="padding: 40px 30px; margin-bottom: 30px; background: linear-gradient(135deg, rgba(30, 144, 255, 0.12), rgba(0, 191, 255, 0.12));">
        <div class="metric-label" style="font-size: 0.9rem;">DETECTED FORGERY TYPE</div>
        <div class="metric-value" style="font-size: 3rem; margin: 15px 0;">{CLASS_NAMES[cls_idx]}</div>
        <div class="metric-label" style="font-size: 1rem; color: #00ff9f;">Confidence: <strong>{cls_probs[cls_idx]*100:.1f}%</strong></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Top Row: Classification & Distribution
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("#### 📈 Classification Scores")
        st.markdown('<div class="probability-container">', unsafe_allow_html=True)
        chart_data = {name: prob.item() for name, prob in zip(CLASS_NAMES, cls_probs)}
        st.bar_chart(chart_data)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Create Tabs for visual modes
        st.markdown("#### 🎨 Visualization")
        tab1, tab2, tab3 = st.tabs(["🔥 Heatmap", "🔳 Mask", "🖼️ Original"])
        
        with tab1:
            # Create Heatmap Overlay
            heatmap = cv2.applyColorMap(np.uint8(255 * mask_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Blend
            overlay = cv2.addWeighted(original_np, 0.6, heatmap, 0.4, 0)
            st.image(overlay, use_column_width=True)
            
        with tab2:
            st.image(mask_binary, clamp=True, use_column_width=True)
            
        with tab3:
            st.image(original_np, use_column_width=True)
    
    # Additional insights section
    st.markdown("---")
    st.markdown("### 📋 Detailed Metrics")
    
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📐 Image Dimensions</div>
            <div class="metric-value" style="font-size: 2rem;">{original_np.shape[1]}×{original_np.shape[0]}</div>
            <div class="metric-label">pixels</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col2:
        manipulated_area = np.sum(mask_binary) / (mask_binary.shape[0] * mask_binary.shape[1]) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🎯 Manipulation Coverage</div>
            <div class="metric-value" style="font-size: 2rem;">{manipulated_area:.1f}%</div>
            <div class="metric-label">detected area</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">✓ Overall Score</div>
            <div class="metric-value" style="font-size: 2rem;">{cls_probs.max().item()*100:.0f}%</div>
            <div class="metric-label">confidence level</div>
        </div>
        """, unsafe_allow_html=True)

else:
    # Empty State
    st.markdown("""
    <div class="ready-state">
        <h2>🚀 Welcome to DeFacto</h2>
        <p>Upload an image to begin advanced forensic analysis</p>
        <p style="color: #7a9cff; font-size: 0.95rem; margin-top: 15px;"><strong>Supported Formats:</strong> JPG • PNG • TIF • JPEG</p>
        <hr style="margin: 30px auto; width: 200px; opacity: 0.3;">
        <p style="color: #90c8ff; font-size: 0.95rem; margin-top: 25px;">🔍 <strong>Detection Capabilities:</strong></p>
        <p style="color: #90c8ff; font-size: 0.9rem; margin-top: 15px;">✓ Image Splicing Detection</p>
        <p style="color: #90c8ff; font-size: 0.9rem;">✓ Copy-Move Forgery Analysis</p>
        <p style="color: #90c8ff; font-size: 0.9rem;">✓ Inpainting Detection</p>
        <p style="color: #90c8ff; font-size: 0.9rem;">✓ Face Manipulation Detection</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown('<p class="footer-text">© 2024 DeFacto Forensics | Advanced AI-Powered Image Forensics</p>', unsafe_allow_html=True)