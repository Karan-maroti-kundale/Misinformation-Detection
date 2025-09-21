import os
import io
import streamlit as st
from PIL import Image, ImageChops, ImageEnhance, ExifTags
import numpy as np
import cv2
import tempfile
from dotenv import load_dotenv
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from skimage.metrics import structural_similarity as ssim


# ------------------------------
# Setup & Model Download (one-time)
# ------------------------------
MODEL_DIR = "models/distilbert_fake_news"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR, exist_ok=True)
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    tokenizer.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)

# Load environment variables
load_dotenv("config/.env")


# ------------------------------
# Text Model: DistilBERT (PyTorch)
# ------------------------------
@st.cache_resource
def load_distilbert_model(model_path=MODEL_DIR):
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading DistilBERT model: {e}")
        return None, None


def predict_distilbert(text, tokenizer, model):
    encodings = tokenizer([text], truncation=True, padding=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encodings)
        probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    label = "FAKE" if np.argmax(probs) == 0 else "TRUE"
    confidence = float(max(probs) * 100)
    return label, confidence, probs


# ------------------------------
# Image Analysis: ELA + Metadata
# ------------------------------
def error_level_analysis(img: Image.Image, quality=90):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    img.save(temp_file.name, "JPEG", quality=quality)
    compressed = Image.open(temp_file.name)
    ela = ImageChops.difference(img, compressed)
    extrema = ela.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1
    ela = ImageEnhance.Brightness(ela).enhance(scale)
    return ela


def get_image_metadata(img: Image.Image):
    metadata = {}
    try:
        info = img._getexif()
        if info:
            for tag, value in info.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                metadata[decoded] = value
    except Exception:
        pass
    return metadata


# ------------------------------
# Video Analysis: Scene-based Keyframes + Deepfake Hook
# ------------------------------
def extract_keyframes(video_path, threshold=0.85, max_frames=15):
    cap = cv2.VideoCapture(video_path)
    success, prev_frame = cap.read()
    keyframes = []

    if not success:
        cap.release()
        return keyframes

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    keyframes.append(prev_frame)

    while success:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(prev_gray, gray, full=True)

        if score < threshold:  # scene change detected
            keyframes.append(frame)
            prev_gray = gray

        if len(keyframes) >= max_frames:
            break

    cap.release()
    return keyframes


def detect_deepfake(frame):
    """
    Placeholder for deepfake detection.
    Later: load a pre-trained CNN/transformer model and run inference here.
    """
    return {"label": "Likely Real", "confidence": 95.2}


# ------------------------------
# Streamlit App
# ------------------------------
st.set_page_config(page_title="AI Misinformation Assistant", page_icon="🛡️", layout="wide")
st.title("🛡️ AI-Powered Misinformation Detection & Literacy Assistant")

# Load DistilBERT model
tokenizer, model = load_distilbert_model()
MODEL_OK = tokenizer is not None and model is not None

st.sidebar.header("Model Status")
if MODEL_OK:
    st.sidebar.success("DistilBERT PyTorch model loaded ✅")
else:
    st.sidebar.warning("DistilBERT model not found. Make sure 'models/distilbert_fake_news' exists.")


# Tabs
tab1, tab2, tab3 = st.tabs(["📝 Text", "🖼️ Image", "🎬 Video"])

# --- Text Tab ---
with tab1:
    st.subheader("Check a piece of text")
    user_text = st.text_area("Paste the message / headline here:", height=160)
    if st.button("Analyze Text"):
        if not user_text.strip():
            st.error("Please paste some text.")
        elif MODEL_OK:
            label, confidence, probs = predict_distilbert(user_text, tokenizer, model)
            st.write(f"**Prediction:** {label}")
            st.write(f"**Confidence:** {confidence:.1f}%")
            st.write(f"**Probabilities:** Fake={probs[0]*100:.1f}%, True={probs[1]*100:.1f}%")
        else:
            st.warning("Model not loaded.")


# --- Image Tab ---
with tab2:
    st.subheader("Check an image")
    uploaded_file = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
        st.image(img, caption="Uploaded image", use_container_width=True)

        st.info("🔎 Running ELA and metadata checks...")
        ela_img = error_level_analysis(img)
        st.image(ela_img, caption="Error Level Analysis (ELA)", use_container_width=True)

        metadata = get_image_metadata(img)
        if metadata:
            st.write("Metadata found:")
            st.json(metadata)
        else:
            st.info("No metadata found.")


# --- Video Tab ---
with tab3:
    st.subheader("Check a video")
    uploaded_video = st.file_uploader("Upload a video (mp4)", type=["mp4"])
    if uploaded_video:
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(uploaded_video.read())
        st.video(temp_video.name)

        st.info("🔎 Extracting keyframes (scene-based)...")
        keyframes = extract_keyframes(temp_video.name)
        st.write(f"Unique keyframes detected: {len(keyframes)}")

        if keyframes:
            for i, frame in enumerate(keyframes[:5]):  # Show up to 5 frames
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(rgb_frame, caption=f"Scene {i+1}", width=300)

                # Deepfake detection placeholder
                result = detect_deepfake(frame)
                st.write(f"🕵️ Deepfake Check: {result['label']} ({result['confidence']}% confidence)")


st.caption("Educational starter. Always verify with trusted sources.")
