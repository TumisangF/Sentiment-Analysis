import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pathlib import Path
import os
import sys
import logging

# Suppress warnings to clean up logs
logging.getLogger('streamlit').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)

# IMPORTANT: Disable Streamlit's file watcher to prevent restart loops
os.environ["STREAMLIT_SERVER_WATCH_FILES"] = "false"
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis Tool",
    page_icon="😊",
    layout="wide"
)

# ── Styling ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .block-container { padding-top: 2rem; }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
        margin-bottom: 0.5rem;
    }
    .joy { border-color: #fbbf24; background: linear-gradient(135deg, #fef3c7, white); }
    .anger { border-color: #ef4444; background: linear-gradient(135deg, #fee2e2, white); }
    .sadness { border-color: #6366f1; background: linear-gradient(135deg, #e0e7ff, white); }
    .fear { border-color: #8b5cf6; background: linear-gradient(135deg, #ede9fe, white); }
    .surprise { border-color: #06b6d4; background: linear-gradient(135deg, #cffafe, white); }
    .neutral { border-color: #9ca3af; background: linear-gradient(135deg, #f3f4f6, white); }
    h1 { color: #1b2a4a; }
    h2, h3 { color: #334155; }
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 2rem;
    }
    .error-box {
        background-color: #fee2e2;
        border-left: 5px solid #ef4444;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────
st.title("Sentiment Analysis Tool")
st.markdown("**Advanced customer emotion detection** — Upload customer reviews and get detailed emotion analysis (Joy, Anger, Sadness, Fear, Surprise, Neutral)")
st.markdown("---")

# ── Load model (cached and with error handling) ─────────────
@st.cache_resource(show_spinner="Loading emotion analysis model...", ttl=3600)
def load_emotion_model():
    """Load your fine-tuned emotion model"""
    try:
        # Define emotion labels
        labels = ["joy", "anger", "sadness", "fear", "surprise", "neutral"]
        id2label = {i: label for i, label in enumerate(labels)}
        
        # Get the directory where this script is located
        current_dir = Path(__file__).parent
        model_path = current_dir / "saved_model"
        
        # Check if model exists
        if model_path.exists():
            st.info("✅ Loading custom emotion model...")
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
            st.success("✅ Custom model loaded successfully!")
        else:
            st.warning("⚠️ Custom model not found. Using fallback model...")
            model_name = "j-hartmann/emotion-english-distilroberta-base"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            st.info("✅ Fallback model loaded successfully!")
        
        # Move to appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        return model, tokenizer, id2label, device, labels
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None, None, None

# Load model
model, tokenizer, id2label, device, labels = load_emotion_model()

# Check if model loaded successfully
if model is None:
    st.markdown("""
    <div class="error-box">
        <strong>⚠️ Model Failed to Load</strong><br>
        The emotion analysis model could not be loaded. This might be because:
        <ul>
            <li>The model files are missing or corrupt</li>
            <li>There's a memory issue on the server</li>
            <li>There's a compatibility issue with the model files</li>
        </ul>
        Please check the logs or contact support.
    </div>
    """, unsafe_allow_html=True)
    
    # Display debug info
    with st.expander("🔧 Debug Information"):
        st.write("**Model Path Check:**")
        current_dir = Path(__file__).parent
        model_path = current_dir / "saved_model"
        st.write(f"Expected model path: {model_path}")
        st.write(f"Path exists: {model_path.exists()}")
        
        if model_path.exists():
            st.write("**Files in saved_model:**")
            for file in model_path.iterdir():
                st.write(f"- {file.name} ({file.stat().st_size / 1024:.1f} KB)")
    
    st.stop()

# ── Text cleaning ───────────────────────────────────
def clean_text(text):
    """Advanced text cleaning"""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s\.\,\!\\?\\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text[:512]

# ── Emotion prediction ──────────────────────────────────────
def predict_emotion(texts, model, tokenizer, id2label, device, batch_size=16):
    """Predict emotions for a list of texts"""
    if model is None:
        return [{"predicted_emotion": "error", "confidence": 0.0, "all_emotions": {}} for _ in texts]
    
    results = []
    progress_bar = st.progress(0, text="Analysing emotions...")
    total = len(texts)
    
    for i in range(0, total, batch_size):
        batch = texts[i: i + batch_size]
        batch = [t if t.strip() else "no content" for t in batch]
        
        cleaned_batch = [clean_text(t) for t in batch]
        
        inputs = tokenizer(
            cleaned_batch,
            truncation=True,
            padding=True,
            max_length=64,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
            predicted_classes = torch.argmax(predictions, dim=-1)
            confidences = torch.max(predictions, dim=-1)[0]
        
        for j, (pred_class, conf) in enumerate(zip(predicted_classes, confidences)):
            emotion = id2label[pred_class.item()]
            confidence = conf.item()
            
            probs = predictions[j].cpu().numpy()
            all_emotions = {id2label[k]: float(probs[k]) for k in range(len(id2label))}
            
            results.append({
                "predicted_emotion": emotion,
                "confidence": round(confidence, 4),
                "all_emotions": all_emotions
            })
        
        progress_bar.progress(min((i + batch_size) / total, 1.0), 
                              text=f"Analysing emotions... {min(i+batch_size, total)}/{total}")
    
    progress_bar.empty()
    return results

# ── Keyword extraction ───────────────────────────
STOP_WORDS = {
    "this","that","with","have","from","they","will","been","were","their",
    "what","when","also","just","very","more","some","than","there","about",
    "would","which","these","other","into","after","over","then","only"
}

def top_keywords_by_emotion(df, emotion, n=10):
    emotion_texts = df[df["emotion"] == emotion]["clean_text"].tolist()
    if not emotion_texts:
        return []
    
    words = []
    for t in emotion_texts:
        words += [w for w in t.split() if len(w) >= 4 and w not in STOP_WORDS]
    
    return Counter(words).most_common(n)

# ── Color mapping ──────────────────────────────
EMOTION_COLORS = {
    "joy": "#fbbf24",
    "anger": "#ef4444",
    "sadness": "#6366f1",
    "fear": "#8b5cf6",
    "surprise": "#06b6d4",
    "neutral": "#9ca3af"
}

EMOTION_EMOJIS = {
    "joy": "😊",
    "anger": "😠",
    "sadness": "😢",
    "fear": "😨",
    "surprise": "😲",
    "neutral": "😐"
}

# ════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════

# Sidebar
with st.sidebar:
    st.header("📁 Upload your data")
    st.markdown("Upload a CSV file with a column containing review text.")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv", "txt"])
    
    col_name = None
    df_raw = None
    
    if uploaded:
        try:
            try:
                df_raw = pd.read_csv(uploaded)
            except Exception:
                uploaded.seek(0)
                df_raw = pd.read_csv(uploaded, sep="\t", header=None, names=["text"])
            
            st.success(f"✅ Loaded {len(df_raw):,} rows")
            st.dataframe(df_raw.head(3), use_container_width=True)
            
            col_name = st.selectbox(
                "Which column contains the review text?",
                options=df_raw.columns.tolist()
            )
        except Exception as e:
            st.error(f"Could not read file: {e}")
    
    sample_size = st.slider(
        "Max reviews to analyse", 50, 2000, 500, step=50,
        help="Reduce for faster results"
    )
    
    st.markdown("---")
    st.markdown("🤖 **Model Information**")
    
    # Safe device display
    device_str = str(device).upper() if device else "CPU"
    st.info(f"""
    - **Model:** Fine-tuned Emotion Classifier
    - **Accuracy:** 82.37% on validation set
    - **Classes:** 6 emotions
    - **Device:** {device_str}
    """)
    
    run = st.button("▶ Analyse Emotions", type="primary", disabled=(df_raw is None or col_name is None))

# Main content
if not uploaded:
    st.info("👈 Upload a CSV file in the sidebar to get started.")
    st.markdown("""
    ### 🎯 What you'll get:
    - 🎭 **6 Emotion categories**: Joy, Anger, Sadness, Fear, Surprise, Neutral
    - 📊 **Confidence scores** for each prediction
    - 📈 **Emotion distribution** charts
    - 🔍 **Emotion-specific keyword analysis**
    - 📥 **Downloadable results**
    
    ### 📁 File Format:
    - CSV file with at least one text column
    - Supports both comma and tab-separated files
    
    ### 📊 Model Performance:
    | Metric | Score |
    |--------|-------|
    | Accuracy | 82.37% |
    | F1 Score | 82.31% |
    | Precision | 83.03% |
    | Recall | 82.37% |
    """)

elif run and model is not None:
    # Prepare data
    df = df_raw[[col_name]].rename(columns={col_name: "text"}).copy()
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        st.caption(f"Analysing a random sample of {sample_size} reviews.")
    
    df["clean_text"] = df["text"].apply(clean_text)
    
    # Predict
    with st.spinner("Analyzing emotions..."):
        predictions = predict_emotion(df["clean_text"].tolist(), model, tokenizer, id2label, device)
    
    df["emotion"] = [p["predicted_emotion"] for p in predictions]
    df["confidence"] = [p["confidence"] for p in predictions]
    
    # Counts
    counts = df["emotion"].value_counts()
    total = len(df)
    
    # Display metrics
    st.subheader("📈 Emotion Analysis Overview")
    
    cols = st.columns(3)
    for i, emotion in enumerate(labels):
        col = cols[i % 3]
        count = counts.get(emotion, 0)
        percentage = round(count / total * 100, 1) if total > 0 else 0
        emoji = EMOTION_EMOJIS.get(emotion, "")
        
        with col:
            st.markdown(f"""
            <div class="metric-card {emotion}">
                <h3 style="margin:0">{emoji} {emotion.title()}</h3>
                <p style="font-size:2rem; margin:0; font-weight:bold;">{count:,}</p>
                <p style="margin:0; color:#6b7280;">{percentage}% of reviews</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts
    if total > 0:
        st.subheader("📊 Emotion Visualization")
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("**Emotion distribution**")
            fig, ax = plt.subplots(figsize=(5, 4))
            counts_list = [counts.get(e, 0) for e in labels]
            colors_list = [EMOTION_COLORS.get(e, "#9ca3af") for e in labels]
            
            bars = ax.bar(labels, counts_list, color=colors_list, alpha=0.8)
            ax.set_ylabel("Number of reviews")
            ax.tick_params(axis='x', rotation=45)
            ax.spines[["top", "right"]].set_visible(False)
            
            for bar, count in zip(bars, counts_list):
                if count > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                            str(count), ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        
        with col_b:
            st.markdown("**Confidence by emotion**")
            fig, ax = plt.subplots(figsize=(5, 4))
            
            boxplot_data = []
            boxplot_labels = []
            for emotion in labels:
                emotion_conf = df[df["emotion"] == emotion]["confidence"]
                if len(emotion_conf) > 0:
                    boxplot_data.append(emotion_conf)
                    boxplot_labels.append(emotion)
            
            if boxplot_data:
                bp = ax.boxplot(boxplot_data, labels=boxplot_labels, patch_artist=True)
                for patch, emotion in zip(bp['boxes'], boxplot_labels):
                    patch.set_facecolor(EMOTION_COLORS.get(emotion, "#9ca3af"))
                    patch.set_alpha(0.7)
                
                ax.set_ylabel("Confidence score")
                ax.set_ylim(0, 1)
                ax.tick_params(axis='x', rotation=45)
                ax.spines[["top", "right"]].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        
        st.markdown("---")
        
        # Results table
        st.subheader("🔍 Detailed emotion analysis")
        
        display_df = df[["text", "emotion", "confidence"]].copy()
        display_df["confidence"] = (display_df["confidence"] * 100).round(1).astype(str) + "%"
        display_df["emotion"] = display_df["emotion"].apply(
            lambda x: f"{EMOTION_EMOJIS.get(x, '')} {x.title()}"
        )
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Download
        st.markdown("---")
        st.subheader("⬇️ Download results")
        
        csv_bytes = df[["text", "emotion", "confidence"]].to_csv(index=False).encode()
        st.download_button(
            "📄 Download predictions CSV",
            data=csv_bytes,
            file_name="emotion_predictions.csv",
            mime="text/csv"
        )

elif run and model is None:
    st.error("❌ Cannot analyze emotions: Model failed to load. Please check the logs.")