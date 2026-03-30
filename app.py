import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from collections import Counter
from transformers import pipeline
import io

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis Tool",
    page_icon="📊",
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
    .positive { border-color: #22c55e; }
    .neutral  { border-color: #f59e0b; }
    .negative { border-color: #ef4444; }
    h1 { color: #1b2a4a; }
    h2, h3 { color: #334155; }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────
st.title("📊 Sentiment Analysis Tool")
st.markdown("**For the Marketing Department** — Upload a CSV of customer reviews and click Analyse.")
st.markdown("---")

# ── Load model (cached so it only loads once) ───────────────
@st.cache_resource(show_spinner="Loading sentiment model — please wait...")
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512,
        top_k=1
    )

# ── Preprocessing ────────────────────────────────────────────
NEGATION = re.compile(
    r"\b(not|no|never|cannot|can't|won't|don't|doesn't|didn't|isn't|wasn't)\s+(\w+)",
    re.IGNORECASE
)

def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = NEGATION.sub(lambda m: m.group(1) + "_" + m.group(2), text)
    text = re.sub(r"[^a-z0-9_ ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:512]

# ── Map 2-class model → 3 classes ───────────────────────────
def map_label(label, score):
    """
    DistilBERT is binary (POSITIVE / NEGATIVE).
    We derive NEUTRAL when the model is uncertain (score between 0.55 and 0.75).
    """
    label = label.upper()
    if score < 0.75:
        return "neutral"
    return "positive" if label == "POSITIVE" else "negative"

# ── Classify ─────────────────────────────────────────────────
def classify(texts, model, batch_size=32):
    results = []
    progress = st.progress(0, text="Analysing reviews...")
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i: i + batch_size]
        batch = [t if t.strip() else "no content" for t in batch]
        preds = model(batch)
        for pred in preds:
            p = pred[0] if isinstance(pred, list) else pred
            sentiment = map_label(p["label"], p["score"])
            results.append({"predicted_sentiment": sentiment, "confidence": round(p["score"], 4)})
        progress.progress(min((i + batch_size) / total, 1.0), text=f"Analysing reviews... {min(i+batch_size, total)}/{total}")
    progress.empty()
    return results

# ── Keyword extraction ───────────────────────────────────────
STOP = {
    "this","that","with","have","from","they","will","been","were","their",
    "what","when","also","just","very","more","some","than","there","about",
    "would","which","these","other","into","after","over","then","only",
    "good","great","like","even","well","still","much","your","dont","cant",
    "book","time","movie","product","first","read","does","could","didn",
    "really","because","while","though","although","however","the","and",
    "for","are","was","but","not","you","its","has","had","him","her"
}

def top_keywords(texts, n=10):
    words = []
    for t in texts:
        words += [w for w in t.split() if len(w) >= 4 and w not in STOP]
    return Counter(words).most_common(n)

# ════════════════════════════════════════════════════════════
# SIDEBAR — Upload
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("📁 Upload your data")
    st.markdown("Upload a CSV file with a column containing review text.")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv", "txt"])

    col_name = None
    df_raw = None

    if uploaded:
        try:
            # Try comma separator first, then tab (for FastText-style files)
            try:
                df_raw = pd.read_csv(uploaded)
            except Exception:
                uploaded.seek(0)
                df_raw = pd.read_csv(uploaded, sep="\t", header=None, names=["text"])

            st.success(f"Loaded {len(df_raw):,} rows")
            st.markdown("**Preview:**")
            st.dataframe(df_raw.head(3), use_container_width=True)

            col_name = st.selectbox(
                "Which column contains the review text?",
                options=df_raw.columns.tolist()
            )
        except Exception as e:
            st.error(f"Could not read file: {e}")

    sample_size = st.slider(
        "Max reviews to analyse", 50, 1000, 300, step=50,
        help="Reduce for faster results"
    )

    run = st.button("▶ Analyse", type="primary", disabled=(df_raw is None or col_name is None))

# ════════════════════════════════════════════════════════════
# MAIN — Results
# ════════════════════════════════════════════════════════════
if not uploaded:
    # Landing state
    st.info("👈  Upload a CSV file in the sidebar to get started.")
    st.markdown("""
    **Accepted formats:**
    - CSV with a text column (e.g. `review`, `text`, `comment`)
    - Amazon FastText format (`__label__1 review text`)

    **What you'll get:**
    - Sentiment label per review (positive / neutral / negative)
    - Confidence score per prediction
    - Sentiment distribution chart
    - Top negative keywords
    - Downloadable results CSV
    """)

elif run:
    model = load_model()

    # Prepare data
    df = df_raw[[col_name]].rename(columns={col_name: "text"}).copy()
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        st.caption(f"Analysing a random sample of {sample_size} reviews.")

    df["clean_text"] = df["text"].apply(preprocess)

    # Classify
    preds = classify(df["clean_text"].tolist(), model)
    df["sentiment"] = [p["predicted_sentiment"] for p in preds]
    df["confidence"] = [p["confidence"] for p in preds]

    # Counts
    counts = df["sentiment"].value_counts()
    total  = len(df)
    pos_pct = round(counts.get("positive", 0) / total * 100, 1)
    neu_pct = round(counts.get("neutral",  0) / total * 100, 1)
    neg_pct = round(counts.get("negative", 0) / total * 100, 1)

    # ── KPI row ──────────────────────────────────────────────
    st.subheader("📈 Sentiment Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total reviews", f"{total:,}")
    c2.metric("Positive", f"{pos_pct}%", delta=None)
    c3.metric("Neutral",  f"{neu_pct}%")
    c4.metric("Negative", f"{neg_pct}%")

    st.markdown("---")

    # ── Charts row ───────────────────────────────────────────
    st.subheader("📊 Visual Summary")
    col_a, col_b, col_c = st.columns(3)

    # Chart 1: Pie
    with col_a:
        st.markdown("**Sentiment distribution**")
        fig1, ax1 = plt.subplots(figsize=(4, 4))
        sizes  = [counts.get(l, 0) for l in ["positive", "neutral", "negative"]]
        labels = [f"Positive\n{pos_pct}%", f"Neutral\n{neu_pct}%", f"Negative\n{neg_pct}%"]
        colors = ["#22c55e", "#9ca3af", "#ef4444"]
        ax1.pie(sizes, labels=labels, colors=colors, startangle=140,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5})
        ax1.axis("equal")
        st.pyplot(fig1)
        plt.close(fig1)

    # Chart 2: Negative keywords
    with col_b:
        st.markdown("**Top keywords in negative reviews**")
        neg_texts = df[df["sentiment"] == "negative"]["clean_text"].tolist()
        if neg_texts:
            kw = top_keywords(neg_texts, n=10)
            kw_df = pd.DataFrame(kw, columns=["keyword", "count"])
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            ax2.barh(kw_df["keyword"][::-1], kw_df["count"][::-1],
                     color="#ef4444", alpha=0.8)
            ax2.set_xlabel("Frequency")
            ax2.tick_params(labelsize=9)
            ax2.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.info("No negative reviews found.")

    # Chart 3: Confidence histogram
    with col_c:
        st.markdown("**Confidence score distribution**")
        fig3, ax3 = plt.subplots(figsize=(4, 4))
        for label, color in zip(["positive", "neutral", "negative"],
                                ["#22c55e", "#9ca3af", "#ef4444"]):
            subset = df[df["sentiment"] == label]["confidence"]
            if len(subset) > 0:
                ax3.hist(subset, bins=15, alpha=0.6,
                         label=label.capitalize(), color=color)
        ax3.set_xlabel("Confidence")
        ax3.set_ylabel("Count")
        ax3.legend(fontsize=8)
        ax3.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig3)
        plt.close(fig3)

    st.markdown("---")

    # ── Per-review table ─────────────────────────────────────
    st.subheader("🔍 Per-review predictions")

    # Colour-coded sentiment column
    def colour_sentiment(val):
        colours = {"positive": "#dcfce7", "neutral": "#fef9c3", "negative": "#fee2e2"}
        return f"background-color: {colours.get(val, 'white')}"

    display_df = df[["text", "sentiment", "confidence"]].copy()
    display_df["confidence"] = (display_df["confidence"] * 100).round(1).astype(str) + "%"

    st.dataframe(
        display_df.style.applymap(colour_sentiment, subset=["sentiment"]),
        use_container_width=True,
        height=320
    )

    # ── Downloads ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⬇️ Download results")

    dl1, dl2 = st.columns(2)

    # CSV download
    csv_bytes = df[["text", "sentiment", "confidence"]].to_csv(index=False).encode()
    dl1.download_button(
        "📄 Download predictions CSV",
        data=csv_bytes,
        file_name="predictions.csv",
        mime="text/csv"
    )

    # Summary report download
    neg_kw = top_keywords(neg_texts if neg_texts else [], n=5)
    report = "\n".join([
        "SENTIMENT ANALYSIS SUMMARY REPORT",
        "=" * 40,
        f"Total reviews analysed : {total}",
        "",
        "Sentiment Distribution:",
        f"  Positive : {counts.get('positive',0):5d} ({pos_pct}%)",
        f"  Neutral  : {counts.get('neutral', 0):5d} ({neu_pct}%)",
        f"  Negative : {counts.get('negative',0):5d} ({neg_pct}%)",
        "",
        "Top 5 Negative Keywords:",
    ] + [f"  {w:20s}: {c}" for w, c in neg_kw] + [
        "",
        "Model: distilbert-base-uncased-finetuned-sst-2-english",
        "Tool : Sentiment Analysis Tool — DLBDSEAIS02"
    ])

    dl2.download_button(
        "📝 Download summary report",
        data=report.encode(),
        file_name="summary_report.txt",
        mime="text/plain"
    )

elif uploaded and not run:
    st.info("👈 Click **▶ Analyse** in the sidebar when you're ready.")
