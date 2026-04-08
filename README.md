# Sentiment Analysis Tool

This is a **Streamlit-based Emotion Analysis web app** that analyzes customer reviews and detects **6 emotions** (joy, anger, sadness, fear, surprise, neutral) with special handling for **multipolar reviews** and **sarcasm/irony detection**.

The app uses a fine-tuned **DistilRoBERTa model** trained on 150,000 customer reviews with class weighting to handle imbalanced data.

---

## 🔗 Live Demo

You can try the app online here:  
[Sentiment Analysis Tool on Hugging Face Spaces](https://huggingface.co/spaces/Tumisang555/Sentiment-Analysis-tool)

---

## ⚡ Features

- **6 Emotion Classes**: Joy, Anger, Sadness, Fear, Surprise, Neutral
- **Multipolarity Detection**: Identifies reviews with both positive and negative aspects
- **Sarcasm/Irony Recognition**: Rule-based detection with emotion flipping
- **Class-Weighted Training**: Handles imbalanced classes (e.g., rare "fear" samples)
- **Real-time predictions** with confidence scores
- **Lightweight & fast** using Streamlit and optimized DistilRoBERTa  

---

## 📊 Model Performance

Trained on 150,000 customer reviews:

| Metric | Score |
|--------|-------|
| Accuracy | 87.2% |
| F1 | 87% |
| Recall | 87% |
| Precision | 87.2% |

*Per-class metrics available in the training report.*

---

## 🛠️ Installation (Optional, for local use)

1. Clone this repository:

```bash
git clone https://github.com/TumisangF/Sentiment-Analysis.git
cd Sentiment-Analysis
