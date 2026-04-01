---
title: Emotion Analysis Tool
emoji: 😊
colorFrom: blue
colorTo: yellow
sdk: streamlit
sdk_version: 1.29.0
app_file: app.py
pinned: false
---

# 😊 Sentiment Analysis Tool

## 🎯 Overview
Advanced emotion detection for customer reviews supporting **6 emotions**:
- 😊 Joy
- 😠 Anger  
- 😢 Sadness
- 😨 Fear
- 😲 Surprise
- 😐 Neutral

## Features
- Real-time emotion analysis
- Interactive visualizations
- Keyword extraction by emotion
- Download results (CSV)

## Model
Fine-tuned DistilRoBERTa emotion classifier achieving **82.37% accuracy** on validation set.

## Usage
1. Upload CSV with review text
2. Click "Analyse Emotions"
3. View detailed analytics
4. Download results

## Model Performance
| Metric | Score |
|--------|-------|
| Accuracy | 82.37% |
| F1 Score | 82.31% |
| Precision | 83.03% |
| Recall | 82.37% |