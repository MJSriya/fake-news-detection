# 📰 Fake News Detection System

A full-stack machine learning project that detects whether a news article is **Fake or Real** using Natural Language Processing and a web-based interface.

---

## 🚀 Features
- Text classification using Logistic Regression
- TF-IDF vectorization for feature extraction
- Fast API backend for real-time predictions
- Interactive frontend with modern UI (glassmorphism design)
- Instant prediction without page reloads

---

## 🏗️ Project Architecture

1. **Model Training (`train.py`)**
   - Uses `TfidfVectorizer` + `LogisticRegression`
   - Trains on dataset and saves model as `model.pkl`

2. **Backend (`main.py`)**
   - Built with FastAPI
   - Provides `/predict` API endpoint for inference

3. **Frontend (`static/index.html`)**
   - HTML, CSS, JavaScript (Vanilla)
   - Uses `fetch()` to communicate with backend
   - Smooth UI with animations

---

## ⚙️ Tech Stack
- Python
- Scikit-learn
- FastAPI
- HTML / CSS / JavaScript

---
