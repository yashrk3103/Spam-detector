# 📩 Email/SMS Spam Detector with Explainability

This project is a complete end-to-end machine learning application that detects whether a given **email or SMS message is spam or not**, using advanced **natural language processing (NLP)** and a **Naive Bayes classifier** — all wrapped in a modern and interactive **Streamlit** web interface.

The application goes beyond simple classification by providing **model explainability using LIME (Local Interpretable Model-Agnostic Explanations)**, helping users understand *why* a message was classified as spam or ham. This makes the system transparent, educational, and more trustworthy for users.

---

## 🚀 Project Goals

- Build a robust ML model to classify messages as **spam** or **ham**
- Apply **real-world NLP preprocessing**: cleaning URLs, numbers, symbols, and noise
- Use **TF-IDF vectorization** with 1–3 gram features for enhanced pattern recognition
- Enable **LIME explainability** to highlight influential words/phrases
- Visualize predictions and explanations in a clean, browser-based UI

---

## 🔍 Key Features

- 🧠 Custom preprocessing pipeline to normalize text
- 📊 TF-IDF vectorizer (1–3 grams, max 10,000 features)
- ⚖️ Tuned Multinomial Naive Bayes model with smoothing
- 🎯 Multi-level spam detection thresholds:
  - 80%+ → Definite spam  
  - 60–80% → Likely spam  
  - 40–60% → Potential spam  
  - Below 40% → Ham
- 🔎 LIME-powered explainability:
  - Top influential words/phrases
  - Impact strength (strong/moderate/weak)
  - Color-coded bar chart for interpretation
- 🧪 Uses the **SMS Spam Collection Dataset** (UCI)

---

## 💡 What You’ll Learn from This Project

- How to clean and vectorize textual data
- How to train, evaluate, and persist ML models using `scikit-learn`
- How to build modular, production-ready ML pipelines
- How to serve ML models as interactive web apps using Streamlit
- How to apply **Explainable AI (XAI)** to NLP use cases

---

## 📂 Technology Stack

- **Frontend & UI**: Streamlit  
- **Data Handling**: Pandas  
- **Modeling**: Scikit-learn (Naive Bayes + TF-IDF)  
- **Explainability**: LIME  
- **Visualization**: Matplotlib  
- **Packaging**: Joblib

---

## 📁 Dataset

This project is trained on the **SMS Spam Collection Dataset**, a popular benchmark dataset for binary text classification tasks involving spam detection. It contains over 5,000 real SMS messages labeled as `spam` or `ham`.

Dataset source: [UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

---

## 📌 Use Cases

- Educational tool for learning NLP and spam detection
- Lightweight explainable AI (XAI) demo
- Prototype for filtering malicious or promotional SMS/email traffic
- Deployment-ready ML app for showcasing end-to-end ML skills

---

Feel free to explore the code, tweak parameters, add new features, or integrate more advanced models like Logistic Regression or BERT in future versions.
