import streamlit as st
import joblib
import pandas as pd
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import os
import re
from sklearn.pipeline import Pipeline

# Configure page
st.set_page_config(
    page_title="Advanced Spam Detector", 
    page_icon="📩",
    layout="wide"
)

st.title("📩 Advanced Email/SMS Spam Detector with Explainability")

# Define preprocessing function at module level so it can be pickled
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d{3,}', 'LONGNUM', text)  # Replace long numbers
    text = re.sub(r'\b\d{4,}\b', 'LONGNUM', text)  # Replace long standalone numbers
    text = re.sub(r'\b\d+\$\b', 'MONEY', text)  # Replace money amounts
    text = re.sub(r'\b\d+\b', 'NUM', text)  # Replace other numbers
    text = re.sub(r'http\S+|www\.\S+', 'URL', text)  # Replace URLs
    text = re.sub(r'\W', ' ', text)  # Remove special chars
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Model management functions
def load_or_train_model():
    os.makedirs('models', exist_ok=True)
    
    if not (os.path.exists('models/vectorizer.pkl') and os.path.exists('models/model.pkl')):
        try:
            # Load and preprocess data
            df = pd.read_csv("data/spam.csv", encoding='latin-1')
            df = df.rename(columns={'v1': 'label', 'v2': 'text'})
            df['label'] = df['label'].map({'ham': 0, 'spam': 1})
            
            # Apply preprocessing
            df['processed_text'] = df['text'].apply(preprocess_text)
            
            # Updated vectorizer with better parameters
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 3),      # Now includes 3-grams to catch more spam patterns
                max_features=10000,       # Increased features
                min_df=2,                # Ignore terms that appear only once
                max_df=0.8               # Ignore terms that appear in >80% of docs
            )
            X = vectorizer.fit_transform(df['processed_text'])
            
            # Train model with balanced class weights
            model = MultinomialNB(alpha=0.1)  # Added smoothing
            model.fit(X, df['label'])
            
            # Save components
            joblib.dump(vectorizer, 'models/vectorizer.pkl')
            joblib.dump(model, 'models/model.pkl')
            
        except Exception as e:
            st.error(f"Model training failed: {str(e)}")
            st.stop()

def load_models():
    try:
        vectorizer = joblib.load('models/vectorizer.pkl')
        model = joblib.load('models/model.pkl')
        return model, vectorizer
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

# Initialize models
load_or_train_model()
model, vectorizer = load_models()

# Initialize LIME explainer
explainer = LimeTextExplainer(class_names=['ham', 'spam'])
predict_fn = lambda x: model.predict_proba(vectorizer.transform([preprocess_text(txt) for txt in x]))

# UI Components
col1, col2 = st.columns([2, 0.1])

with col1:
    user_input = st.text_area(
        "Enter your message:", 
        height=200,
        placeholder="Paste your email or SMS message here..."
    )

    analyze_btn = st.button("Analyze Message", type="primary")


if analyze_btn:
    if not user_input.strip():
        st.warning("Please enter a message to analyze")
    else:
        try:
            # Preprocess and transform
            processed_input = preprocess_text(user_input)
            X = vectorizer.transform([processed_input])
            proba = model.predict_proba(X)[0]
            spam_confidence = proba[1]
            
            # Improved spam detection logic with better thresholds
            if spam_confidence >= 0.8:
                st.error(f"🚨 SPAM DETECTED! (confidence: {spam_confidence:.0%})")
                st.markdown("""
                **Spam characteristics detected:**
                - Suspicious keywords or phrases
                - Urgent call-to-action
                - Requests for sensitive information
                - Financial offers or debt relief
                """)
            elif spam_confidence >= 0.6:
                st.warning(f"⚠️ Likely spam (confidence: {spam_confidence:.0%})")
                st.info("This message has strong spam characteristics but didn't meet the highest confidence threshold.")
            elif spam_confidence >= 0.4:
                st.warning(f"⚠️ Potential spam (confidence: {spam_confidence:.0%})")
                st.info("This message has some spam characteristics but needs manual review.")
            else:
                st.success(f"✅ HAM (Not spam) (confidence: {1 - spam_confidence:.0%})")
            
            # Explanation section
            st.divider()
            st.subheader("Explanation")
            
            if len(user_input.split()) < 2:
                st.warning("Message too short for detailed explanation (needs 2+ words)")
            else:
                with st.spinner("Analyzing message features..."):
                    try:
                        explanation = explainer.explain_instance(
                            user_input,
                            predict_fn,
                            num_features=10,  # Show more features
                            num_samples=5000   # More samples for better accuracy
                        )
                        
                        # Display explanation
                        weights = explanation.as_list()
                        
                        st.markdown("#### Top Influential Words/Phrases")
                        for word, weight in weights:
                            color = "red" if weight > 0 else "green"
                            impact = "strong" if abs(weight) > 0.1 else "moderate" if abs(weight) > 0.05 else "weak"
                            st.markdown(
                                f"- <span style='color:{color}'>{word}</span> ({'spam' if weight > 0 else 'ham'} indicator, {impact} impact)", 
                                unsafe_allow_html=True
                            )
                        
                        # Visual explanation
                        fig, ax = plt.subplots(figsize=(8, 4))
                        words = [w for w, _ in weights][::-1]
                        values = [v for _, v in weights][::-1]
                        ax.barh(words, values, color=['red' if v > 0 else 'green' for v in values])
                        ax.set_title("Feature Impact on Prediction")
                        ax.set_xlabel("Impact on Spam Probability")
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"Explanation failed: {str(e)}")
                        st.info("Try a slightly longer message for better explanation")

        except Exception as e:
            st.error(f"Analysis error: {str(e)}")

# About section
st.markdown("### About this detector")
st.markdown("""
- Uses enhanced Naive Bayes classifier with smoothing  
- Trained on SMS Spam Collection dataset with improved preprocessing  
- Uses 1-3 gram features to capture more spam patterns  
- Special handling for numbers, URLs, and money amounts  
- Provides detailed explanations using LIME with impact strength  
""")

# Model info expander
with st.expander("Technical Details"):
    st.write(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    st.write("Model type:", model.__class__.__name__)
    st.write("Confidence thresholds:")
    st.write("- 80%+ = Spam")
    st.write("- 60-80% = Likely spam")
    st.write("- 40-60% = Potential spam")
    st.write("- Below 40% = Ham")
    st.write("Minimum words for explanation:", "2")