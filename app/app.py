import joblib
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import os
import pandas as pd

# Configure page
st.set_page_config(page_title="Spam Detector", page_icon="📩")
st.title("📩 Email/SMS Spam Detector")

# Load or train model
def load_or_train_model():
    # Create models directory if needed
    os.makedirs('models', exist_ok=True)
    
    # Check if we need to train
    if not os.path.exists('models/vectorizer.pkl') or not os.path.exists('models/model.pkl'):
        try:
            # Load your dataset
            df = pd.read_csv(r"data/spam.csv", encoding='latin-1')
            
            # Clean and prepare data
            df = df.rename(columns={'v1': 'label', 'v2': 'text'})
            df['label'] = df['label'].map({'ham': 0, 'spam': 1})
            
            # Train vectorizer
            vectorizer = TfidfVectorizer(stop_words='english')
            X = vectorizer.fit_transform(df['text'])
            
            # Train model
            model = MultinomialNB()
            model.fit(X, df['label'])
            
            # Save models
            joblib.dump(vectorizer, 'models/vectorizer.pkl')
            joblib.dump(model, 'models/model.pkl')
            
            st.success("Model trained successfully on your spam.csv data!")
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            st.stop()

# Load models with verification
def load_models():
    try:
        vectorizer = joblib.load('models/vectorizer.pkl')
        model = joblib.load('models/model.pkl')
        
        # Verify vectorizer is fitted
        if not hasattr(vectorizer, 'vocabulary_'):
            raise ValueError("Vectorizer not fitted properly")
            
        return model, vectorizer
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

# Initialize models
load_or_train_model()
model, vectorizer = load_models()

# User interface
st.markdown("""
This system detects spam messages using machine learning trained on your dataset.
Enter a message below to check if it's spam or ham (legitimate).
""")

msg = st.text_area("Enter your message here:", height=150)

if st.button("Check Message"):
    if not msg.strip():
        st.warning("Please enter a message to analyze")
    else:
        try:
            # Transform and predict
            X = vectorizer.transform([msg])
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0][pred]
            
            # Display results
            if pred == 1:
                st.error(f"🚨 SPAM DETECTED! (confidence: {proba:.0%})")
                st.markdown("""
                **Characteristics of this spam message:**
                - Contains suspicious offers
                - Urgent call-to-action
                - Request for personal information
                """)
            else:
                st.success(f"✅ HAM (Not spam) (confidence: {proba:.0%})")
                st.markdown("""
                **This appears to be a legitimate message.**
                """)
                
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")

# Model information (collapsible)
with st.expander("Model Information"):
    st.write(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    st.write("Model type:", model.__class__.__name__)
    st.write("Training data source: spam.csv")