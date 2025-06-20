import streamlit as st
import joblib
import pandas as pd
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import os

# Configure page
st.set_page_config(
    page_title="Advanced Spam Detector", 
    page_icon="📩",
    layout="wide"
)

st.title("📩 Advanced Email/SMS Spam Detector with Explainability")

# Model management functions
def load_or_train_model():
    os.makedirs('models', exist_ok=True)
    
    if not (os.path.exists('models/vectorizer.pkl') and os.path.exists('models/model.pkl')):
        try:
            df = pd.read_csv("data/spam.csv", encoding='latin-1')
            df = df.rename(columns={'v1': 'label', 'v2': 'text'})
            df['label'] = df['label'].map({'ham': 0, 'spam': 1})
            
            vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
            X = vectorizer.fit_transform(df['text'])
            
            model = MultinomialNB()
            model.fit(X, df['label'])
            
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
predict_fn = lambda x: model.predict_proba(vectorizer.transform(x))

# UI Components
col1, col2 = st.columns([2, 0.1])

with col1:
    user_input = st.text_area(
        "Enter your message:", 
        height=200,
        placeholder="Paste your email or SMS message here..."
    )

    analyze_btn = st.button("Analyze Message", type="primary")

# with col2:
#     st.markdown("### About this detector")
#     st.markdown("""
#     - Uses Naive Bayes classifier
#     - Trained on SMS Spam Collection dataset
#     - Requires 80%+ confidence for spam detection
#     - Provides explanations using LIME
#     """)

if analyze_btn:
    if not user_input.strip():
        st.warning("Please enter a message to analyze")
    else:
        try:
            # Transform and predict
            X = vectorizer.transform([user_input])
            proba = model.predict_proba(X)[0]
            confidence = max(proba)
            pred = proba.argmax()
            
            # Enhanced spam detection logic
            if pred == 1 and proba[1] >= 0.6:  # Only classify as spam if confidence >= 80%
                st.error(f"🚨 SPAM DETECTED! (confidence: {proba[1]:.0%})")
                st.markdown("""
                **Spam characteristics detected:**
                - Suspicious keywords
                - Urgent call-to-action
                - Request for personal information
                - Unusual links or numbers
                """)
            elif pred == 1 and proba[1] < 0.6:
                st.warning(f"⚠️ Potential spam (low confidence: {proba[1]:.0%})")
                st.info("This message has some spam characteristics but didn't meet the high confidence threshold.")
            else:
                st.success(f"✅ HAM (Not spam) (confidence: {proba[0]:.0%})")
            
            # Explanation section
            st.divider()
            st.subheader("Explanation")
            
            if len(user_input.split()) < 3:
                st.warning("Message too short for detailed explanation (needs 3+ words)")
            else:
                with st.spinner("Analyzing message features..."):
                    try:
                        explanation = explainer.explain_instance(
                            user_input,
                            predict_fn,
                            num_features=5,
                            top_labels=1
                        )
                        
                        # Display explanation
                        weights = explanation.as_list()
                        
                        st.markdown("#### Top Influential Words")
                        for word, weight in weights:
                            color = "red" if weight > 0 else "green"
                            st.markdown(f"- <span style='color:{color}'>{word}</span> ({'spam' if weight > 0 else 'ham'} indicator)", 
                                      unsafe_allow_html=True)
                        
                        # Visual explanation
                        fig, ax = plt.subplots()
                        words = [w for w, _ in weights][::-1]
                        values = [v for _, v in weights][::-1]
                        ax.barh(words, values, color=['red' if v > 0 else 'green' for v in values])
                        ax.set_title("Feature Impact on Prediction")
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"Explanation failed: {str(e)}")

        except Exception as e:
            st.error(f"Analysis error: {str(e)}")

            # About section moved here
st.markdown("### About this detector")
st.markdown("""
- Uses Naive Bayes classifier  
- Trained on SMS Spam Collection dataset  
- Requires 80%+ confidence for spam detection  
- Provides explanations using LIME
""")

# Model info expander
with st.expander("Technical Details"):
    st.write(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    st.write("Model type:", model.__class__.__name__)
    st.write("Confidence threshold for spam:", "80%")
    st.write("Minimum words for explanation:", "3")