import streamlit as st
import joblib
import pandas as pd
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components
import os
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Spam Detector with Explainability", page_icon="📩")

st.title("📩 Email/SMS Spam Detector with LIME Explanation")

# Load model and vectorizer
model = joblib.load('models/model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

# Initialize LIME explainer
explainer = LimeTextExplainer(class_names=['ham', 'spam'])
predict_fn = lambda x: model.predict_proba(vectorizer.transform(x))

# Input area
user_input = st.text_area("Enter the message to classify:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Basic prediction
        prediction = model.predict(vectorizer.transform([user_input]))[0]
        proba = model.predict_proba(vectorizer.transform([user_input]))[0]

        st.subheader("🔍 Prediction")
        st.write(f"**Class:** {prediction}")
        st.write(f"**Confidence (Ham):** {proba[0]*100:.2f}%")
        st.write(f"**Confidence (Spam):** {proba[1]*100:.2f}%")

        st.subheader("🧠 Why did the model make this decision?")
        
        # Only generate explanation if text is long enough
        if len(user_input.split()) < 3:  # Adjust this threshold as needed
            st.warning("Explanation requires at least 3 words of text.")
        else:
            with st.spinner("Generating explanation..."):
                try:
                    # Generate explanation
                    explanation = explainer.explain_instance(
                        user_input, 
                        predict_fn, 
                        num_features=6
                    )
                    
                    # Show top features
                    weights = explanation.as_list()
                    st.markdown("### 🔠 Top Words Influencing the Prediction:")
                    for word, weight in weights:
                        label = "Spam" if weight > 0 else "Ham"
                        st.write(f"**{word}** → `{label}` with weight `{weight:.4f}`")

                    # Bar chart
                    st.markdown("### 📊 Feature Impact Chart:")
                    words = [w for w, _ in weights]
                    values = [v for _, v in weights]
                    fig, ax = plt.subplots()
                    bars = ax.barh(
                        words, 
                        values, 
                        color=['orange' if v > 0 else 'skyblue' for v in values]
                    )
                    ax.set_xlabel("Impact on Prediction")
                    ax.set_ylabel("Word")
                    ax.invert_yaxis()
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Could not generate explanation: {str(e)}")
                    st.info("Try entering a longer message for explanation.")