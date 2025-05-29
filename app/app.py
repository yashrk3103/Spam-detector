import joblib, streamlit as st
model = joblib.load('models/spam_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

st.title("Spam/Phishing Message Detector")
msg = st.text_area("Enter your message")
if st.button("Detect"):
    pred = model.predict(vectorizer.transform([msg]))[0]
    st.write("Result:", "🚨 Spam!" if pred else "✅ Ham (Safe)")
