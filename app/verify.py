from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib

# Example dataset
texts = ["win money now", "hello how are you", "free lottery", "meeting tomorrow"]
labels = [1, 0, 1, 0]  # 1=spam, 0=ham

# Train vectorizer & model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = SVC(probability=True)
model.fit(X, labels)

# Save for later
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(model, 'spam_model.pkl')

# Load and predict
loaded_vectorizer = joblib.load('vectorizer.pkl')
loaded_model = joblib.load('spam_model.pkl')

new_msg = ["free prize"]
X_new = loaded_vectorizer.transform(new_msg)
pred = loaded_model.predict(X_new)

print("Prediction:", "Spam" if pred[0] == 1 else "Ham")