from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import os

# Load models
model = joblib.load('models/model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

# Create FastAPI instance
app = FastAPI()

# Input schema
class Message(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "Spam Detector API is up!"}

@app.post("/predict")
async def predict(message: Message):
    try:
        X = vectorizer.transform([message.text])
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0][pred]
        return {
            "prediction": "spam" if pred == 1 else "ham",
            "confidence": f"{proba:.2f}"
        }
    except Exception as e:
        return {"error": str(e)}
