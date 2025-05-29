from lime.lime_text import LimeTextExplainer
import joblib, pandas as pd
model = joblib.load('models/spam_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

df = pd.read_csv('data/spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
explainer = LimeTextExplainer(class_names=['ham', 'spam'])
predict_fn = lambda x: model.predict_proba(vectorizer.transform(x))
exp = explainer.explain_instance(df['message'][10], predict_fn, num_features=6)
exp.save_to_file('explainability/explanation.html')
