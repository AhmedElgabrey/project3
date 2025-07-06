from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from backend.preprocessing import preprocess_text
import joblib

# Load once here
model: BaseEstimator = joblib.load("./models/random_forest_model.pkl")
vectorizer: TfidfVectorizer = joblib.load("./models/tfidf_vectorizer.pkl")
label_encoder: LabelEncoder = joblib.load("./models/label_encoder.pkl")

def predict_emotion(text: str) -> str:
    """
    توقع المشاعر من نص واحد باستخدام نموذج مُحمّل مسبقاً.
    """
    processed = preprocess_text(text)
    vect_text = vectorizer.transform([processed])
    prediction = model.predict(vect_text)
    return label_encoder.inverse_transform(prediction)[0]
