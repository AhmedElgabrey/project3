
import joblib
import os
from .preprocessing import preprocess_text

class EmotionModel:
    def __init__(self, model_path: str, vectorizer_path: str):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.label_encoder = joblib.load(label_encoder_path)

    def predict_emotion(self, text: str) -> str:
        cleaned_text = preprocess_text(text)
        vec = self.vectorizer.transform([cleaned_text])
        
        if vec.shape[1] != self.model.n_features_in_:
            raise ValueError(
                f"Vectorizer produced {vec.shape[1]} features, but model expects {self.model.n_features_in_}"
            )
        
        pred = self.model.predict(vec)
        return self.label_encoder.inverse_transform(pred)[0]


model_path = os.path.join("models", "random_forest_model.pkl")
vectorizer_path = os.path.join("models", "tfidf_vectorizer.pkl")
label_encoder_path = os.path.join("models", "label_encoder.pkl")
emotion_model = EmotionModel(model_path, vectorizer_path)

def predict(text: str) -> str:
    """تحليل المشاعر لنص معين"""
    return emotion_model.predict_emotion(text)
