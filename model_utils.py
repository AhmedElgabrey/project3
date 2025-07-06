from typing import List, Dict
import joblib
import os

class EmotionModel:
    def __init__(self, model_path: str, vectorizer_path: str):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict_emotion(self, text: str) -> str:
        # تنظيف النص أولاً
        cleaned_text = preprocess_text(text)
        vec = self.vectorizer.transform([cleaned_text])
        
        # تحقق من تطابق الخصائص
        if vec.shape[1] != self.model.n_features_in_:
            raise ValueError(
                f"Vectorizer produced {vec.shape[1]} features, but model expects {self.model.n_features_in_}"
            )
        
        pred = self.model.predict(vec)
        return pred[0]


# تحميل موديلات
model_path = os.path.join("models", "random_forest_model.pkl")
vectorizer_path = os.path.join("models", "tfidf_vectorizer.pkl")
emotion_model = EmotionModel(model_path, vectorizer_path)

def predict(text: str) -> str:
    """تحليل المشاعر لنص معين"""
    return emotion_model.predict_emotion(text)
