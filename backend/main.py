from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from pathlib import Path
import pandas as pd
import joblib

from backend.model_utils import predict_emotion
from backend.recommender import recommend_content
from backend.daily_emotion_report import generate_daily_emotion_report

app = FastAPI(
    title="Emotion-Based Recommendation API",
    description="Predict emotion, recommend content, and generate daily reports.",
    version="1.0.0"
)

# تحميل الأدوات
try:
    model = joblib.load("./models/random_forest_model.pkl")
    vectorizer = joblib.load("./models/tfidf_vectorizer.pkl")
    label_encoder = joblib.load("./models/label_encoder.pkl")
except Exception as e:
    raise RuntimeError(f"❌ Error loading models: {e}")

# تحميل البيانات المصنفة
DATA_PATH = Path("./data")
try:
    books_df = pd.read_csv(DATA_PATH / "classified_books.csv")
    articles_df = pd.read_csv(DATA_PATH / "classified_articles.csv")
except Exception as e:
    raise RuntimeError(f"❌ Error loading data: {e}")

# ==== Data Models ====
class TextInput(BaseModel):
    text: str

class DailyPostsInput(BaseModel):
    posts: List[str]

# ==== Endpoints ====

@app.get("/")
async def root():
    return {"message": "✅ API is running!"}


@app.post("/predict")
def predict_emotion_api(data: TextInput):
    """
    توقع المشاعر من نص
    """
    try:
        emotion = predict_emotion(data.text)
        return {"emotion": emotion}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend")
def recommend_based_on_text(data: TextInput):
    """
    توصية محتوى بناءً على النص بعد التنبؤ بالمشاعر
    """
    try:
        emotion = predict_emotion(data.text)
        recommendations = recommend_content(emotion)
        return {
            "predicted_emotion": emotion,
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/daily-report")
def daily_report_api(data: DailyPostsInput):
    """
    إنشاء تقرير يومي للمشاعر من مجموعة منشورات
    """
    try:
        report = generate_daily_emotion_report(
            posts=data.posts,
            model=model,
            vectorizer=vectorizer,
            label_encoder=label_encoder
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/emotions")
async def get_available_emotions():
    """
    عرض المشاعر المتاحة
    """
    try:
        return {"emotions": label_encoder.classes_.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
