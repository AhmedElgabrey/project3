from fastapi import FastAPI, HTTPException, Query
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

# ==== Endpoints ====

@app.get("/")
async def root():
    return {"message": "✅ API is running!"}


@app.get("/predict")
def predict_emotion_api(text: str = Query(..., description="Text to analyze")):
    """
    توقع المشاعر من نص (GET)
    """
    try:
        emotion = predict_emotion(text, model, vectorizer, label_encoder)
        return {"emotion": emotion}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend")
def recommend_based_on_text(text: str = Query(..., description="Text to analyze and recommend content")):
    """
    توصية محتوى بناءً على النص بعد التنبؤ بالمشاعر (GET)
    """
    try:
        emotion = predict_emotion(text, model, vectorizer, label_encoder)
        recommendations = recommend_content(emotion)
        return {
            "predicted_emotion": emotion,
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/daily-report")
def daily_report_api(posts: List[str] = Query(..., description="List of daily posts")):
    """
    إنشاء تقرير يومي للمشاعر من مجموعة منشورات (GET)
    """
    try:
        report = generate_daily_emotion_report(
            posts=posts,
            model=model,
            vectorizer=vectorizer,
            label_encoder=label_encoder
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
