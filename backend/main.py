from backend.model_utils import predict_emotion
from backend.recommender import recommend_content
from backend.daily_emotion_report import generate_daily_emotion_report
from fastapi import FastAPI, HTTPException, Query
from typing import List
from pathlib import Path
import pandas as pd
import joblib

app = FastAPI(
    title="Emotion-Based Recommendation API",
    description="Predict emotion, recommend content, and generate daily reports.",
    version="1.0.0"
)

# Load models
try:
    model = joblib.load("./models/random_forest_model.pkl")
    vectorizer = joblib.load("./models/tfidf_vectorizer.pkl")
    label_encoder = joblib.load("./models/label_encoder.pkl")
except Exception as e:
    raise RuntimeError(f"❌ Error loading models: {e}")

# Load classified content data
data_path = Path("./data")
try:
    books_df = pd.read_csv(data_path / "classified_books.csv")
    articles_df = pd.read_csv(data_path / "classified_articles.csv")
except Exception as e:
    raise RuntimeError(f"❌ Error loading data: {e}")

# ==== Endpoints ====

@app.get("/")
async def root():
    return {"message": "✅ API is running!"}


@app.get("/predict")
def predict_emotion_api(text: str = Query(..., description="Text to analyze")):
    """
    Predict emotion from text (GET)
    """
    try:
        emotion = predict_emotion(text)
        return {"emotion": emotion}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Emotion mapping to suggest positive content
mood_map = {
    'sadness': ['joy', 'surprise'],
    'anger': ['love', 'joy'],
    'fear': ['love', 'joy'],
    'joy': ['joy', 'surprise', 'love'],
    'surprise': ['love', 'joy'],
    'love': ['joy', 'love', 'surprise']
}


@app.get("/recommend/text")
def recommend_by_text(text: str = Query(..., description="Input text to analyze and recommend content")):
    """
    Recommend content based on predicted emotion from text using mood mapping (GET)
    """
    try:
        emotion = predict_emotion(text)

        # Map the predicted emotion to positive target emotions
        related_emotions = mood_map.get(emotion.lower(), [emotion])

        combined_books = []
        combined_articles = []

        for emo in related_emotions:
            result = recommend_content(emo)
            combined_books.extend(result["books"])
            combined_articles.extend(result["articles"])

        # Remove duplicates
        unique_books = {book["title"]: book for book in combined_books}.values()
        unique_articles = {article["title"]: article for article in combined_articles}.values()

        return {
            "input_text": text,
            "predicted_emotion": emotion,
            "recommendations": {
                "books": list(unique_books),
                "articles": list(unique_articles)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend/emotion")
def recommend_by_emotion_api(emotion: str = Query(..., description="User emotion (e.g. joy, sadness, fear)")):
    """
    Recommend content based on input emotion using mood mapping.
    Returns books and articles related to mapped positive emotions.
    """
    try:
        mapped_emotions = mood_map.get(emotion.lower(), [emotion])

        combined_books = []
        combined_articles = []

        for emo in mapped_emotions:
            result = recommend_content(emo)
            combined_books.extend(result["books"])
            combined_articles.extend(result["articles"])

        unique_books = {book["title"]: book for book in combined_books}.values()
        unique_articles = {article["title"]: article for article in combined_articles}.values()

        return {
            "recommendations": {
                "books": list(unique_books),
                "articles": list(unique_articles)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/daily-report")
def daily_report_api(posts: List[str] = Query(..., description="List of daily posts")):
    """
    Generate a daily emotion report from a list of posts (GET)
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
