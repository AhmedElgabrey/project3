from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os
from model_utils import predict
from recommender import recommend_content
from daily_emotion_report import generate_daily_emotion_report

# إنشاء تطبيق FastAPI
app = FastAPI()

# إعداد CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://myfrontend.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("./models/random_forest_model.pkl")
tfidf = joblib.load("./models/tfidf_vectorizer.pkl")
le = joblib.load("./models/label_encoder.pkl")

# Load content data
DATA_PATH = os.Path("./data")
try:
    books_df = pd.read_csv(DATA_PATH / "classified_books.csv")
    articles_df = pd.read_csv(DATA_PATH / "classified_articles.csv")
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# ====== Data Models ======
class TextInput(BaseModel):
    text: str

class DailyPostsInput(BaseModel):
    posts: List[str]

# ====== Endpoints ======

@app.post("/predict")
def predict_emotion_endpoint(data: TextInput):
    emotion = predict(data.text)
    return {"emotion": emotion}

@app.post("/recommend")
def recommend_endpoint(data: TextInput):
    """
    Get recommendations based on emotions
    """
    try:
        # Filter content based on emotions
        book_recommendations = books_df[books_df['emotion'].isin(request.emotions)].head(request.num_recommendations)
        article_recommendations = articles_df[articles_df['emotion'].isin(request.emotions)].head(request.num_recommendations)
        
        return {
            "books": book_recommendations.to_dict(orient='records'),
            "articles": article_recommendations.to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/daily-report", response_model=Dict[str, Any])
def daily_report_api(data: DailyPostsInput) -> Dict[str, Any]:
    """
    Generate a daily emotion report from a list of posts.
    """
    try:
        report: Dict[str, Any] = generate_daily_emotion_report(
            posts=data.posts,
            model=model,
            vectorizer=tfidf,
            label_encoder=le
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
