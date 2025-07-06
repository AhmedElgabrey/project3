import pandas as pd
from typing import List, Dict
from .model_utils import predict_emotion

mood_map = {
    'sadness': ['joy', 'surprise'],
    'anger': ['love', 'joy'],
    'fear': ['love', 'joy'],
    'joy': ['joy','surprise', 'love'],
    'surprise': ['love', 'joy'],
    'love': ['joy', 'love', 'surprise']
}

def recommend_books(emotion: str, top_n: int = 5) -> List[Dict]:
    df = pd.read_csv("./data/classified_books.csv")
    return df[df["emotion"].isin(mood_map.get(emotion, []))].head(top_n)[["title", "authors", "emotion"]].to_dict(orient="records")

def recommend_articles(emotion: str, top_n: int = 5) -> List[Dict]:
    df = pd.read_csv("./data/classified_articles.csv")
    return df[df["emotion"].isin(mood_map.get(emotion, []))].head(top_n)[["title", "url", "emotion"]].to_dict(orient="records")

def recommend_content(emotion: str, top_n: int = 5) -> Dict[str, List[Dict]]:
    return {
        "books": recommend_books(emotion, top_n),
        "articles": recommend_articles(emotion, top_n)
    }