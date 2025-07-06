import pandas as pd
from typing import Dict, List
from backend.model_utils import predict_emotion


def recommend_content(emotion: str, top_n: int = 5) -> Dict[str, List[Dict]]:
    books_df = pd.read_csv("./data/classified_books.csv")
    articles_df = pd.read_csv("./data/classified_articles.csv")

    book_recs = books_df[books_df['emotion'] == emotion].head(top_n)
    article_recs = articles_df[articles_df['emotion'] == emotion].head(top_n)

    return {
        "books": book_recs.to_dict(orient='records'),
        "articles": article_recs.to_dict(orient='records')
    }