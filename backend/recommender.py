import pandas as pd
from typing import Dict, List
from pathlib import Path

data_path = Path("./data")
books_df = pd.read_csv(data_path / "classified_books.csv")
articles_df = pd.read_csv(data_path / "classified_articles.csv")

mood_map = {
'sadness': ['joy', 'surprise'],
'anger': ['love', 'joy'],
'fear': ['love', 'joy'],
'joy': ['joy', 'surprise', 'love'],
'surprise': ['love', 'joy'],
'love': ['joy', 'love', 'surprise']
}



def recommend_content(emotion: str, top_n: int = 5) -> Dict[str, List[Dict]]:
    books_df = pd.read_csv("./data/classified_books.csv")
    articles_df = pd.read_csv("./data/classified_articles.csv")

    book_recs = books_df[books_df['emotion'] == emotion].head(top_n)
    article_recs = articles_df[articles_df['emotion'] == emotion].head(top_n)

    return {
        "books": book_recs.to_dict(orient='records'),
        "articles": article_recs.to_dict(orient='records')
    }
    