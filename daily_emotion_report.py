from typing import List, Dict
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import date
from .model_utils import predict_emotion
from .recommender import recommend_content

def generate_daily_emotion_report(posts: List[str], model, vectorizer, label_encoder) -> Dict:
    emotions = [predict_emotion(p, model, vectorizer, label_encoder) for p in posts]
    emotion_counts = dict(Counter(emotions))
    total = sum(emotion_counts.values())
    emotion_distribution = {e: round(c / total * 100, 2) for e, c in emotion_counts.items()}

    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(emotion_distribution.keys()), y=list(emotion_distribution.values()), palette='Set2')
    plt.title("توزيع المشاعر اليومي")
    plt.ylabel("%")
    plt.xlabel("المشاعر")
    plt.tight_layout()
    plt.savefig("daily_emotion_chart.png")
    plt.close()

    negative_emotions = ['anger', 'sadness', 'fear']
    top_negative = [(text, emo) for text, emo in zip(posts, emotions) if emo in negative_emotions][:3]

    dominant_negative = top_negative[0][1] if top_negative else "joy"
    recommendations = recommend_content(dominant_negative)

    report = {
        "date": str(date.today()),
        "total_posts": total,
        "emotion_distribution": emotion_distribution,
        "top_negative_posts": [{"text": t, "emotion": e} for t, e in top_negative],
        "recommendations": recommendations
    }

    with open("daily_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    return report
