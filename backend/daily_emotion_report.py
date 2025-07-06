import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import date
import json
from typing import List, Dict
from backend.model_utils import predict_emotion
from backend.recommender import recommend_content

def generate_daily_emotion_report(posts: List[str], model, vectorizer, label_encoder) -> Dict:
    """
    إنشاء تقرير يومي عن المشاعر المستخرجة من منشورات المستخدم.
    """
    emotions = [predict_emotion(p, model, vectorizer, label_encoder) for p in posts]
    emotion_counts = dict(Counter(emotions))
    total = sum(emotion_counts.values())
    emotion_distribution = {
        emo: round((count / total) * 100, 2) for emo, count in emotion_counts.items()
    }

    # رسم بياني
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(emotion_distribution.keys()), y=list(emotion_distribution.values()), palette='Set2')
    plt.title("توزيع المشاعر اليومي")
    plt.xlabel("المشاعر")
    plt.ylabel("النسبة (%)")
    plt.tight_layout()
    plt.savefig("daily_emotion_chart.png")
    plt.close()

    # المنشورات السلبية
    negative_emotions = ['anger', 'sadness', 'fear']
    top_negative = [
        {"text": text, "emotion": emo}
        for text, emo in zip(posts, emotions) if emo in negative_emotions
    ][:3]

    dominant_negative = top_negative[0]['emotion'] if top_negative else 'joy'
    recommendations = recommend_content(dominant_negative)

    report = {
        "date": str(date.today()),
        "total_posts": total,
        "emotion_distribution": emotion_distribution,
        "top_negative_posts": top_negative,
        "recommendations": recommendations
    }

    with open("daily_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

    return report