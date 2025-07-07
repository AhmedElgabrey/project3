import requests
import joblib
from backend.daily_emotion_report import generate_daily_emotion_report

# Load model artifacts
model = joblib.load("./models/random_forest_model.pkl")
vectorizer = joblib.load("./models/tfidf_vectorizer.pkl")
label_encoder = joblib.load("./models/label_encoder.pkl")

# Example daily posts
posts = [
    "I'm feeling very anxious about tomorrow.",
    "Today was great, I went to the park and felt amazing!",
    "I feel so alone and sad.",
    "That documentary was surprisingly inspiring.",
    "Life feels beautiful and full of love."
]

# Run the report
report = generate_daily_emotion_report(
    posts=posts,
    model=model,
    vectorizer=vectorizer,
    label_encoder=label_encoder
)

print(report)



