# 📚 Emotion-Based Content Recommendation API

This project is an AI-powered backend system for recommending books and articles based on the detected emotion from user-provided text. It also generates a daily emotional impact report based on user posts.

---

## 🔧 Features

* **Emotion Detection** using a trained Random Forest classifier.
* **Content Recommendation** based on predicted emotion.
* **Daily Emotion Report** summarizing user emotional trends and content suggestions.
* **Preprocessing Pipeline** for clean and meaningful feature extraction.
* **RESTful API** built with FastAPI.

---

## 🚀 How It Works

1. **Text Input** ➜ cleaned & lemmatized.
2. **TF-IDF Vectorization** ➜ converts text into features.
3. **Emotion Prediction** ➜ model predicts emotion label.
4. **Recommendation System** ➜ maps emotion to positive content.
5. **Daily Report Generator** ➜ analyzes batch posts and returns a JSON report + chart.

---

## 📂 Folder Structure

```
project/
├── backend/
│   ├── main.py                  # API Entry point
│   ├── model_utils.py           # Emotion prediction logic
│   ├── recommender.py           # Book/article recommendation
│   ├── daily_emotion_report.py  # Daily report generator
│   └── preprocessing.py         # Text cleaning & lemmatization
├── models/                      # Trained model & vectorizer
├── data/                        # Classified content datasets
├── daily_report.json            # Daily report output
├── daily_emotion_chart.png      # Report visualization
└── requirements.txt             # Python dependencies
```

---

## 🔌 API Endpoints

* `POST /predict` → Predict emotion for a single text
* `POST /recommend` → Recommend content based on user text
* `POST /daily-report` → Generate report from list of posts
* `POST /recommend/content` → Recommend from list of emotions (if needed)
* `GET /emotions` → Return all supported emotion labels

---

## 🛠️ Setup Instructions

```bash
# 1. Clone the project
$ git clone https://github.com/your-username/project2.git
$ cd project2

# 2. Create a virtual environment
$ python -m venv venv
$ source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
$ pip install -r requirements.txt

# 4. Run the API
$ uvicorn backend.main:app --reload
```

---

## 🧠 Technologies Used

* Python
* FastAPI
* Scikit-learn
* NLTK
* Pandas / NumPy
* SMOTE (imbalanced-learn)
* Seaborn / Matplotlib

---

## 👨‍💻 Author

Ahmed Elgabrey – [GitHub](https://github.com/AhmedElgabrey)

Feel free to contribute, suggest improvements, or fork the project!
