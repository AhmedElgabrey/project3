import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text: str) -> str:
    """
    Clean the input text by applying:
    - Lowercasing
    - Removing special characters
    - Tokenization
    - Lemmatization (without POS tagging)
    - Stopword removal
    Returns a single lemmatized string.
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    lemmatized = [
        lemmatizer.lemmatize(word)
        for word in tokens if word not in stop_words
    ]
    return " ".join(lemmatized)