"""Utilities for the Customer Sentiment Analysis Streamlit application."""
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Small, illustrative seed dataset so the demo runs without downloading data.
# Replace/extend this with a properly labelled dataset for meaningful evaluation.
TRAINING_DATA = [
    ("Absolutely love this product, it works perfectly", "Positive"),
    ("Excellent quality and fast delivery", "Positive"),
    ("The support team was helpful and friendly", "Positive"),
    ("Very happy with my purchase", "Positive"),
    ("Great value for money and easy to use", "Positive"),
    ("The product exceeded my expectations", "Positive"),
    ("Wonderful experience, I would buy again", "Positive"),
    ("Five stars, highly recommended", "Positive"),
    ("Quick shipping and fantastic packaging", "Positive"),
    ("The app is smooth and works great", "Positive"),
    ("Good quality, good price, very satisfied", "Positive"),
    ("Customer service solved my issue quickly", "Positive"),
    ("Terrible quality, it broke on the first day", "Negative"),
    ("Delivery was late and the package was damaged", "Negative"),
    ("Very disappointed with this purchase", "Negative"),
    ("The support team was rude and unhelpful", "Negative"),
    ("Waste of money, do not recommend", "Negative"),
    ("It stopped working after two days", "Negative"),
    ("Poor experience and frustrating service", "Negative"),
    ("The product is faulty and overpriced", "Negative"),
    ("I want a refund, this is awful", "Negative"),
    ("Slow shipping and terrible packaging", "Negative"),
    ("The app crashes constantly and is unusable", "Negative"),
    ("Not worth the price, very bad quality", "Negative"),
    ("The product arrived today", "Neutral"),
    ("I purchased this item last week", "Neutral"),
    ("The package contains two pieces", "Neutral"),
    ("The product is available in three sizes", "Neutral"),
    ("Customer support is open from nine to five", "Neutral"),
    ("My order number is 12345", "Neutral"),
    ("The delivery was scheduled for Monday", "Neutral"),
    ("This item is black and weighs one kilogram", "Neutral"),
    ("I contacted support about my account", "Neutral"),
    ("The product has a one year warranty", "Neutral"),
    ("I received the replacement item yesterday", "Neutral"),
    ("The order status says shipped", "Neutral"),
]

def clean_text(text):
    """Normalize review text while retaining words and useful punctuation boundaries."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def train_models():
    """Fit the three demo classifiers and return fitted models plus simple metrics."""
    frame = pd.DataFrame(TRAINING_DATA, columns=["text", "sentiment"])
    X = frame["text"].map(clean_text)
    y = frame["sentiment"]

    # Train on the full tiny demo set for dependable predictions in the app.
    # Report these as training metrics only; they are not held-out test scores.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    X_vec = vectorizer.fit_transform(X)

    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.35),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=42),
    }
    metrics = {}
    for name, model in models.items():
        model.fit(X_vec, y)
        preds = model.predict(X_vec)
        metrics[name] = {
            "training_accuracy": round(float(accuracy_score(y, preds)), 3),
            "training_samples": len(frame),
            "note": "Training-set score only; not a measure of generalization."
        }
    return models, vectorizer, metrics

def _confidence(model, vectorized):
    """Return a confidence-like score where supported; SVM uses a softmax of decision scores."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vectorized)[0]
        return float(max(probabilities))
    if hasattr(model, "decision_function"):
        import numpy as np
        scores = model.decision_function(vectorized)
        scores = np.asarray(scores).ravel()
        if len(scores) == 1:
            return float(1.0 / (1.0 + np.exp(-abs(scores[0]))))
        scores = scores - scores.max()
        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum()
        return float(probs.max())
    return 0.0

def predict_sentiment(text, model_name, models, vectorizer):
    """Predict one review's sentiment."""
    cleaned = clean_text(text)
    if not cleaned:
        return {"sentiment": "Neutral", "confidence": 0.0}
    vectorized = vectorizer.transform([cleaned])
    model = models[model_name]
    sentiment = str(model.predict(vectorized)[0])
    confidence = _confidence(model, vectorized)
    return {"sentiment": sentiment, "confidence": confidence}

def analyze_batch(dataframe, text_column, model_name, models, vectorizer):
    """Classify all reviews in a DataFrame and return a results DataFrame."""
    result = dataframe.copy()
    predictions, confidences = [], []
    for value in result[text_column].fillna("").astype(str):
        prediction = predict_sentiment(value, model_name, models, vectorizer)
        predictions.append(prediction["sentiment"])
        confidences.append(round(prediction["confidence"], 4))
    result["sentiment"] = predictions
    result["confidence"] = confidences
    return result

def load_sample_data():
    """Return small sample reviews for the batch-analysis demo."""
    return pd.DataFrame({
        "review": [
            "The product is fantastic and delivery was fast.",
            "I am unhappy with the quality and want a refund.",
            "The order arrived on Tuesday.",
            "Customer support was friendly and solved my problem.",
            "The app keeps crashing. Very disappointing.",
            "The package contains one charger and a cable.",
            "Good value for money, I recommend it.",
            "Shipping was late and the box was damaged.",
            "I bought this product yesterday.",
        ]
    })

def sentiment_color(sentiment):
    """Return a simple display color for a sentiment label."""
    return {"Positive": "#16a34a", "Negative": "#dc2626", "Neutral": "#64748b"}.get(sentiment, "#334155")
