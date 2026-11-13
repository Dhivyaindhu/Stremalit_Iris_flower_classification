# model_utils.py - the "brain" shared by BOTH apps (Streamlit and Gradio).
# Keeping the model code in one file means each app only has to build the screen.
import random
from functools import lru_cache
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

SEED = 42
UNCERTAIN_BELOW = 0.45
IRIS_FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# ---------------------------------------------------------------- sentiment (tiny demo data)
_SUBJECTS = ["The saree", "The fabric", "The delivery", "The packaging", "Customer service",
             "The colour", "The zari work", "The product", "The stitching", "The size"]
_POSITIVE = ["is absolutely beautiful", "was excellent and on time", "exceeded my expectations",
             "is of great quality", "looks even better than the photos",
             "was wonderful, very happy", "is perfect, highly recommended",
             "was fast and friendly", "feels premium and soft"]
_NEGATIVE = ["was terrible", "arrived damaged and late", "is very disappointing",
             "is poor quality, not worth the price", "looks nothing like the photos",
             "was awful, I am very unhappy", "is bad, do not buy", "was slow and rude",
             "feels cheap and rough"]
_EXTRA_POS = ["Loved it, will buy again!", "Great value for money.",
              "Super quality, fast delivery.", "Beautiful design and perfect fit.",
              "Five stars, totally worth it."]
_EXTRA_NEG = ["Waste of money.", "Worst purchase ever.", "Colour faded after one wash.",
              "Very late delivery and no reply from support.", "Poor stitching, returned it."]


def _sentiment_data():
    rows = [(f"{s} {p}.", "positive") for s in _SUBJECTS for p in _POSITIVE]
    rows += [(f"{s} {n}.", "negative") for s in _SUBJECTS for n in _NEGATIVE]
    rows += [(t, "positive") for t in _EXTRA_POS] + [(t, "negative") for t in _EXTRA_NEG]
    random.Random(SEED).shuffle(rows)
    texts, labels = zip(*rows)
    return list(texts), list(labels)


@lru_cache(maxsize=1)
def get_sentiment_model() -> Pipeline:
    """Trains in well under a second, so the demo needs no model file."""
    texts, labels = _sentiment_data()
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(C=10, max_iter=1000, random_state=SEED)),
    ]).fit(texts, labels)


def predict_sentiment(texts: List[str], uncertain_below: float = UNCERTAIN_BELOW) -> List[dict]:
    model = get_sentiment_model()
    results = []
    for row in model.predict_proba(texts):
        best = int(row.argmax())
        conf = float(row[best])
        label = model.classes_[best] if conf >= uncertain_below else "uncertain"
        results.append({"label": label, "confidence": round(conf, 4),
                        "probabilities": {c: round(float(p), 4)
                                          for c, p in zip(model.classes_, row)}})
    return results


# ---------------------------------------------------------------- iris
@lru_cache(maxsize=1)
def iris_dataset() -> pd.DataFrame:
    data = load_iris(as_frame=True)
    df = data.frame.rename(columns=dict(zip(data.feature_names, IRIS_FEATURES)))
    df["species"] = df["target"].map(dict(enumerate(data.target_names)))
    return df.drop(columns="target")


@lru_cache(maxsize=1)
def get_iris_model() -> RandomForestClassifier:
    df = iris_dataset()
    return RandomForestClassifier(n_estimators=100, random_state=SEED).fit(
        df[IRIS_FEATURES].values, df["species"])


def predict_iris(sepal_length, sepal_width, petal_length, petal_width) -> Dict[str, float]:
    model = get_iris_model()
    probs = model.predict_proba(np.array([[sepal_length, sepal_width, petal_length, petal_width]]))[0]
    return {str(c): round(float(p), 4) for c, p in zip(model.classes_, probs)}
