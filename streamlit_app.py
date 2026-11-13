# streamlit_app.py - run locally with:  streamlit run streamlit_app.py
import os

import pandas as pd
import requests
import streamlit as st

from model_utils import IRIS_FEATURES, iris_dataset, predict_iris, predict_sentiment

st.set_page_config(page_title="ML Demo Hub", layout="wide")


# ---------- helpers ------------------------------------------------------------------------
def get_secret(name: str, default: str) -> str:
    """st.secrets on Streamlit Cloud, environment variable or default when running locally."""
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)


@st.cache_resource                      # runs ONCE, shared by all users and all reruns
def load_models():
    from model_utils import get_iris_model, get_sentiment_model
    get_sentiment_model()
    get_iris_model()
    return True


def run_sentiment(texts, source, api_url, threshold):
    if source == "Built-in model":
        return predict_sentiment(texts, uncertain_below=threshold)
    reply = requests.post(f"{api_url}/v1/predict/batch", json={"texts": texts},
                          headers={"X-API-Key": get_secret("API_KEY", "demo-key")}, timeout=10)
    reply.raise_for_status()
    return reply.json()["results"]


# ---------- sidebar ------------------------------------------------------------------------
load_models()
with st.sidebar:
    st.header("Settings")
    source = st.radio("Prediction source", ["Built-in model", "ReviewSense API (Day 35)"])
    api_url = "http://127.0.0.1:8000"
    if source != "Built-in model":
        api_url = st.text_input("API URL", api_url)
        st.caption("The key is read from st.secrets['API_KEY'] (default: demo-key).")
    threshold = st.slider("Say 'uncertain' below", 0.50, 0.90, 0.65, 0.05,
                          disabled=(source != "Built-in model"))
    st.divider()
    st.caption("Day 36 demo. Tiny teaching model: not for production use.")

st.title("ML Demo Hub")
st.write("One small app, three ways to show a model: a text box, sliders, and a file upload.")
tab1, tab2, tab3 = st.tabs(["Sentiment", "Iris explorer", "Batch CSV"])

# ---------- tab 1: sentiment ---------------------------------------------------------------
with tab1:
    examples = {"Positive": "The saree is absolutely beautiful and the delivery was fast.",
                "Negative": "The product arrived damaged and customer service was rude.",
                "Unsure": "It is okay I guess."}
    cols = st.columns(len(examples))
    for col, (name, sample) in zip(cols, examples.items()):
        if col.button(f"Try: {name}"):
            st.session_state["review"] = sample             # fill the text box

    text = st.text_area("Review text", key="review", height=120, max_chars=1000)
    if st.button("Analyse", type="primary"):
        if not text.strip():
            st.warning("Please type a review first.")
        else:
            try:
                with st.spinner("Predicting..."):
                    result = run_sentiment([text.strip()], source, api_url, threshold)[0]
            except requests.RequestException as err:
                st.error(f"Could not reach the API: {err}")
            else:
                c1, c2 = st.columns(2)
                c1.metric("Label", result["label"])
                c2.metric("Confidence", f"{result['confidence']:.1%}")
                st.bar_chart(pd.Series(result["probabilities"]))
                st.session_state.setdefault("history", []).append(
                    {"review": text.strip(), "label": result["label"],
                     "confidence": result["confidence"]})

    if st.session_state.get("history"):                     # session_state survives reruns
        st.subheader("History (this browser session)")
        history = pd.DataFrame(st.session_state["history"])
        st.dataframe(history)
        st.download_button("Download history (CSV)", history.to_csv(index=False),
                           "history.csv", "text/csv")

# ---------- tab 2: iris --------------------------------------------------------------------
with tab2:
    data = iris_dataset()
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Measure a flower")
        values = [st.slider(name.replace("_", " ") + " (cm)", float(data[name].min()),
                            float(data[name].max()), float(data[name].mean()), 0.1)
                  for name in IRIS_FEATURES]
        probs = predict_iris(*values)
        st.metric("Predicted species", max(probs, key=probs.get))
        st.bar_chart(pd.Series(probs))
    with right:
        st.subheader("Where does your flower sit?")
        plot = data[["petal_length", "petal_width", "species"]].copy()
        plot.loc[len(plot)] = [values[2], values[3], "your flower"]
        st.scatter_chart(plot, x="petal_length", y="petal_width", color="species")

# ---------- tab 3: batch csv ---------------------------------------------------------------
with tab3:
    st.write("Upload a CSV file that has a column named **review** (max 500 rows).")
    sample_df = pd.DataFrame({"review": list(examples.values()) + ["Great value for money, loved it!"]})
    st.download_button("Download a sample CSV", sample_df.to_csv(index=False),
                       "sample_reviews.csv", "text/csv")
    upload = st.file_uploader("CSV file", type="csv")
    if upload is not None:
        table = pd.read_csv(upload)
        if "review" not in table.columns:
            st.error("The file needs a column named 'review'.")
        elif len(table) > 500:
            st.error("Please upload at most 500 rows.")
        else:
            with st.spinner(f"Scoring {len(table)} reviews..."):
                scored = run_sentiment(table["review"].astype(str).tolist(), source, api_url, threshold)
            out = table.assign(label=[r["label"] for r in scored],
                               confidence=[r["confidence"] for r in scored])
            st.dataframe(out)
            st.bar_chart(out["label"].value_counts())
            st.download_button("Download results (CSV)", out.to_csv(index=False),
                               "predictions.csv", "text/csv")
