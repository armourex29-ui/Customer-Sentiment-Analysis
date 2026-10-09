import streamlit as st
import pandas as pd
import plotly.express as px
from utils import (
    train_models, predict_sentiment, analyze_batch, load_sample_data,
    sentiment_color
)

st.set_page_config(
    page_title="Customer Sentiment Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Lightweight styling
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #f7f9fc 0%, #eef3fb 100%); }
    .hero {
        padding: 1.5rem 1.8rem; border-radius: 18px;
        background: linear-gradient(110deg, #172554, #2563eb);
        color: white; margin-bottom: 1rem;
    }
    .hero h1 { color: white; margin: 0; }
    .hero p { color: #e0e7ff; margin-bottom: 0; }
    .metric-card {
        background: white; border: 1px solid #dbeafe; border-radius: 14px;
        padding: 1rem; box-shadow: 0 3px 12px rgba(15, 23, 42, .05);
    }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #dbeafe; padding: 14px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_models():
    return train_models()

models, vectorizer, training_info = get_models()

with st.sidebar:
    st.title("⚙️ Analyzer settings")
    model_name = st.selectbox(
        "Machine learning model",
        ["Multinomial Naive Bayes", "Logistic Regression", "Linear SVM"],
        help="All models are trained on the built-in demonstration dataset when the app starts.",
    )
    st.caption("Models use TF-IDF text features. No external API is required.")
    st.divider()
    st.markdown("**Project workflow**")
    st.markdown("1. Enter or upload feedback\n2. Clean and vectorize text\n3. Predict sentiment\n4. Explore the results")

st.markdown("""
<div class="hero">
  <h1>💬 Customer Sentiment Analysis</h1>
  <p>Understand customer feedback with Natural Language Processing and Machine Learning.</p>
</div>
""", unsafe_allow_html=True)

tab_single, tab_batch, tab_about = st.tabs(
    ["✍️ Analyze feedback", "📂 Batch analysis", "📊 About the model"]
)

with tab_single:
    st.subheader("Analyze one customer review")
    review = st.text_area(
        "Customer feedback",
        placeholder="Example: The delivery was quick and the product quality is excellent!",
        height=130,
    )
    if st.button("Analyze sentiment", type="primary", use_container_width=False):
        if not review.strip():
            st.warning("Please enter a review first.")
        else:
            result = predict_sentiment(review, model_name, models, vectorizer)
            sentiment = result["sentiment"]
            confidence = result["confidence"]
            emoji = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}.get(sentiment, "💬")
            st.markdown(f"### {emoji} {sentiment} sentiment")
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted sentiment", sentiment)
            c2.metric("Model confidence", f"{confidence:.1%}")
            c3.metric("Characters analyzed", len(review))
            st.progress(min(max(confidence, 0.0), 1.0))
            st.caption(
                "Confidence is the model's estimated class probability when available; "
                "it is not a guarantee of correctness."
            )

with tab_batch:
    st.subheader("Analyze multiple reviews")
    st.write("Upload a CSV containing a column named `text`, `review`, `feedback`, or `customer_review`.")
    uploaded = st.file_uploader("Upload customer reviews (.csv)", type=["csv"])
    sample_col1, sample_col2 = st.columns([1, 3])
    with sample_col1:
        if st.button("Load sample reviews"):
            st.session_state["sample_reviews"] = load_sample_data()
    if "sample_reviews" in st.session_state:
        batch_df = st.session_state["sample_reviews"].copy()
    elif uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            batch_df = pd.DataFrame()
    else:
        batch_df = pd.DataFrame()

    if not batch_df.empty:
        st.dataframe(batch_df.head(100), use_container_width=True)
        text_candidates = [c for c in batch_df.columns if c.lower().strip() in
                           {"text", "review", "feedback", "customer_review", "comment"}]
        if not text_candidates:
            st.error("No review text column found. Rename one column to `text` or `review`.")
        else:
            text_col = st.selectbox("Choose review text column", text_candidates)
            if st.button("Analyze all reviews", type="primary"):
                results_df = analyze_batch(batch_df, text_col, model_name, models, vectorizer)
                st.session_state["batch_results"] = results_df
            if "batch_results" in st.session_state:
                results_df = st.session_state["batch_results"]
                st.success(f"Analyzed {len(results_df)} reviews.")
                st.dataframe(results_df, use_container_width=True)
                counts = results_df["sentiment"].value_counts().rename_axis("Sentiment").reset_index(name="Count")
                left, right = st.columns(2)
                with left:
                    fig = px.pie(counts, names="Sentiment", values="Count",
                                 title="Sentiment distribution", hole=0.45,
                                 color="Sentiment",
                                 color_discrete_map={"Positive": "#16a34a", "Negative": "#dc2626", "Neutral": "#64748b"})
                    st.plotly_chart(fig, use_container_width=True)
                with right:
                    fig2 = px.bar(counts, x="Sentiment", y="Count", title="Review counts",
                                  color="Sentiment",
                                  color_discrete_map={"Positive": "#16a34a", "Negative": "#dc2626", "Neutral": "#64748b"})
                    st.plotly_chart(fig2, use_container_width=True)
                csv = results_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download results as CSV", csv,
                                   file_name="sentiment_analysis_results.csv",
                                   mime="text/csv")
    else:
        st.info("Upload a CSV or click **Load sample reviews** to try batch analysis.")

with tab_about:
    st.subheader("How the system works")
    st.markdown("""
    1. **Input:** customer reviews entered manually or uploaded as CSV.
    2. **Preprocessing:** lowercase conversion and basic text normalization.
    3. **Feature extraction:** TF-IDF converts text into numerical features.
    4. **Classification:** choose Naive Bayes, Logistic Regression, or Linear SVM.
    5. **Output:** Positive, Negative, or Neutral sentiment and visual summaries.
    """)
    st.warning(
        "Academic demo note: the app trains on a small built-in illustrative dataset. "
        "Its predictions are for demonstration, not a production-quality benchmark. "
        "For a serious evaluation, train and test on a larger, representative labelled dataset."
    )
    st.markdown("**Algorithms included**")
    st.markdown("- Multinomial Naive Bayes\n- Logistic Regression\n- Linear Support Vector Machine (SVM)")
    st.markdown("**Training-set metrics**")
    st.json(training_info)
