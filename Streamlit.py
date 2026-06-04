# ============================================================
# STREAMLIT APP: IMDB SENTIMENT ANALYSIS USING WORD2VEC + GRU
# ============================================================

import re
import joblib
import pandas as pd
import streamlit as st

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# CONFIGURATION
# ============================================================

MAX_LEN = 200

GRU_MODEL_PATH = "best_word2vec_gru_model.keras"
TOKENIZER_PATH = "tokenizer.pkl"


# ============================================================
# TEXT CLEANING FUNCTION
# ============================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# LOAD MODEL AND TOKENIZER
# ============================================================

@st.cache_resource
def load_gru_model():
    return load_model(GRU_MODEL_PATH)


@st.cache_resource
def load_saved_tokenizer():
    return joblib.load(TOKENIZER_PATH)


gru_model = load_gru_model()
tokenizer = load_saved_tokenizer()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_sentiment(texts):
    cleaned_texts = [clean_text(text) for text in texts]

    sequences = tokenizer.texts_to_sequences(cleaned_texts)

    padded = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    probabilities = gru_model.predict(padded).ravel()

    results = []

    for original, prob in zip(texts, probabilities):
        sentiment = "Positive" if prob >= 0.5 else "Negative"
        confidence = prob if sentiment == "Positive" else 1 - prob

        results.append({
            "Review": original,
            "Positive Probability": round(float(prob), 4),
            "Result": sentiment,
            "Confidence": round(float(confidence), 4)
        })

    return pd.DataFrame(results)


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="IMDb Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 IMDb Movie Review Sentiment Analysis")
st.write(
    "This app uses the trained **Word2Vec + GRU** model to classify movie reviews "
    "as **positive** or **negative**."
)

st.sidebar.header("Input Options")

input_mode = st.sidebar.radio(
    "Choose input type:",
    ["Example Reviews", "Manual Text Input", "Upload CSV File", "Model Accuracy"]
)


# ============================================================
# OPTION 1: EXAMPLE REVIEWS
# ============================================================

if input_mode == "Example Reviews":
    st.subheader("Example Movie Reviews")

    example_reviews = [
        "This movie was amazing. The acting was excellent and the story was very enjoyable.",
        "The film was boring and too long. I would not recommend it.",
        "It was not bad, but I expected something more exciting.",
        "The visuals were beautiful, but the plot was weak and confusing.",
        "I loved this movie. It was emotional, funny, and very well directed."
    ]

    for i, review in enumerate(example_reviews, start=1):
        st.write(f"**Review {i}:** {review}")

    if st.button("Predict Example Reviews"):
        output_df = predict_sentiment(example_reviews)

        st.subheader("Prediction Results")
        st.dataframe(output_df, use_container_width=True)


# ============================================================
# OPTION 2: MANUAL TEXT INPUT
# ============================================================

elif input_mode == "Manual Text Input":
    st.subheader("Enter a Movie Review")

    user_review = st.text_area(
        "Type your movie review here:",
        height=180,
        placeholder="Example: The movie was surprisingly good and the acting was excellent..."
    )

    if st.button("Predict Sentiment"):
        if user_review.strip() == "":
            st.warning("Please enter a review first.")
        else:
            output_df = predict_sentiment([user_review])

            st.subheader("Prediction Result")
            st.dataframe(output_df, use_container_width=True)

            sentiment = output_df["Result"].iloc[0]
            confidence = output_df["Confidence"].iloc[0]

            if sentiment == "Positive":
                st.success(f"Prediction: Positive sentiment with {confidence:.2%} confidence.")
            else:
                st.error(f"Prediction: Negative sentiment with {confidence:.2%} confidence.")


# ============================================================
# OPTION 3: UPLOAD CSV FILE
# ============================================================

elif input_mode == "Upload CSV File":
    st.subheader("Upload CSV File")

    st.write(
        "Upload a CSV file containing movie reviews. "
        "The file should have a column named **review** or **text**."
    )

    # ------------------------------------------------------------
    # SAMPLE CSV DOWNLOAD
    # ------------------------------------------------------------

    sample_df = pd.DataFrame({
        "review": [
            "This movie was amazing. The acting was excellent and the story was very enjoyable.",
            "The film was boring and too long. I would not recommend it.",
            "It was not bad, but I expected something more exciting.",
            "The visuals were beautiful, but the plot was weak and confusing.",
            "I loved this movie. It was emotional, funny, and very well directed."
        ]
    })

    sample_csv = sample_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Sample CSV File",
        data=sample_csv,
        file_name="sample_reviews.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader(
        "Upload your CSV file:",
        type=["csv"]
    )

    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)

        st.write("Uploaded file preview:")
        st.dataframe(uploaded_df.head(), use_container_width=True)

        possible_columns = ["review", "text", "Review", "Text"]

        review_column = None
        for col in possible_columns:
            if col in uploaded_df.columns:
                review_column = col
                break

        if review_column is None:
            st.warning("No column named review or text was found. Please select the correct column.")
            review_column = st.selectbox("Select review column:", uploaded_df.columns)

        if st.button("Predict Uploaded Reviews"):
            reviews = uploaded_df[review_column].astype(str).tolist()

            final_df = predict_sentiment(reviews)

            st.subheader("Prediction Results")
            st.dataframe(final_df, use_container_width=True)

            csv_output = final_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Prediction Results as CSV",
                data=csv_output,
                file_name="sentiment_predictions.csv",
                mime="text/csv"
            )


# ============================================================
# OPTION 4: MODEL ACCURACY
# ============================================================

elif input_mode == "Model Accuracy":
    st.subheader("Model Accuracy and Evaluation Results")

    st.write(
        "This page shows the evaluation results and training curves of the trained "
        "**Word2Vec + GRU** model."
    )

    # ------------------------------------------------------------
    # INTERPRETATION / FINAL METRICS
    # ------------------------------------------------------------

    try:
        metrics_df = pd.read_csv("model_metrics.csv")

        accuracy = metrics_df["Accuracy"].iloc[0]
        precision = metrics_df["Precision"].iloc[0]
        recall = metrics_df["Recall"].iloc[0]
        f1 = metrics_df["F1-score"].iloc[0]

        st.write("### Interpretation")
        st.write(f"**Accuracy:** {accuracy:.4f}")
        st.write(f"**Precision:** {precision:.4f}")
        st.write(f"**Recall:** {recall:.4f}")
        st.write(f"**F1-score:** {f1:.4f}")

        st.info(
            "The F1-score is the main evaluation metric because it balances precision and recall. "
            "This is useful for sentiment classification because it gives a clearer view of how well "
            "the model performs across both positive and negative reviews."
        )

    except FileNotFoundError:
        st.warning(
            "model_metrics.csv was not found. Please save the model metrics from the notebook "
            "and place the file in the same folder as Streamlit.py."
        )
    # ------------------------------------------------------------
    # TRAINING CURVES
    # ------------------------------------------------------------

    try:
        history_df = pd.read_csv("gru_training_history.csv")

        col1, col2 = st.columns(2)

        with col1:
            st.write("### Accuracy Curve")

            accuracy_chart_df = history_df.set_index("Epoch")[
                ["Training Accuracy", "Validation Accuracy"]
            ]

            st.line_chart(accuracy_chart_df)

        with col2:
            st.write("### Loss Curve")

            loss_chart_df = history_df.set_index("Epoch")[
                ["Training Loss", "Validation Loss"]
            ]

            st.line_chart(loss_chart_df)

        history_csv = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Training History CSV",
            data=history_csv,
            file_name="gru_training_history.csv",
            mime="text/csv"
        )

    except FileNotFoundError:
        st.warning(
            "gru_training_history.csv was not found. Please save the GRU training history "
            "from the notebook and place it in the same folder as Streamlit.py."
        )
# ============================================================
# MODEL INFO
# ============================================================

st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
st.sidebar.write("Model: Word2Vec + GRU")
st.sidebar.write("Task: Binary sentiment classification")
st.sidebar.write("Labels: Negative / Positive")


