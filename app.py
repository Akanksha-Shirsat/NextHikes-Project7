
import re
from pathlib import Path

import joblib
import streamlit as st


# Page configuration
st.set_page_config(
    page_title="Disaster Tweet Classifier",
    page_icon="🚨",
    layout="centered"
)


# Load the saved model
MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "final_disaster_tweet_model.pkl"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# Clean tweet text using the same logic as the notebook
def clean_text(text):
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtag symbol but keep the word
    text = re.sub(r"#", "", text)

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Dashboard heading
st.title("🚨 Disaster Tweet Classifier")

st.write(
    "Enter a tweet to predict whether it is related "
    "to a disaster or a non-disaster."
)

st.divider()


# Tweet input
tweet = st.text_area(
    "Enter your tweet",
    placeholder="Example: Heavy flooding has affected several areas...",
    height=150
)


# Prediction button
if st.button("Predict Tweet", type="primary", use_container_width=True):

    if not tweet.strip():
        st.warning("Please enter a tweet first.")

    else:
        try:
            model = load_model()

            cleaned_tweet = clean_text(tweet)

            prediction = model.predict([cleaned_tweet])[0]
            probabilities = model.predict_proba([cleaned_tweet])[0]

            disaster_probability = probabilities[1] * 100

            st.subheader("Prediction Result")

            if prediction == 1:
                st.error("🚨 Disaster-related Tweet")
            else:
                st.success("✅ Non-Disaster Tweet")

            st.metric(
                "Estimated Disaster Probability",
                f"{disaster_probability:.2f}%"
            )

        except FileNotFoundError:
            st.error(
                "Model file not found. Please check that "
                "the .pkl file is inside the models folder."
            )

        except Exception as e:
            st.error(f"An error occurred: {e}")


st.divider()

st.caption("NLP Project for Disaster Tweet Classification")