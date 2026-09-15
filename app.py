import streamlit as st
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_models():
    model = joblib.load(BASE_DIR / "LogisticRegression.pkl")
    vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")

    return model, vectorizer


model, tfidf_vectorizer = load_models()


emotion_map = {
    0: "Anger",
    1: "Fear",
    2: "Joy",
    3: "Love",
    4: "Sadness",
    5: "Surprise"
}


st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="😊",
    layout="centered"
)

st.title("😊 Emotion Classifier")

st.write(
    "Enter a sentence and the trained NLP model will predict its emotion."
)


text = st.text_area(
    "Enter your text",
    placeholder="Example: I am very happy today because I achieved my goal.",
    height=150
)


if st.button("Predict Emotion", type="primary"):

    if not text.strip():

        st.warning("Please enter some text.")

    else:

        text_tfidf = tfidf_vectorizer.transform([text])

        prediction = model.predict(text_tfidf)[0]

        emotion = emotion_map.get(
            int(prediction),
            str(prediction)
        )

        st.success(f"Predicted Emotion: {emotion}")

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(text_tfidf)[0]

            confidence = float(max(probabilities)) * 100

            st.info(f"Confidence: {confidence:.2f}%")
