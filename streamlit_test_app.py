import streamlit as st
from libretranslatepy import LibreTranslateAPI
import joblib
import re

# Load your trained model and vectorizer
model = joblib.load("sentiment_analysis_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Initialize LibreTranslate API
translator = LibreTranslateAPI()

# Text preprocessing (optional, adjust to your model’s needs)
def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text.lower())  # lowercase and remove punctuation
    return text

# Translate to English
def translate_to_english(text):
    try:
        translated_text = translator.translate(text, source="auto", target="en")
        return translated_text, "auto"
    except Exception as e:
        st.error("❌ Erreur lors de la traduction.")
        raise e

# Streamlit UI
st.title("🌍 Analyseur de Sentiment")

user_input = st.text_area("Entrez votre commentaire (dans n'importe quelle langue):")

if user_input:
    with st.spinner("Analyse en cours..."):
        try:
            translated_text, detected_lang = translate_to_english(user_input)
            processed_text = preprocess(translated_text)
            vectorized_input = vectorizer.transform([processed_text])
            prediction = model.predict(vectorized_input)[0]
            sentiment = "😊 Positif" if prediction == 1 else "☹️ Négatif"

            st.subheader("🔍 Résultat de l'analyse")
            st.write(f"**Langue détectée:** {detected_lang}")
            st.write(f"**Traduction du commentaire:** {translated_text}")
            st.write(f"**Sentiment :** {sentiment}")

        except Exception as e:
            st.error(f"❌ Erreur: {e}")
