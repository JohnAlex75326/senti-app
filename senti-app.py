import streamlit as st
import spacy
import pytesseract
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from transformers import pipeline

# -----------------------------
# Load NLP models
# -----------------------------
# Load spaCy NLP model
nlp = spacy.load("Models/en_core_web_sm-3.7.1")
sentiment_model = pipeline("sentiment-analysis")  # Hugging Face BERT sentiment

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📝 SentAL - Smart Sentiment & Insights Analyzer")
st.write("Upload **text** or an **image (OCR)** and get **summary, AI-powered sentiment, entities, and charts**")

# Input type
option = st.radio("Choose Input Type:", ("Text", "Image"))

user_text = ""

if option == "Text":
    user_text = st.text_area("Enter your text here:")

elif option == "Image":
    uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        user_text = pytesseract.image_to_string(img)

# -----------------------------
# Process if text exists
# -----------------------------
if user_text.strip():
    st.subheader("📌 Extracted / Input Text")
    st.write(user_text)

    doc = nlp(user_text)

    # --- Brief Summary ---
    sentences = list(doc.sents)
    summary_sentences = sorted(
        sentences, key=lambda s: len([t for t in s if t.pos_ in ["NOUN", "VERB"]]), reverse=True
    )
    summary = " ".join([s.text.strip() for s in summary_sentences[:2]])
    st.subheader("📖 Brief Summary")
    st.write(summary if summary else "Could not generate a summary.")

    # --- TextBlob Sentiment ---
    tb = TextBlob(user_text)
    st.subheader("📊 TextBlob Sentiment")
    st.write(f"Polarity (−1 negative → +1 positive): **{tb.polarity:.2f}**")
    st.write(f"Subjectivity (0 = objective → 1 = subjective): **{tb.subjectivity:.2f}**")

    # --- Hugging Face Transformers Sentiment ---
    st.subheader("🤖 Transformer Sentiment (Deep Learning)")
    hf_result = sentiment_model(user_text[:512])[0]  # limit length for demo
    st.write(f"Label: **{hf_result['label']}** | Confidence: **{hf_result['score']:.2f}**")

    # --- Most Common Words ---
    words = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
    common_words = Counter(words).most_common(10)
    st.subheader("🔑 Most Common Words")
    st.write(dict(common_words))

    if common_words:
        labels, values = zip(*common_words)
        fig2, ax2 = plt.subplots()
        ax2.bar(labels, values, color="blue")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)

    # --- WordCloud ---
    st.subheader("☁️ WordCloud")
    if words:
        wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(words))
        fig3, ax3 = plt.subplots()
        ax3.imshow(wc, interpolation="bilinear")
        ax3.axis("off")
        st.pyplot(fig3)

    # --- Named Entities ---
    st.subheader("🏷 Named Entities")
    if doc.ents:
        for ent in doc.ents:
            st.markdown(f"- **{ent.text}** ({ent.label_})")
    else:
        st.write("No named entities found.")