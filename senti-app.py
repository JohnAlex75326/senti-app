import streamlit as st
import spacy
import pytesseract
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
from spacy.cli import download

# Load spaCy NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.warning("Downloading spaCy model 'en_core_web_sm'...")
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    st.success("Model downloaded successfully! Please refresh the page.")

# Positive/Negative words list for simple sentiment
positive_words = ["good", "great", "excellent", "happy", "love", "progress"]
negative_words = ["bad", "terrible", "sad", "hate", "angry", "problem"]

st.title("📝 Text & Image Sentiment Analyzer")
st.write("Upload text or an image and get a **summary, sentiment analysis, charts, and insights**")

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

# Process only if text is provided
if user_text.strip():
    doc = nlp(user_text)

    # --- Brief Summary ---
    sentences = list(doc.sents)
    summary_sentences = sorted(
        sentences, key=lambda s: len([t for t in s if t.pos_ in ["NOUN", "VERB"]]), reverse=True
    )
    summary = " ".join([s.text.strip() for s in summary_sentences[:2]])

    st.subheader("📌 Brief Summary")
    st.write(summary if summary else "Could not generate a summary.")

    # --- Sentiment Analysis ---
    pos_count = sum(1 for token in doc if token.text.lower() in positive_words)
    neg_count = sum(1 for token in doc if token.text.lower() in negative_words)

    st.subheader("📊 Sentiment Analysis (Numbers)")
    st.write(f"**Total Words:** {len([t for t in doc if t.is_alpha])}")
    st.write(f"**Positive Words:** {pos_count}")
    st.write(f"**Negative Words:** {neg_count}")

    if pos_count > neg_count:
        st.success("✅ Overall Sentiment: Positive")
    elif neg_count > pos_count:
        st.error("❌ Overall Sentiment: Negative")
    else:
        st.info("⚖️ Overall Sentiment: Neutral")

    # --- Sentiment Bar Chart ---
    st.subheader("📊 Sentiment Distribution")
    fig, ax = plt.subplots()
    ax.bar(["Positive", "Negative"], [pos_count, neg_count], color=["green", "red"])
    ax.set_ylabel("Word Count")
    st.pyplot(fig)

    # --- Most Common Words ---
    words = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
    common_words = Counter(words).most_common(5)

    st.subheader("🔑 Most Common Words")
    st.write(dict(common_words))

    # Chart for most common words
    if common_words:
        labels, values = zip(*common_words)
        fig2, ax2 = plt.subplots()
        ax2.bar(labels, values, color="blue")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)

    # --- Named Entities ---
    st.subheader("🏷 Named Entities")
    if doc.ents:
        for ent in doc.ents:
            st.markdown(f"- **{ent.text}** ({ent.label_})")
    else:
        st.write("No named entities found.")