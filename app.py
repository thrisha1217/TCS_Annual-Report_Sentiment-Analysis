# ======================================================
# app.py — Annual Report NLP Analyzer (No MALLET Version)
# ======================================================
import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import re
from io import BytesIO
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel

# ------------------------------------------------------
# NLTK setup
# ------------------------------------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ------------------------------------------------------
# Helper functions
# ------------------------------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def tokenize_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [t.lower() for t in tokens if t.isalpha()]
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return tokens

# ------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------
st.set_page_config(page_title="Annual Report NLP Analyzer", layout="wide")
st.title("📊 Annual Report NLP Analyzer")
st.write("Upload an **Annual Report (PDF)** to perform NLP tasks: cleaning, sentiment, wordcloud, and topic modeling (LDA).")

uploaded_file = st.file_uploader("📄 Upload Annual Report PDF", type=["pdf"])

if uploaded_file:
    # Step 1: Read PDF
    st.subheader("1️⃣ Reading PDF...")
    pages = []
    with pdfplumber.open(uploaded_file) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append(text)

    df_pages = pd.DataFrame({
        "page_no": list(range(1, len(pages)+1)),
        "text": pages
    })
    df_pages["text_length"] = df_pages["text"].str.len()
    df_pages = df_pages[df_pages["text_length"] > 20].reset_index(drop=True)
    st.success(f"Read {len(df_pages)} pages successfully.")
    st.dataframe(df_pages.head(3))

    # Step 2: Clean text
    st.subheader("2️⃣ Cleaning text...")
    df_pages["clean_text"] = df_pages["text"].apply(clean_text)
    df_pages["tokens"] = df_pages["clean_text"].apply(tokenize_text)
    st.dataframe(df_pages[["page_no","clean_text"]].head(3))

    # Step 3: Sentiment analysis
    st.subheader("3️⃣ Sentence-level Sentiment Analysis")
    sent_rows = []
    for _, row in df_pages.iterrows():
        page_no = row["page_no"]
        text = row["text"] or ""
        sentences = nltk.sent_tokenize(text)
        for sent in sentences:
            tb = TextBlob(sent)
            sent_rows.append((page_no, sent, tb.sentiment.polarity, tb.sentiment.subjectivity))
    df_sents = pd.DataFrame(sent_rows, columns=["page_no","sentence","polarity","subjectivity"])
    st.dataframe(df_sents.head(10))

    avg_polarity = df_sents["polarity"].mean()
    st.metric("Average Sentiment Polarity", f"{avg_polarity:.3f}")

    # Step 4: Frequent words
    st.subheader("4️⃣ Frequent Words & WordCloud")
    all_tokens = [t for tokens in df_pages["tokens"] for t in tokens]
    freq = Counter(all_tokens)
    top_words = pd.DataFrame(freq.most_common(20), columns=["word","count"])
    st.dataframe(top_words)

    wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(all_tokens))
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    # Step 5: TF-IDF and DTM
    st.subheader("5️⃣ TF-IDF & Document-Term Matrix")
    documents = df_pages["clean_text"].tolist()
    cv = CountVectorizer(max_df=0.9, min_df=2)
    dtm = cv.fit_transform(documents)
    st.write("DTM shape:", dtm.shape)

    tfidf_vect = TfidfVectorizer(max_df=0.9, min_df=2)
    tfidf = tfidf_vect.fit_transform(documents)
    st.write("TF-IDF shape:", tfidf.shape)

    # Step 6: Topic modeling (LDA)
    st.subheader("6️⃣ Topic Modeling (LDA - Gensim)")
    tokenized_docs = df_pages["tokens"].tolist()
    dictionary = Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below=3, no_above=0.9)
    corpus = [dictionary.doc2bow(text) for text in tokenized_docs]

    num_topics = st.slider("Select number of topics:", 5, 20, 10)
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )

    st.write("### Top Topics")
    topics_out = []
    for i, topic in lda_model.print_topics(num_topics=num_topics, num_words=10):
        st.write(f"**Topic {i}**: {topic}")
        topics_out.append((i, topic))

    coherence_model = CoherenceModel(model=lda_model, texts=tokenized_docs,
                                     dictionary=dictionary, coherence='c_v')
    coherence = coherence_model.get_coherence()
    st.metric("Coherence Score", f"{coherence:.3f}")

    # Step 7: Save results
    st.subheader("💾 Download Results")
    df_sents.to_csv("sentiment_results.csv", index=False)
    top_words.to_csv("frequent_words.csv", index=False)
    df_pages.to_csv("cleaned_pages.csv", index=False)

    with open("sentiment_results.csv", "rb") as f:
        st.download_button("⬇️ Download Sentiment Results", f, file_name="sentiment_results.csv")

    with open("frequent_words.csv", "rb") as f:
        st.download_button("⬇️ Download Frequent Words", f, file_name="frequent_words.csv")

    with open("cleaned_pages.csv", "rb") as f:
        st.download_button("⬇️ Download Cleaned Text Data", f, file_name="cleaned_pages.csv")

else:
    st.info("👆 Upload an Annual Report PDF to begin.")
