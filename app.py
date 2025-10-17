import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
from tqdm import tqdm
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim import corpora
from gensim.models import LdaModel

# --- Page Configuration ---
st.set_page_config(
    page_title="TCS Annual Report NLP Analysis",
    page_icon="📄",
    layout="wide"
)

# --- NLTK Data Download ---
# Using a function to ensure this only runs once.
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
download_nltk_data()


# --- Caching Functions for Performance ---
# CHANGE: Replaced deprecated st.cache with st.cache_data
@st.cache_data
def read_pdf_to_dataframe(file_path):
    """Reads a PDF file and extracts text from each page into a pandas DataFrame."""
    if not os.path.exists(file_path):
        st.error(f"Error: The file '{file_path}' was not found.")
        return pd.DataFrame()

    doc = fitz.open(file_path)
    pages_data = []
    for page_num, page in enumerate(doc):
        pages_data.append({
            'page_number': page_num + 1,
            'text': page.get_text()
        })
    doc.close()
    return pd.DataFrame(pages_data)

# CHANGE: Replaced deprecated st.cache with st.cache_data
@st.cache_data
def preprocess_text(text):
    """Cleans and preprocesses a single string of text."""
    stop_words = set(stopwords.words('english'))
    custom_stopwords = ['tata', 'consultancy', 'services', 'tcs', 'company', 'ltd', 'limited', 'report', 'annual', 'financial', 'crore', 'rs', 'lakh', 'also', 'year', 'march']
    stop_words.update(custom_stopwords)
    
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(filtered_tokens)


# --- Main Application ---
st.title("📄 TCS Annual Report NLP Analysis")
st.markdown("This application performs a complete NLP analysis on the TCS Annual Report for FY 2024-25.")

# --- Step 1 & 2: Load and Display Data ---
st.header("1. Reading the PDF into a DataFrame")
PDF_FILE_PATH = 'tcs-annual-report-2024-2025.pdf'
df = read_pdf_to_dataframe(PDF_FILE_PATH)

if not df.empty:
    st.success("Successfully read the PDF into a DataFrame.")
    st.write(f"Total pages read: {len(df)}")
    st.dataframe(df.head())

    # --- Step 3: Preprocessing ---
    st.header("2. Preprocessing the Text")
    with st.spinner("Cleaning text data... this may take a moment."):
        df['processed_text'] = df['text'].apply(preprocess_text)
    st.success("Text preprocessing complete.")
    st.write("Showing original text vs. cleaned text:")
    st.dataframe(df[['text', 'processed_text']].head())

    # --- Step 4: Sentiment Analysis ---
    st.header("3. Sentiment Analysis")
    with st.spinner("Analyzing sentiment for each page..."):
        sentiments = []
        for text in df['text']:
            sentences = sent_tokenize(text)
            page_polarity = 0
            if sentences:
                for sentence in sentences:
                    page_polarity += TextBlob(sentence).sentiment.polarity
                sentiments.append(page_polarity / len(sentences))
            else:
                sentiments.append(0)
        df['polarity'] = sentiments
    st.success("Sentiment analysis complete.")
    st.write("Average sentiment polarity per page (Positive > 0, Negative < 0):")
    st.dataframe(df[['page_number', 'polarity']].head())
    
    avg_polarity = df['polarity'].mean()
    st.metric(label="Overall Average Polarity of the Report", value=f"{avg_polarity:.4f}")
    st.progress((avg_polarity + 1) / 2) # Normalize polarity from [-1, 1] to [0, 1] for progress bar

    # --- Step 5 & 6: Frequent Words & Word Cloud ---
    st.header("4. Frequent Words and Word Cloud")
    with st.spinner("Generating word cloud..."):
        full_text_corpus = " ".join(df['processed_text'])
        all_words = word_tokenize(full_text_corpus)
        fdist = FreqDist(all_words)
        
        st.subheader("Top 20 Most Frequent Words")
        freq_df = pd.DataFrame(fdist.most_common(20), columns=['Word', 'Frequency'])
        st.dataframe(freq_df)

        st.subheader("Word Cloud")
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            min_font_size=10
        ).generate(full_text_corpus)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

    # --- Step 7 & 8: Topic Modeling ---
    st.header("5. Topic Modeling with LDA")
    with st.spinner("Building topic model... this is the most intensive step."):
        tokenized_data = [word_tokenize(text) for text in df['processed_text']]
        id2word = corpora.Dictionary(tokenized_data)
        corpus = [id2word.doc2bow(text) for text in tokenized_data]
        
        lda_model = LdaModel(
            corpus=corpus,
            id2word=id2word,
            num_topics=10,
            random_state=100,
            update_every=1,
            chunksize=100,
            passes=10,
            alpha='auto'
        )
        
        st.success("Topic model built successfully!")
        st.subheader("Discovered Topics")
        topics = lda_model.print_topics(num_words=10)
        for i, topic in enumerate(topics):
            st.markdown(f"**Topic {i+1}:** {topic[1]}")

else:
    st.warning("Could not proceed with analysis as the DataFrame is empty.")
