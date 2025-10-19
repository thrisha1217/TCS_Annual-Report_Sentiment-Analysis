# ----------------------------------------------------------------------------
#                                LIBRARIES
# ----------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import os
import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings

# ----------------------------------------------------------------------------
#                               PAGE CONFIGURATION
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="TCS Report Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------------
#                               CUSTOM CSS
# ----------------------------------------------------------------------------
def local_css():
    """Injects custom CSS for a high-contrast dark theme."""
    st.markdown("""
        <style>
        /* --- General --- */
        body {
            background-color: #0a192f;
            color: #ffffff;
            font-family: 'Inter', sans-serif;
        }
        .main .block-container {
            padding: 2.5rem 3rem;
        }
        .stSpinner > div > div {
            border-top-color: #FFC300;
        }

        /* --- Sidebar --- */
        [data-testid="stSidebar"] {
            background-color: #192A41;
            border-right: 1px solid #233554;
        }
        [data-testid="stSidebar"] * {
            color: #ffffff;
        }

        /* --- Headings & Main Text --- */
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff;
            font-weight: 600;
        }

        /* --- Cards (Dark Sections) for Plots --- */
        .card-dark {
            background-color: #192A41;
            border-radius: 12px;
            border: 1px solid #233554;
            padding: 2.5rem;
            margin-bottom: 1.5rem;
            height: 100%;
        }

        /* --- Light Cards for Tables and Text --- */
        .card-light {
            background-color: #ffffff;
            color: #000000;
            border-radius: 12px;
            border: 1px solid #cccccc;
            padding: 2.5rem;
            margin-bottom: 1.5rem;
        }
        .card-light h1, .card-light h2, .card-light h3, .card-light h4, .card-light h5, .card-light h6, .card-light p {
            color: #000000;
        }
        .card-light .stDataFrame, .card-light .stTable {
            background-color: #ffffff !important;
        }
        .card-light table, .card-light th, .card-light td {
            color: #000000 !important;
            border-color: #dddddd !important;
        }
        .card-light .stTextArea textarea {
            background-color: #f0f2f6;
            color: #000000;
            border: 1px solid #cccccc;
        }
        
        /* --- Metrics --- */
        .stMetric {
            background-color: #ffffff;
            border: 1px solid #cccccc;
            border-radius: 10px;
            text-align: center;
            padding: 15px;
        }
        .stMetric label {
            color: #333333 !important;
        }
        .stMetric div[data-testid="metric-value"] {
            color: #000000 !important;
            font-size: 2.5rem;
        }

        /* --- Plots and Images --- */
        .stPlot, .stImage {
            background-color: transparent;
        }
        </style>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
#                               INITIAL NLTK SETUP
# ----------------------------------------------------------------------------
@st.cache_resource
def download_nltk_data():
    """Downloads necessary NLTK models if not already present."""
    for resource in ['punkt', 'stopwords']:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)

# ----------------------------------------------------------------------------
#                               CACHED HELPER FUNCTIONS
# ----------------------------------------------------------------------------
@st.cache_data
def load_and_extract_text(pdf_path):
    """Loads a PDF from a file path and extracts its text using PyMuPDF."""
    if not os.path.exists(pdf_path):
        return None, 0
    all_text = ""
    with fitz.open(pdf_path) as doc:
        page_count = doc.page_count
        for page in doc:
            text = page.get_text()
            if text:
                all_text += text + "\n"
    return all_text, page_count

@st.cache_data
def preprocess_and_tokenize(text):
    """Cleans text and returns a list of tokens."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return tokens

@st.cache_data
def analyze_sentiment(raw_text):
    """Analyzes the sentiment of each sentence in the text."""
    sentences = sent_tokenize(raw_text)
    sentiments = []
    for s in sentences:
        blob = TextBlob(s)
        sentiments.append({
            "Sentence": s,
            "Polarity": blob.sentiment.polarity,
            "Subjectivity": blob.sentiment.subjectivity
        })
    df = pd.DataFrame(sentiments)
    df['Sentiment'] = df['Polarity'].apply(
        lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral')
    )
    return df

@st.cache_data
def get_topic_model(_clean_text, num_topics):
    """Builds a scikit-learn LDA topic model."""
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    doc_term_matrix = tfidf_vectorizer.fit_transform([_clean_text])
    
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(doc_term_matrix)
    
    return lda_model, tfidf_vectorizer, doc_term_matrix.shape

# ----------------------------------------------------------------------------
#                               PLOT STYLING
# ----------------------------------------------------------------------------
def style_plot(fig, ax, title, dark_theme=True):
    """Applies styling to a matplotlib plot."""
    bg_color = '#192A41' if dark_theme else '#ffffff'
    text_color = '#ffffff' if dark_theme else '#000000'
    label_color = '#a0a0d0' if dark_theme else '#333333'
    tick_color = '#e0e0e0' if dark_theme else '#333333'
    spine_color = '#233554' if dark_theme else '#cccccc'
        
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.set_title(title, color=text_color, fontsize=18, pad=20)
    ax.xaxis.label.set_color(label_color)
    ax.yaxis.label.set_color(label_color)
    ax.tick_params(axis='x', colors=tick_color)
    ax.tick_params(axis='y', colors=tick_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(spine_color)

# ----------------------------------------------------------------------------
#                               MAIN APP LOGIC
# ----------------------------------------------------------------------------
def main():
    warnings.filterwarnings("ignore")
    local_css()
    download_nltk_data()

    # --- SIDEBAR ---
    st.sidebar.title("📄 TCS Report Analyzer")
    st.sidebar.markdown("Analysis of the TCS Annual Report 2024-2025.")
    st.sidebar.markdown("---")
    
    pdf_path = r"D:\SEM-VII\NLP\NLP Mini Project\tcs-annual-report-2024-2025.pdf"
    page_options = ["Overview", "Sentiment Analysis", "Word Analysis", "Topic Modeling"]
    
    st.sidebar.markdown("---")
    selected_page = st.sidebar.radio("Navigate", page_options)

    # --- LOAD DATA ---
    with st.spinner("Analyzing the TCS Annual Report..."):
        raw_text, page_count = load_and_extract_text(pdf_path)

    # --- MAIN PANEL ---
    if raw_text:
        all_tokens = preprocess_and_tokenize(raw_text)
        clean_text_for_model = ' '.join(all_tokens)
        df_sentiments = analyze_sentiment(raw_text)

        if selected_page == "Overview":
            st.title("📊 Document Overview")
            
            st.subheader("Key Document Metrics")
            col1, col2, col3, col4 = st.columns(4

