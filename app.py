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
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Pages", page_count)
            col2.metric("Characters", f"{len(raw_text):,}")
            col3.metric("Sentences", f"{len(df_sentiments):,}")
            col4.metric("Total Tokens", f"{len(all_tokens):,}")
            
            st.markdown('<div class="card-light" style="margin-top: 2rem;">', unsafe_allow_html=True)
            st.subheader("Raw Text Preview")
            st.text_area("", raw_text[:2500], height=350, key="overview_text")
            st.markdown("</div>", unsafe_allow_html=True)
            
        elif selected_page == "Sentiment Analysis":
            st.title("😊 Sentiment Analysis")

            st.markdown('<div class="card-dark">', unsafe_allow_html=True)
            avg_polarity = df_sentiments['Polarity'].mean()
            sentiment_counts = df_sentiments['Sentiment'].value_counts()
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Average Polarity", f"{avg_polarity:.3f}")
                fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
                wedges, texts, autotexts = ax_pie.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
                           startangle=90, colors=['#57a773', '#8c8c8c', '#d62728'], textprops={'color':"w", 'fontsize': 12})
                ax_pie.axis('equal')
                style_plot(fig_pie, ax_pie, "Sentence Sentiment Breakdown")
                st.pyplot(fig_pie)
            with col2:
                st.subheader("Polarity & Subjectivity Distribution")
                fig_hist, ax_hist = plt.subplots(figsize=(10, 5.5))
                sns.histplot(df_sentiments['Polarity'], bins=50, kde=True, ax=ax_hist, color="#ff6b6b", label="Polarity")
                sns.histplot(df_sentiments['Subjectivity'], bins=50, kde=True, ax=ax_hist, color="#4ecdc4", label="Subjectivity")
                ax_hist.legend()
                style_plot(fig_hist, ax_hist, "Distribution of Sentiment Scores")
                st.pyplot(fig_hist)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="card-light">', unsafe_allow_html=True)
            st.subheader("Sentiment Samples")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Most Positive Sentences")
                st.dataframe(df_sentiments.nlargest(5, 'Polarity'), use_container_width=True)
            with col2:
                st.markdown("##### Most Negative Sentences")
                st.dataframe(df_sentiments.nsmallest(5, 'Polarity'), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        elif selected_page == "Word Analysis":
            st.title("🔍 Word Frequency & Cloud")
            col1, col2 = st.columns([1, 1.5])
            
            with col1:
                st.markdown('<div class="card-dark" style="height: 680px;">', unsafe_allow_html=True)
                st.subheader("Top 20 Frequent Words")
                freq_dist = nltk.FreqDist(all_tokens)
                df_freq = pd.DataFrame(freq_dist.most_common(20), columns=['Word', 'Count'])
                fig, ax = plt.subplots(figsize=(8, 9))
                sns.barplot(x='Count', y='Word', data=df_freq, palette='mako_r', ax=ax)
                style_plot(fig, ax, 'Frequent Words')
                st.pyplot(fig)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="card-dark" style="height: 680px;">', unsafe_allow_html=True)
                st.subheader("Word Cloud")
                wordcloud = WordCloud(width=800, height=600, background_color=None, mode="RGBA", colormap='viridis').generate(clean_text_for_model)
                fig_wc, ax_wc = plt.subplots(figsize=(10, 8))
                ax_wc.imshow(wordcloud, interpolation='bilinear')
                ax_wc.axis("off")
                fig_wc.patch.set_facecolor('#192A41')
                st.pyplot(fig_wc)
                st.markdown("</div>", unsafe_allow_html=True)

        elif selected_page == "Topic Modeling":
            st.title("🧩 Topic Modeling (LDA)")
            st.markdown('<div class="card-dark">', unsafe_allow_html=True)
            st.subheader("Discover Latent Topics")
            num_topics = st.slider("Select the number of topics:", min_value=3, max_value=15, value=10, step=1)
            
            with st.spinner(f"Building LDA model for {num_topics} topics..."):
                lda_model, tfidf_vectorizer, matrix_shape = get_topic_model(clean_text_for_model, num_topics)
                feature_names = tfidf_vectorizer.get_feature_names_out()
                
                st.write(f"TF-IDF Matrix Shape: **{matrix_shape}** (documents, features).")

                for topic_idx, topic in enumerate(lda_model.components_):
                    top_words_idx = topic.argsort()[:-8:-1]
                    top_words = [feature_names[j] for j in top_words_idx]
                    top_weights = topic[top_words_idx]
                    
                    with st.expander(f"Topic #{topic_idx + 1}: {', '.join(top_words[:3])}..."):
                        fig, ax = plt.subplots(figsize=(10, 5))
                        sns.barplot(x=top_weights, y=top_words, ax=ax, palette="rocket_r")
                        style_plot(fig, ax, f'Top Words for Topic #{topic_idx + 1}')
                        st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.title("Welcome to the TCS Annual Report Analyzer")
        st.error(f"Error: The report file was not found at the specified path:")
        st.code(pdf_path)
        st.info("Please make sure the file exists at that location and the script has permission to read it.")
        
# --- SCRIPT EXECUTION ---
if __name__ == "__main__":
    main()

