import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from gensim import corpora
from gensim.models import LdaModel
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Annual Report NLP Assistant",
    page_icon="🤖",
    layout="wide"
)

# --- NLTK Data Download (Cached) ---
@st.cache_resource
def download_nltk_data():
    """Downloads necessary NLTK models if not already present."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')

download_nltk_data()

# --- Core NLP Processing and Caching ---
@st.cache_data
def load_and_process_data(uploaded_file):
    """
    (Tasks 1, 2, 3, 5, 7)
    Reads PDF, preprocesses text, and creates a TF-IDF matrix.
    This entire process is cached to run only once per file upload.
    """
    with st.spinner("Analyzing the annual report... This may take a moment."):
        # 1. Import PDF and read all pages
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        all_paragraphs = []
        for page_num, page in enumerate(doc):
            blocks = page.get_text("blocks")
            for block in blocks:
                para_text = block[4].replace('\n', ' ').strip()
                if len(para_text.split()) > 10: # Filter for substantive paragraphs
                    all_paragraphs.append({'page': page_num + 1, 'text': para_text})
        doc.close()
        
        # 2. Save into a DataFrame
        df = pd.DataFrame(all_paragraphs)
        if df.empty:
            return None, None, None, None

        # 3. Preprocess text
        stop_words = set(stopwords.words('english'))
        custom_stopwords = ['tata', 'consultancy', 'services', 'tcs', 'company', 'ltd', 'limited', 'report', 'annual', 'financial', 'crore', 'rs', 'lakh', 'also', 'year', 'march', 'fy']
        stop_words.update(custom_stopwords)

        def preprocess(text):
            text = text.lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = re.sub(r'\d+', '', text) # Remove digits
            tokens = word_tokenize(text) # 5. Word Tokenize
            return ' '.join([word for word in tokens if word not in stop_words and len(word) > 2])

        df['processed_text'] = df['text'].apply(preprocess)
        
        # 7. Convert to TF-IDF Document Term Matrix
        vectorizer = TfidfVectorizer(min_df=2, max_df=0.9)
        tfidf_matrix = vectorizer.fit_transform(df['processed_text'])
        
    return df, vectorizer, tfidf_matrix

# --- Chatbot & Analysis Functions ---
def find_answer(query, df, vectorizer, tfidf_matrix):
    """
    Finds the most relevant paragraphs for a query using TF-IDF.
    """
    processed_query = vectorizer.transform([query.lower()])
    similarities = cosine_similarity(processed_query, tfidf_matrix).flatten()
    top_indices = np.argsort(-similarities)[:3] # Get top 3 matches
    
    if similarities[top_indices[0]] < 0.1: # Confidence threshold
        return "I could not find a relevant answer in the document.", "Neutral"
        
    # Combine answer snippets
    answer_text = ""
    for i in top_indices:
        snippet = df.iloc[i]
        answer_text += f"- {snippet['text']} (Page {snippet['page']})\n"
    
    # 4. Sentence Tokenize and Calculate Sentiment
    blob = TextBlob(answer_text)
    sentiment_score = blob.sentiment.polarity
    if sentiment_score > 0.05:
        sentiment_label = f"Positive (Score: {sentiment_score:.2f})"
    elif sentiment_score < -0.05:
        sentiment_label = f"Negative (Score: {sentiment_score:.2f})"
    else:
        sentiment_label = f"Neutral (Score: {sentiment_score:.2f})"
        
    return answer_text, sentiment_label

def generate_wordcloud(df):
    """(Task 6) Generates and displays a word cloud and frequent words."""
    full_text = " ".join(df['processed_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(full_text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    words = word_tokenize(full_text)
    freq_dist = nltk.FreqDist(words)
    freq_df = pd.DataFrame(freq_dist.most_common(20), columns=['Word', 'Frequency'])
    st.dataframe(freq_df)

def run_topic_modeling(df):
    """(Task 8) Builds and displays the LDA topic model."""
    with st.spinner("Discovering topics..."):
        tokenized_data = [text.split() for text in df['processed_text']]
        id2word = corpora.Dictionary(tokenized_data)
        corpus = [id2word.doc2bow(text) for text in tokenized_data]
        
        lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=10, random_state=100, passes=15, alpha='auto', per_word_topics=True)
        
        st.subheader("Top 10 Discovered Topics")
        topics = lda_model.print_topics(num_words=8)
        for i, topic in enumerate(topics):
            # Clean up the output for better display
            topic_words = re.sub(r'[^a-zA-Z_ ]', '', topic[1]).replace("  ", " ")
            st.markdown(f"**Topic {i+1}:** {topic_words}")

# --- Streamlit App UI ---
st.title("📄 TCS Annual Report NLP Assistant")
st.markdown("Upload the TCS Annual Report PDF to begin. You can ask questions, or run a full analysis.")

# --- File Uploader and Main Logic ---
uploaded_file = st.file_uploader("Upload your PDF Report", type="pdf")

if uploaded_file:
    # Load and process data (this will be cached)
    df, vectorizer, tfidf_matrix = load_and_process_data(uploaded_file)
    
    if df is not None:
        st.success(f"Successfully analyzed {len(df)} paragraphs from the report.")
        
        # Create tabs for different functionalities
        tab1, tab2, tab3 = st.tabs(["💬 Chatbot Q&A", "📊 Word Cloud Analysis", "🌐 Topic Modeling"])

        with tab1:
            st.header("Ask a Question")
            st.markdown("Ask a question about the report, and the chatbot will find the most relevant paragraphs and analyze their sentiment.")
            user_query = st.text_input("e.g., What are the main business risks?", key="query_input")

            if user_query:
                answer, sentiment = find_answer(user_query, df, vectorizer, tfidf_matrix)
                st.subheader("Answer from Document:")
                st.markdown(answer)
                st.subheader("Sentiment of this Answer:")
                st.markdown(sentiment)

        with tab2:
            st.header("Frequent Words and Word Cloud")
            with st.spinner("Generating visuals..."):
                generate_wordcloud(df)
        
        with tab3:
            st.header("Latent Dirichlet Allocation (LDA) Topic Model")
            run_topic_modeling(df)
    else:
        st.error("Could not extract text from the uploaded PDF. Please try another file.")
else:
    st.info("Please upload a file to get started.")

