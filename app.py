import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from gensim import corpora
from gensim.models import LdaModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Page Configuration ---
st.set_page_config(
    page_title="Annual Report Chatbot",
    page_icon="🤖",
    layout="centered"
)

# --- NLTK Data Download ---
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

# --- Caching Functions for Performance ---
@st.cache_data
def process_pdf(uploaded_file):
    """
    Reads, processes, and prepares the entire PDF.
    This function is cached to run only once per file.
    """
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    all_paragraphs = []
    for page_num, page in enumerate(doc):
        # Extract text blocks with their metadata, which is better than splitting by '\\n\\n'
        blocks = page.get_text("blocks")
        for block in blocks:
            # block[4] is the text content
            paragraph_text = block[4].replace('\n', ' ').strip()
            if len(paragraph_text) > 50: # Filter out short, irrelevant blocks
                all_paragraphs.append({'page': page_num + 1, 'text': paragraph_text})
    doc.close()
    
    df = pd.DataFrame(all_paragraphs)

    # --- Preprocessing for search and topic modeling ---
    stop_words = set(stopwords.words('english'))
    custom_stopwords = ['tata', 'consultancy', 'services', 'tcs', 'company', 'ltd', 'limited', 'report', 'annual', 'financial', 'crore', 'rs', 'lakh', 'also', 'year', 'march']
    stop_words.update(custom_stopwords)

    def preprocess(text):
        text = text.lower()
        # Keep digits as they can be important in reports
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.strip()
        tokens = word_tokenize(text)
        return ' '.join([word for word in tokens if word not in stop_words and len(word) > 2])

    df['processed_text'] = df['text'].apply(preprocess)
    
    # --- Create and cache the TF-IDF model ---
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['processed_text'])
    
    return df, vectorizer, tfidf_matrix

# --- Chatbot Functions ---
def get_sentiment(text):
    """Calculates sentiment polarity for a given text."""
    blob = TextBlob(text)
    return blob.sentiment.polarity

def answer_question(df, vectorizer, tfidf_matrix, query):
    """
    Finds the most relevant paragraphs using TF-IDF and cosine similarity.
    """
    # Preprocess the user's query using the same rules
    processed_query = vectorizer.transform([query.lower()])
    
    # Calculate cosine similarity between the query and all paragraphs
    similarities = cosine_similarity(processed_query, tfidf_matrix).flatten()
    
    # Get the indices of the top 3 most similar paragraphs
    top_indices = similarities.argsort()[-3:][::-1]
    
    # Check if the top match has a reasonable similarity score
    if similarities[top_indices[0]] < 0.1: # Threshold to avoid irrelevant answers
        return "I couldn't find a confident answer to that in the document. Please try asking in a different way.", "Neutral"

    # Combine top snippets for the answer
    answer_text = ""
    for i in top_indices:
        snippet = df.iloc[i]
        answer_text += f"...{snippet['text']}... (Page {snippet['page']})\n\n"
        
    # Analyze sentiment of the combined answer
    sentiment_score = get_sentiment(answer_text)
    if sentiment_score > 0.05:
        sentiment_label = f"Positive (Score: {sentiment_score:.2f})"
    elif sentiment_score < -0.05:
        sentiment_label = f"Negative (Score: {sentiment_score:.2f})"
    else:
        sentiment_label = f"Neutral (Score: {sentiment_score:.2f})"
        
    return answer_text, sentiment_label

def get_topics(df):
    """Performs topic modeling on the document."""
    with st.spinner("Analyzing topics across the document... this can take a moment."):
        tokenized_data = [text.split() for text in df['processed_text']]
        id2word = corpora.Dictionary(tokenized_data)
        corpus = [id2word.doc2bow(text) for text in tokenized_data]
        
        lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=10, random_state=100, passes=10, alpha='auto')
        
        topics = lda_model.print_topics(num_words=8)
        response = "I found the following key topics in the report:\n\n"
        for i, topic in enumerate(topics):
            response += f"**Topic {i+1}:** {re.sub(r'[^a-zA-Z_ ]', '', topic[1]).replace('  ', ' ')}\n\n"
    return response

# --- Main Application UI ---
st.title("🤖 Annual Report Analysis Chatbot")
st.markdown("This chatbot can answer questions about the uploaded annual report. Upload a PDF to begin.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", label_visibility="collapsed")

if uploaded_file:
    # Process the document and store everything in session state
    if "processed_data" not in st.session_state:
        with st.spinner("Processing the document... Please wait."):
            df, vectorizer, tfidf_matrix = process_pdf(uploaded_file)
            st.session_state.processed_data = {
                "df": df,
                "vectorizer": vectorizer,
                "tfidf_matrix": tfidf_matrix
            }
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Thank you for uploading the report. I am ready to answer your questions. \n\n**You can ask me things like:**\n- *What does the report say about employee growth?*\n- *Show me the main topics.*\n- *What is the sentiment on risk management?*"
        })

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about the report..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = ""
            if "topic" in prompt.lower():
                response = get_topics(st.session_state.processed_data["df"])
            else:
                data = st.session_state.processed_data
                answer, sentiment = answer_question(data["df"], data["vectorizer"], data["tfidf_matrix"], prompt)
                response = f"**Answer from the document:**\n\n{answer}\n\n---\n\n**Sentiment of this answer:** {sentiment}"
            
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("Please upload a PDF file to start the chat.")

