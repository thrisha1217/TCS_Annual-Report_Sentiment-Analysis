import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from textblob import TextBlob
from gensim import corpora
from gensim.models import LdaModel
from collections import Counter

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
    pages_data = []
    for page_num, page in enumerate(doc):
        pages_data.append({
            'page_number': page_num + 1,
            'text': page.get_text("text", sort=True) # Sort by reading order
        })
    doc.close()
    df = pd.DataFrame(pages_data)

    # --- Preprocessing ---
    stop_words = set(stopwords.words('english'))
    custom_stopwords = ['tata', 'consultancy', 'services', 'tcs', 'company', 'ltd', 'limited', 'report', 'annual', 'financial', 'crore', 'rs', 'lakh', 'also', 'year', 'march']
    stop_words.update(custom_stopwords)

    def preprocess(text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.strip()
        tokens = word_tokenize(text)
        return [word for word in tokens if word not in stop_words and len(word) > 2]

    df['processed_tokens'] = df['text'].apply(preprocess)
    return df

# --- Chatbot Functions ---
def get_sentiment(text):
    """Calculates sentiment polarity for a given text."""
    blob = TextBlob(text)
    return blob.sentiment.polarity

def answer_question(df, query):
    """
    Finds the most relevant paragraphs to a query, provides an answer,
    and analyzes its sentiment.
    """
    processed_query = [word for word in query.lower().split() if word not in stopwords.words('english')]
    
    best_snippets = []
    for index, row in df.iterrows():
        # Find paragraphs within the page text
        paragraphs = row['text'].split('\n\n')
        for para in paragraphs:
            if len(para.strip()) > 50: # Ensure paragraph has some substance
                matches = sum(1 for word in processed_query if word in para.lower())
                if matches > 0:
                    best_snippets.append((matches, para, row['page_number']))
    
    # Sort snippets by number of matches and get top 3
    best_snippets.sort(key=lambda x: x[0], reverse=True)
    
    if not best_snippets:
        return "I couldn't find a direct answer to that in the document. Please try asking in a different way.", "Neutral"

    # Combine top snippets for the answer
    top_snippets = best_snippets[:3]
    answer_text = "\n\n".join([f"...{snippet[1]}... (Page {snippet[2]})" for snippet in top_snippets])
    
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
        tokenized_data = df['processed_tokens'].tolist()
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
    # Process the document and store in session state
    if "processed_df" not in st.session_state:
        with st.spinner("Processing the document... Please wait."):
            st.session_state.processed_df = process_pdf(uploaded_file)
        
        # Add initial assistant message after processing
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Thank you for uploading the report. I am ready to answer your questions. \n\n**You can ask me things like:**\n- *What does the report say about employee growth?*\n- *Show me the main topics.*\n- *What is the sentiment on risk management?*"
        })

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about the report..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            response = ""
            # Check for specific commands first
            if "topic" in prompt.lower():
                response = get_topics(st.session_state.processed_df)
            else:
                answer, sentiment = answer_question(st.session_state.processed_df, prompt)
                response = f"**Answer from the document:**\n\n{answer}\n\n---\n\n**Sentiment of this answer:** {sentiment}"
            
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("Please upload a PDF file to start the chat.")

