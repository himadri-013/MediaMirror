# app.py (with Final Verdict and Better Keywords)
import streamlit as st
import pandas as pd
from newsapi import NewsApiClient
import nltk

# Import functions from your project modules
from data_fetcher import fetch_news_data
from bias_analyzer import initialize_bias_analyzer, get_bias_score
from sentiment_analyzer import get_sentiment_score
from visualizer import (
    plot_bias_distribution,
    plot_sentiment_distribution,
    plot_bias_over_time,
    generate_wordcloud_image,
    get_top_headlines_html,
    display_top_keywords_tables # <-- New function import
)

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="News Analysis Dashboard")

# --- Caching Functions for Performance ---
@st.cache_resource
def load_bias_model():
    """Loads the heavy GloVe model and expanded keyword lists once."""
    return initialize_bias_analyzer()

@st.cache_data
def load_news_data(_newsapi_client, source_id):
    """Fetches and caches news data."""
    return fetch_news_data(_newsapi_client, source_id)

# --- Main App ---
st.title("📰 News Analysis Dashboard")

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("⚙️ Controls")           
    source_options = {
        "ABC News": "abc-news", 
        "Associated Press": "associated-press", 
        "CNN": "cnn",
        "Fox News": "fox-news", 
        "Politico": "politico", 
        "The Wall Street Journal": "the-wall-street-journal",
        "WSJ": "the-wall-street-journal", 
        "The Washington Post": "the-washington-post",
        "BBC News": "bbc-news", 
        "Reuters": "reuters",
        "The Times of India": "the-times-of-india"
    }
    source_name = st.selectbox("Select a News Source", options=sorted(list(source_options.keys())))
    source_id = source_options[source_name]
    analyze_button = st.button("Analyze News Source", type="primary", use_container_width=True)

# --- Main Panel for Displaying Analysis ---
if "NEWS_API_KEY" not in st.secrets or not st.secrets["NEWS_API_KEY"]:
    st.error("Your NewsAPI key is not configured. Please add it to your secrets.toml file.")
    st.info("Create a file at `.streamlit/secrets.toml` and add your key like this: NEWS_API_KEY = 'your_key_here'")
    st.stop()

if analyze_button:
    try:
        with st.spinner(f"Analyzing articles from {source_name}... This may take a minute."):
            # NLTK Setup logic
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                st.info("First-time setup: Downloading NLTK 'punkt' model...")
                nltk.download('punkt', quiet=True)
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                st.info("First-time setup: Downloading NLTK 'stopwords' model...")
                nltk.download('stopwords', quiet=True)

            # Initialization
            newsapi = NewsApiClient(api_key=st.secrets["NEWS_API_KEY"])
            # Now capturing all return values from the model loader
            model, bias_axis, left_keywords, right_keywords = load_bias_model()

            # Fetch and Validate Data
            df = load_news_data(newsapi, source_id)
            if df.empty:
                st.error(f"Could not retrieve any articles from '{source_name}'.")
                st.stop()
            df.dropna(subset=['content', 'title', 'url'], inplace=True)
            df = df[df['content'].str.strip() != '']
            if df.empty:
                st.error("Found articles, but none had usable content for analysis after cleaning.")
                st.stop()

            # Perform Analysis
            df['bias'] = df['content'].apply(lambda text: get_bias_score(text, model, bias_axis))
            df['sentiment'] = df['content'].apply(get_sentiment_score)
            df['publishedAt'] = pd.to_datetime(df['publishedAt'])

            # --- Display the Dashboard ---
            st.header(f"Analysis for: {source_name}", divider="rainbow")

            # --- NEW: Final Verdict Section ---
            avg_bias = df['bias'].mean()
            if avg_bias < -0.1:
                verdict = "Left-Leaning"
                delta_color = "inverse"
            elif avg_bias > 0.1:
                verdict = "Right-Leaning"
                delta_color = "normal"
            else:
                verdict = "Centrist"
                delta_color = "off"
            
            st.metric(label="Overall Bias Verdict", value=verdict, delta=f"{avg_bias:.3f} Avg. Score", delta_color=delta_color)
            st.markdown("---")
            # --- END OF NEW SECTION ---


            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Bias Distribution")
                st.plotly_chart(plot_bias_distribution(df), use_container_width=True)
            with col2:
                st.subheader("Sentiment Distribution")
                st.plotly_chart(plot_sentiment_distribution(df), use_container_width=True)

            st.subheader("Bias Over Time (30-Day Trend)")
            st.plotly_chart(plot_bias_over_time(df), use_container_width=True)

            # --- UPDATED: More Robust Word Analysis ---
            st.subheader("Most Used Politically Biased Words")
            wordcloud_image = generate_wordcloud_image(df, left_keywords, right_keywords)
            if wordcloud_image:
                st.image(wordcloud_image, use_column_width=True)
            else:
                st.info("No politically charged keywords were found to generate a word cloud.")
            
            # --- NEW: Top Keywords Tables ---
            left_table, right_table = display_top_keywords_tables(df, left_keywords, right_keywords)
            col3, col4 = st.columns(2)
            with col3:
                st.markdown(left_table, unsafe_allow_html=True)
            with col4:
                st.markdown(right_table, unsafe_allow_html=True)
            st.markdown("---")
            # --- END OF UPDATED SECTION ---


            st.subheader("Most Biased Headlines")
            left_headlines_table, right_headlines_table = get_top_headlines_html(df)
            col5, col6 = st.columns(2)
            with col5:
                st.markdown(left_headlines_table, unsafe_allow_html=True)
            with col6:
                st.markdown(right_headlines_table, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")
        st.info("This may be due to an invalid API key, network problems, or temporary issues with the news source.")
else:
    st.info("Select a news source from the sidebar and click 'Analyze' to see the dashboard.")