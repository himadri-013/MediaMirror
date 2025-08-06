# 🪞 Media Mirror: News Bias Analyzer

**Media Mirror** is a Streamlit-powered application that analyzes the political bias and sentiment of news articles from popular media sources. It leverages NLP techniques, GloVe word embeddings, and interactive visualizations to help users assess the ideological leanings of modern media outlets.

---

## 🔍 Features

- Fetch latest news articles from various sources using NewsAPI.
- Calculate political bias using unsupervised word vector analysis (GloVe).
- Perform sentiment analysis on article content.
- Visualize:
  - Bias distribution (Left / Right / Neutral)
  - Sentiment breakdown
  - Bias trends over time
  - Word clouds of politically charged terms
- View:
  - Most biased article headlines
  - Top ideological keywords used

---

## 🗂️ Project Structure

📦 MediaMirror/
├── app.py # Main Streamlit application
<br>├── bias_analyzer.py # Core logic to compute political bias using GloVe
<br>├── visualizer.py # Visualization functions for charts, word clouds, and keyword tables
<br>├── data_fetcher.py # [Assumed] Fetches news articles using NewsAPI
<br>├── sentiment_analyzer.py # [Assumed] Computes sentiment polarity of text
<br>├── requirements.txt # List of Python dependencies
<br>└── .streamlit/
<br>         └── secrets.toml # Stores your NewsAPI key for secure access




---

## 📄 Detailed File Descriptions

### `app.py` – **Main Dashboard Interface**
- The entry point of the app (`streamlit run app.py`).
- Displays UI: title, sidebar controls, and charts.
- On button click:
  - Loads the GloVe model and keywords.
  - Fetches articles using `data_fetcher`.
  - Computes bias using `bias_analyzer.py`.
  - Computes sentiment using `sentiment_analyzer.py`.
  - Displays results using visualizations from `visualizer.py`.

### `bias_analyzer.py` – **Bias Detection Logic**
- Loads GloVe 100D word embeddings from Gensim.
- Expands seed keywords (e.g. "freedom", "equality") into larger ideological sets.
- Calculates the **bias axis** using PCA on left and right keyword vectors.
- Projects article content onto the bias axis to get a **bias score** (negative = left, positive = right).

### `visualizer.py` – **Charts, Tables & Wordcloud**
- Generates:
  - 📊 Bias Distribution Pie Chart
  - 😊 Sentiment Distribution Pie Chart
  - 📈 Line Plot of Bias over Time
  - ☁️ Word Cloud of ideological keywords
  - 📋 Tables of top used left/right words
  - 📰 Most biased article headlines (with links)

### `data_fetcher.py` – **[Expected: News Fetching Logic]**
- Uses the NewsAPI to pull recent articles from selected sources.
- Returns cleaned and structured data to the app.

### `sentiment_analyzer.py` – **[Expected: Sentiment Scoring]**
- Assigns a polarity score to article text.
- Used to classify content as Positive, Negative, or Neutral.

### `.streamlit/secrets.toml`
- Stores your NewsAPI key securely like this:
  ```toml
  NEWS_API_KEY = "your_api_key_here"
