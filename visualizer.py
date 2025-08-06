# visualizer.py (Updated for better word analysis)
import plotly.graph_objects as go
from wordcloud import WordCloud
import pandas as pd
import nltk
from collections import Counter

def plot_bias_distribution(df):
    """Creates and returns the bias distribution pie chart."""
    left_count = df[df['bias'] < -0.2].shape[0]
    right_count = df[df['bias'] > 0.2].shape[0]
    neutral_count = len(df) - left_count - right_count
    fig = go.Figure(data=[go.Pie(
        labels=['🔵 Left-biased', '⚪ Neutral', '🔴 Right-biased'],
        values=[left_count, neutral_count, right_count],
        marker_colors=['#007bff', '#cccccc', '#dc3545'],
    )])
    fig.update_layout(showlegend=True, margin=dict(l=0, r=0, t=0, b=0))
    return fig

def plot_sentiment_distribution(df):
    """Creates and returns the sentiment distribution pie chart."""
    positive_count = df[df['sentiment'] > 0.05].shape[0]
    negative_count = df[df['sentiment'] < -0.05].shape[0]
    neutral_sent_count = len(df) - positive_count - negative_count
    fig = go.Figure(data=[go.Pie(
        labels=['✅ Positive', '⚪ Neutral', '⚠️ Negative'],
        values=[positive_count, neutral_sent_count, negative_count],
        marker_colors=['#28a745', '#cccccc', '#ffc107'],
    )])
    fig.update_layout(showlegend=True, margin=dict(l=0, r=0, t=0, b=0))
    return fig

def plot_bias_over_time(df):
    """Creates and returns the bias over time line chart."""
    df['date'] = pd.to_datetime(df['publishedAt']).dt.date
    daily_bias = df.groupby('date')['bias'].mean().reset_index()
    fig = go.Figure(data=go.Scatter(
        x=daily_bias['date'], y=daily_bias['bias'], mode='lines+markers',
        hovertemplate="<b>%{x|%b %d}</b><br>Avg Bias Score = %{y:.2f}<extra></extra>"
    ))
    fig.add_hline(y=0, line_width=2, line_dash="dash", line_color="grey")
    fig.update_layout(xaxis_title=None, yaxis_title='Average Bias Score')
    return fig

def generate_wordcloud_image(df, left_keywords, right_keywords):
    """
    Generates and returns a word cloud image object using the expanded keyword lists.
    """
    all_ideological_words = set(left_keywords + right_keywords)
    tokens = nltk.word_tokenize(" ".join(df['content'].dropna()).lower())
    
    # Use the more comprehensive list of keywords from the bias analyzer
    biased_words = [word for word in tokens if word in all_ideological_words]
    word_freq = Counter(biased_words)
    
    if not word_freq:
        return None
        
    wc = WordCloud(width=800, height=300, background_color='white', collocations=False).generate_from_frequencies(word_freq)
    return wc.to_image()

def get_top_headlines_html(df):
    """Generates and returns HTML tables for the most biased headlines."""
    top_left = df.sort_values('bias', ascending=True).head(5)
    top_right = df.sort_values('bias', ascending=False).head(5)
    def create_table(df_slice, title):
        style = "<style>table{width:100%;border-collapse:collapse;}th,td{padding:6px;border:1px solid #ddd;text-align:left;}th{background-color:#f8f9fa;}</style>"
        table_html = f"{style}<h4>{title}</h4><table><tr><th>Score</th><th>Headline</th></tr>"
        for _, row in df_slice.iterrows():
            table_html += f"<tr><td>{row['bias']:.2f}</td><td><a href='{row['url']}' target='_blank' rel='noopener noreferrer'>{row['title']}</a></td></tr>"
        table_html += "</table>"
        return table_html
    left_html = create_table(top_left, 'Top 5 Most Left-Leaning')
    right_html = create_table(top_right, 'Top 5 Most Right-Leaning')
    return left_html, right_html

def display_top_keywords_tables(df, left_keywords, right_keywords):
    """
    NEW: Generates HTML tables for the top 10 most frequent
    left and right-leaning keywords.
    """
    tokens = nltk.word_tokenize(" ".join(df['content'].dropna()).lower())
    
    left_freq = Counter(word for word in tokens if word in left_keywords)
    right_freq = Counter(word for word in tokens if word in right_keywords)

    def create_table(freq_counter, title):
        if not freq_counter:
            return f"<h4>{title}</h4><p>No keywords found.</p>"
        
        df_freq = pd.DataFrame(freq_counter.most_common(10), columns=['Keyword', 'Count'])
        style = "<style>table{width:100%;border-collapse:collapse;}th,td{padding:6px;border:1px solid #ddd;text-align:left;}th{background-color:#f8f9fa;}</style>"
        table_html = f"{style}<h4>{title}</h4>" + df_freq.to_html(index=False, justify='left')
        return table_html

    left_table = create_table(left_freq, 'Top 10 Left-Leaning Keywords')
    right_table = create_table(right_freq, 'Top 10 Right-Leaning Keywords')
    return left_table, right_table