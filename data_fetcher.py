import pandas as pd
from datetime import datetime, timedelta

def fetch_news_data(newsapi_client, source_id):
    """Fetches news data from the NewsAPI for a given source ID."""
    to_date = datetime.now()
    from_date = to_date - timedelta(days=29) # NewsAPI free tier supports 30 days max

    all_articles = newsapi_client.get_everything(
        sources=source_id,
        from_param=from_date.strftime("%Y-%m-%d"),
        to=to_date.strftime("%Y-%m-%d"),
        language='en',
        sort_by='publishedAt', # Sort by date for clearer trend analysis
        page_size=100
    )

    articles = all_articles.get('articles', [])
    if not articles:
        return pd.DataFrame()

    return pd.DataFrame(articles)



