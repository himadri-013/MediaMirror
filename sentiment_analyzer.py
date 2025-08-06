from textblob import TextBlob

def get_sentiment_score(text):
    """
    Returns a sentiment polarity score between -1 (negative) to 1 (positive).
    """
    blob = TextBlob(text)
    return round(blob.sentiment.polarity, 3)
