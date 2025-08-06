import numpy as np
import gensim.downloader as api
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import streamlit as st

@st.cache_resource
def initialize_bias_analyzer():
    """
    Loads the GloVe model and pre-computes the bias axis and keyword lists.
    """
    model = api.load("glove-wiki-gigaword-100")

    LEFT_SEEDS = [
        "liberal", "equality", "diversity", "climate", "feminism", "welfare",
        "progressive", "inclusion", "regulation", "redistribution", "socialism",
        "justice", "union", "environment", "activism", "lgbt", "abortion", "universal"
    ]
    RIGHT_SEEDS = [
        "conservative", "freedom", "tradition", "patriotism", "market", "capitalism",
        "gun", "military", "taxes", "merit", "border", "immigration", "faith", "religion",
        "morality", "family", "order", "discipline"
    ]

    def expand_anchors(seed_words, topn=15):
        expanded = []
        for word in seed_words:
            if word in model:
                similar_words = model.most_similar(word, topn=topn)
                expanded.extend([w for w, _ in similar_words])
        return list(set(seed_words + expanded))

    expanded_left = expand_anchors(LEFT_SEEDS)
    expanded_right = expand_anchors(RIGHT_SEEDS)

    left_vecs = [model[w] for w in expanded_left if w in model]
    right_vecs = [model[w] for w in expanded_right if w in model]

    left_centroid = np.mean(left_vecs, axis=0)
    right_centroid = np.mean(right_vecs, axis=0)

    pca = PCA(n_components=1)
    pca.fit(np.array([left_centroid, right_centroid]))
    bias_axis = pca.components_[0]

    # Return the keyword lists along with the model and axis
    return model, bias_axis, expanded_left, expanded_right

def get_bias_score(text, model, bias_axis):
    """
    Calculates the political bias score for a given text.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0

    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation and t in model]

    if not tokens:
        return 0.0

    vectors = np.array([model[t] for t in tokens])
    projections = vectors @ bias_axis
    bias_score = np.mean(projections)

    return float(bias_score)