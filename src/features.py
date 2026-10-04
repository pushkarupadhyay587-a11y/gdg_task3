from sklearn.feature_extraction.text import TfidfVectorizer



def build_vectorizer():
    return TfidfVectorizer(
    stop_words="english", ngram_range=(1, 2),
    max_features=100000, sublinear_tf=True
)
    

    