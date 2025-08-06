import nltk

def initial_setup():
    print("Downloading NLTK corpora...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    print("NLTK corpora download complete.")

if __name__ == '__main__':
    initial_setup()
