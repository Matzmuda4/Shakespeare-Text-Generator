import nltk
import string
from collections import defaultdict

# Download NLTK tokenizer if not already installed
nltk.download('punkt')

def load_text(file_path):
    """Reads the text file and returns its content."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def preprocess_text(text):
    """Cleans text: converts to lowercase, removes punctuation, and tokenizes words."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = nltk.word_tokenize(text)  # Tokenize
    return tokens