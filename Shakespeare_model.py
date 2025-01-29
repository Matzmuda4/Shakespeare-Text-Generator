import nltk
import string
from collections import defaultdict
# Download NLTK tokenizer if not already installed
nltk.download('punkt')
nltk.download('punkt_tab')

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

def generate_bigrams(tokens):
    """Creates a list of bigrams from tokenized text."""
    bigrams = list(nltk.bigrams(tokens))
    return bigrams

def create_bigram_counts(bigrams):
    """Creates a dictionary mapping bigrams to the count of the next token occurrences."""
    bigram_dict = defaultdict(lambda: defaultdict(int))
    
    for i in range(len(bigrams) - 1):
        bigram = bigrams[i]
        next_word = bigrams[i + 1][1]  # The next token that follows the bigram
        bigram_dict[bigram][next_word] += 1
    
    return bigram_dict

if __name__ == "__main__":
    # Load and process text
    file_path = "shakespeare.txt"  # Ensure this file exists in your project root
    text = load_text(file_path)
    tokens = preprocess_text(text)
    
    # Generate bigrams and count occurrences
    bigrams = generate_bigrams(tokens)
    bigram_counts = create_bigram_counts(bigrams)

    # Print sample output
    print("Sample Bigram Count Dictionary:")
    for bigram, next_word_counts in list(bigram_counts.items())[:5]:  # Print first 5 entries
        print(f"{bigram}: {dict(next_word_counts)}")