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
    # Removing punctuation from text
    text = text.translate(str.maketrans('', '', string.punctuation)) 
    # Tokenizing the text
    tokens = nltk.word_tokenize(text)  
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
        # The next token that follows the bigram
        next_word = bigrams[i + 1][1] 
        bigram_dict[bigram][next_word] += 1
    
    return bigram_dict

if __name__ == "__main__":
    # Load and process text from project root
    file_path = "shakespeare.txt" 
    text = load_text(file_path)
    tokens = preprocess_text(text)
    
    # Generate bigrams and count occurrences
    bigrams = generate_bigrams(tokens)
    bigram_counts = create_bigram_counts(bigrams)

    # Print sample output
    print("Sample Bigram Count Dictionary:")
    for bigram, next_word_counts in list(bigram_counts.items())[:10]:
        print(f"{bigram}: {dict(next_word_counts)}")

def compute_bigram_probabilities(bigram_counts):
    """Converts bigram counts into probabilities."""
    bigram_probs = {}

    for bigram, next_word_counts in bigram_counts.items():
        total_count = sum(next_word_counts.values())  # Total occurrences of this bigram
        bigram_probs[bigram] = {word: count / total_count for word, count in next_word_counts.items()}
    
    return bigram_probs


# Compute probabilities using the previously created bigram_counts dictionary
bigram_probs = compute_bigram_probabilities(bigram_counts)

# Print sample output
print("Sample Bigram Probability Dictionary:")
for bigram, next_word_probs in list(bigram_probs.items())[:5]:  # Print first 5 entries
    print(f"{bigram}: {next_word_probs}")
