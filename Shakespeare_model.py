import nltk
import string
from collections import defaultdict
import random

# Download NLTK tokenizer if not already installed
nltk.download('punkt')

def load_text(file_path):
    """Reads the text file and returns its content."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def preprocess_text(text):
    """Cleans text: converts to lowercase, removes punctuation, and tokenizes words."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  # Removing punctuation
    tokens = nltk.word_tokenize(text)  # Tokenizing the text
    return tokens

# ---------------------- Bigrams ---------------------- #
def generate_bigrams(tokens):
    """Creates a list of bigrams from tokenized text."""
    return list(nltk.bigrams(tokens))

def create_bigram_counts(bigrams):
    """Creates a dictionary mapping bigrams to the count of the next token occurrences."""
    bigram_dict = defaultdict(lambda: defaultdict(int))
    
    for i in range(len(bigrams) - 1):
        bigram = bigrams[i]
        next_word = bigrams[i + 1][1]  # The next token that follows the bigram
        bigram_dict[bigram][next_word] += 1
    
    return bigram_dict

def compute_bigram_probabilities(bigram_counts):
    """Converts bigram counts into probabilities."""
    bigram_probs = {}
    for bigram, next_word_counts in bigram_counts.items():
        total_count = sum(next_word_counts.values())  
        bigram_probs[bigram] = {word: count / total_count for word, count in next_word_counts.items()}
    return bigram_probs

# ---------------------- Trigrams ---------------------- #
def generate_trigrams(tokens):
    """Creates a list of trigrams from tokenized text."""
    return list(nltk.trigrams(tokens))

def create_trigram_counts(trigrams):
    """Creates a dictionary mapping trigrams to the count of the next token occurrences."""
    trigram_dict = defaultdict(lambda: defaultdict(int))
    
    for i in range(len(trigrams) - 1):
        trigram = trigrams[i]
        next_word = trigrams[i + 1][2]  # The next token that follows the trigram
        trigram_dict[trigram][next_word] += 1
    
    return trigram_dict

def compute_trigram_probabilities(trigram_counts):
    """Converts trigram counts into probabilities."""
    trigram_probs = {}
    for trigram, next_word_counts in trigram_counts.items():
        total_count = sum(next_word_counts.values())  
        trigram_probs[trigram] = {word: count / total_count for word, count in next_word_counts.items()}
    return trigram_probs

# ---------------------- Quadgrams ---------------------- #
def generate_quadgrams(tokens):
    """Creates a list of quadgrams from tokenized text."""
    return list(nltk.ngrams(tokens, 4))

def create_quadgram_counts(quadgrams):
    """Creates a dictionary mapping quadgrams to the count of the next token occurrences."""
    quadgram_dict = defaultdict(lambda: defaultdict(int))
    
    for i in range(len(quadgrams) - 1):
        quadgram = quadgrams[i]
        next_word = quadgrams[i + 1][3]  # The next token that follows the quadgram
        quadgram_dict[quadgram][next_word] += 1
    
    return quadgram_dict

def compute_quadgram_probabilities(quadgram_counts):
    """Converts quadgram counts into probabilities."""
    quadgram_probs = {}
    for quadgram, next_word_counts in quadgram_counts.items():
        total_count = sum(next_word_counts.values())  
        quadgram_probs[quadgram] = {word: count / total_count for word, count in next_word_counts.items()}
    return quadgram_probs

# ---------------------- Sampling Functions ---------------------- #
def sample_next_token(ngram, ngram_probs):
    """Samples the next token based on probability distribution."""
    if ngram not in ngram_probs:
        return None
    next_words = list(ngram_probs[ngram].keys())
    probabilities = list(ngram_probs[ngram].values())
    return random.choices(next_words, probabilities)[0]

# ---------------------- Text Generation Functions ---------------------- #
def generate_text_from_bigram(start_bigram, num_words, bigram_probs):
    """Generates text using bigrams."""
    if start_bigram not in bigram_probs:
        return "Bigram not found in dataset."
    
    generated_words = list(start_bigram)
    for _ in range(num_words - 2):
        next_word = sample_next_token((generated_words[-2], generated_words[-1]), bigram_probs)
        if next_word is None:
            break
        generated_words.append(next_word)

    return ' '.join(generated_words)

def generate_text_from_trigram(start_trigram, num_words, trigram_probs):
    """Generates text using trigrams."""
    if start_trigram not in trigram_probs:
        return "Trigram not found in dataset."
    
    generated_words = list(start_trigram)
    for _ in range(num_words - 3):
        next_word = sample_next_token((generated_words[-3], generated_words[-2], generated_words[-1]), trigram_probs)
        if next_word is None:
            break
        generated_words.append(next_word)

    return ' '.join(generated_words)

def generate_text_from_quadgram(start_quadgram, num_words, quadgram_probs):
    """Generates text using quadgrams."""
    if start_quadgram not in quadgram_probs:
        return "Quadgram not found in dataset."
    
    generated_words = list(start_quadgram)
    for _ in range(num_words - 4):
        next_word = sample_next_token((generated_words[-4], generated_words[-3], generated_words[-2], generated_words[-1]), quadgram_probs)
        if next_word is None:
            break
        generated_words.append(next_word)

    return ' '.join(generated_words)

# ---------------------- Main Execution ---------------------- #
if __name__ == "__main__":
    # Load and process text
    file_path = "shakespeare.txt"
    text = load_text(file_path)
    tokens = preprocess_text(text)

    # Generate bigrams, trigrams, quadgrams
    bigrams = generate_bigrams(tokens)
    trigrams = generate_trigrams(tokens)
    quadgrams = generate_quadgrams(tokens)

    # Create count dictionaries
    bigram_counts = create_bigram_counts(bigrams)
    trigram_counts = create_trigram_counts(trigrams)
    quadgram_counts = create_quadgram_counts(quadgrams)

    # Compute probability dictionaries
    bigram_probs = compute_bigram_probabilities(bigram_counts)
    trigram_probs = compute_trigram_probabilities(trigram_counts)
    quadgram_probs = compute_quadgram_probabilities(quadgram_counts)

    # Generate text examples
    print("\nGenerated Text Using Bigrams:")
    print(generate_text_from_bigram(('to', 'be'), 50, bigram_probs))

    print("\nGenerated Text Using Trigrams:")
    print(generate_text_from_trigram(('to', 'be', 'or'), 50, trigram_probs))

    print("\nGenerated Text Using Quadgrams:")
    print(generate_text_from_quadgram(('to', 'be', 'or', 'not'), 50, quadgram_probs))
