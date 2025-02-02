import nltk
import string
from collections import defaultdict
import random
from Shakespeare_model import (
    preprocess_text, generate_bigrams, create_bigram_counts, compute_bigram_probabilities,
    generate_trigrams, create_trigram_counts, compute_trigram_probabilities,
    generate_quadgrams, create_quadgram_counts, compute_quadgram_probabilities,
    sample_next_token, generate_text_from_bigram, generate_text_from_trigram, generate_text_from_quadgram
)

nltk.download('punkt')

def test_preprocess_text():
    """Tests whether the text preprocessing function correctly tokenizes input text."""
    text = "Hello, World! This is a test."
    expected = ['hello', 'world', 'this', 'is', 'a', 'test']
    assert preprocess_text(text) == expected, "preprocess_text failed"
    print("test_preprocess_text passed")

def test_generate_bigrams():
    """Tests whether bigrams are generated correctly from tokenized text."""
    tokens = ['hello', 'world', 'this', 'is', 'a', 'test']
    expected = [('hello', 'world'), ('world', 'this'), ('this', 'is'), ('is', 'a'), ('a', 'test')]
    assert generate_bigrams(tokens) == expected, "generate_bigrams failed"
    print("test_generate_bigrams passed")

def test_create_bigram_counts():
    """Tests whether bigram counts are correctly recorded in a dictionary."""
    bigrams = [('hello', 'world'), ('hello', 'world'), ('world', 'test')]
    counts = create_bigram_counts(bigrams)
    assert counts[('hello', 'world')]['world'] == 1, "create_bigram_counts failed"
    print("test_create_bigram_counts passed")

def test_compute_bigram_probabilities():
    """Tests whether bigram probabilities are calculated correctly."""
    counts = defaultdict(lambda: defaultdict(int), {("hello", "world"): {"test": 2, "is": 1}})
    probs = compute_bigram_probabilities(counts)
    assert round(probs[("hello", "world")]["test"], 2) == 0.67, "compute_bigram_probabilities failed"
    print("test_compute_bigram_probabilities passed")

def test_generate_trigrams():
    """Tests whether trigrams are generated correctly from tokenized text."""
    tokens = ['hello', 'world', 'this', 'is', 'a', 'test']
    expected = [('hello', 'world', 'this'), ('world', 'this', 'is'), ('this', 'is', 'a'), ('is', 'a', 'test')]
    assert generate_trigrams(tokens) == expected, "generate_trigrams failed"
    print("test_generate_trigrams passed")

def test_generate_quadgrams():
    """Tests whether quadgrams are generated correctly from tokenized text."""
    tokens = ['hello', 'world', 'this', 'is', 'a', 'test']
    expected = [('hello', 'world', 'this', 'is'), ('world', 'this', 'is', 'a'), ('this', 'is', 'a', 'test')]
    assert generate_quadgrams(tokens) == expected, "generate_quadgrams failed"
    print("test_generate_quadgrams passed")

def test_sample_next_token():
    """Tests whether the sampling function correctly selects a word based on probability distribution."""
    ngram_probs = {('hello', 'world'): {'test': 1.0}}
    assert sample_next_token(('hello', 'world'), ngram_probs) == 'test', "sample_next_token failed"
    print("test_sample_next_token passed")

def test_generate_text_from_bigram():
    """Tests whether text generation using bigrams produces expected output."""
    bigram_probs = {('hello', 'world'): {'test': 1.0}, ('world', 'test'): {'case': 1.0}}
    result = generate_text_from_bigram(('hello', 'world'), 4, bigram_probs)
    assert result == "hello world test case", "generate_text_from_bigram failed"
    print("test_generate_text_from_bigram passed")

def test_generate_text_from_trigram():
    """Tests whether text generation using trigrams produces expected output."""
    trigram_probs = {('hello', 'world', 'this'): {'is': 1.0}, ('world', 'this', 'is'): {'a': 1.0}}
    result = generate_text_from_trigram(('hello', 'world', 'this'), 5, trigram_probs)
    assert result == "hello world this is a", "generate_text_from_trigram failed"
    print("test_generate_text_from_trigram passed")

def test_generate_text_from_quadgram():
    """Tests whether text generation using quadgrams produces expected output."""
    quadgram_probs = {('hello', 'world', 'this', 'is'): {'a': 1.0}}
    result = generate_text_from_quadgram(('hello', 'world', 'this', 'is'), 5, quadgram_probs)
    assert result == "hello world this is a", "generate_text_from_quadgram failed"
    print("test_generate_text_from_quadgram passed")

if __name__ == "__main__":
    # Run all test cases
    test_preprocess_text()
    test_generate_bigrams()
    test_create_bigram_counts()
    test_compute_bigram_probabilities()
    test_generate_trigrams()
    test_generate_quadgrams()
    test_sample_next_token()
    test_generate_text_from_bigram()
    test_generate_text_from_trigram()
    test_generate_text_from_quadgram()
    print("All tests completed successfully.")