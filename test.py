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
    text = "Hello, World! This is a test."
    expected = ['hello', 'world', 'this', 'is', 'a', 'test']
    assert preprocess_text(text) == expected, "preprocess_text failed"
    print("test_preprocess_text passed")

def test_generate_bigrams():
    tokens = ['hello', 'world', 'this', 'is', 'a', 'test']
    expected = [('hello', 'world'), ('world', 'this'), ('this', 'is'), ('is', 'a'), ('a', 'test')]
    assert generate_bigrams(tokens) == expected, "generate_bigrams failed"
    print("test_generate_bigrams passed")

def test_create_bigram_counts():
    bigrams = [('hello', 'world'), ('hello', 'world'), ('world', 'test')]
    counts = create_bigram_counts(bigrams)
    assert counts[('hello', 'world')]['world'] == 1, "create_bigram_counts failed"
    print("test_create_bigram_counts passed")

def test_compute_bigram_probabilities():
    counts = defaultdict(lambda: defaultdict(int), {("hello", "world"): {"test": 2, "is": 1}})
    probs = compute_bigram_probabilities(counts)
    assert round(probs[("hello", "world")]["test"], 2) == 0.67, "compute_bigram_probabilities failed"
    print("test_compute_bigram_probabilities passed")
def test_create_trigram_counts():
    trigrams = [('hello', 'world', 'this'), ('hello', 'world', 'this'), ('world', 'this', 'is')]
    counts = create_trigram_counts(trigrams)
    assert counts[('hello', 'world', 'this')]['this'] == 1, "create_trigram_counts failed"
    print("test_create_trigram_counts passed")

def test_compute_trigram_probabilities():
    counts = defaultdict(lambda: defaultdict(int), {("hello", "world", "this"): {"is": 2, "a": 1}})
    probs = compute_trigram_probabilities(counts)
    assert round(probs[("hello", "world", "this")]["is"], 2) == 0.67, "compute_trigram_probabilities failed"
    print("test_compute_trigram_probabilities passed")

def test_create_quadgram_counts():
    quadgrams = [('hello', 'world', 'this', 'is'), ('hello', 'world', 'this', 'is'), ('world', 'this', 'is', 'a')]
    counts = create_quadgram_counts(quadgrams)
    assert counts[('hello', 'world', 'this', 'is')]['is'] == 1, "create_quadgram_counts failed"
    print("test_create_quadgram_counts passed")

def test_compute_quadgram_probabilities():
    counts = defaultdict(lambda: defaultdict(int), {("hello", "world", "this", "is"): {"a": 2, "test": 1}})
    probs = compute_quadgram_probabilities(counts)
    assert round(probs[("hello", "world", "this", "is")]["a"], 2) == 0.67, "compute_quadgram_probabilities failed"
    print("test_compute_quadgram_probabilities passed")

def test_generate_trigrams():
    tokens = ['hello', 'world', 'this', 'is', 'a', 'test']
    expected = [('hello', 'world', 'this'), ('world', 'this', 'is'), ('this', 'is', 'a'), ('is', 'a', 'test')]
    assert generate_trigrams(tokens) == expected, "generate_trigrams failed"
    print("test_generate_trigrams passed")

def test_generate_quadgrams():
    tokens = ['hello', 'world', 'this', 'is', 'a', 'test']
    expected = [('hello', 'world', 'this', 'is'), ('world', 'this', 'is', 'a'), ('this', 'is', 'a', 'test')]
    assert generate_quadgrams(tokens) == expected, "generate_quadgrams failed"
    print("test_generate_quadgrams passed")

def test_sample_next_token():
    ngram_probs = {('hello', 'world'): {'test': 1.0}}
    assert sample_next_token(('hello', 'world'), ngram_probs) == 'test', "sample_next_token failed"
    print("test_sample_next_token passed")

def test_generate_text_from_bigram():
    bigram_probs = {('hello', 'world'): {'test': 1.0}, ('world', 'test'): {'case': 1.0}}
    result = generate_text_from_bigram(('hello', 'world'), 4, bigram_probs)
    assert result == "hello world test case", "generate_text_from_bigram failed"
    print("test_generate_text_from_bigram passed")

def test_generate_text_from_trigram():
    trigram_probs = {('hello', 'world', 'this'): {'is': 1.0}, ('world', 'this', 'is'): {'a': 1.0}}
    result = generate_text_from_trigram(('hello', 'world', 'this'), 5, trigram_probs)
    assert result == "hello world this is a", "generate_text_from_trigram failed"
    print("test_generate_text_from_trigram passed")

def test_generate_text_from_quadgram():
    quadgram_probs = {('hello', 'world', 'this', 'is'): {'a': 1.0}}
    result = generate_text_from_quadgram(('hello', 'world', 'this', 'is'), 5, quadgram_probs)
    assert result == "hello world this is a", "generate_text_from_quadgram failed"
    print("test_generate_text_from_quadgram passed")

if __name__ == "__main__":
    test_preprocess_text()
    test_generate_bigrams()
    test_create_bigram_counts()
    test_compute_bigram_probabilities()
    test_generate_trigrams()
    test_generate_quadgrams()
    test_sample_next_token()
    test_create_trigram_counts()
    test_compute_trigram_probabilities()
    test_create_quadgram_counts()
    test_compute_quadgram_probabilities()
    test_generate_text_from_bigram()
    test_generate_text_from_trigram()
    test_generate_text_from_quadgram()
    print("All tests completed successfully.")
