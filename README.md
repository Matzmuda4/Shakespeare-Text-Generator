# Shakespeare-Text-Generator
A text generation model that works with 2-grams/3-grams/4-grams to imitate the style of William Shakespeare. This will be done by implement various functions that will help in generating text based on bigrams, trigrams and quadgrams.

## Setting up the environment
1. Create a Virtual Environment from terminal ```python -m venv venv``` and activate with ```venv\Scripts\activate``` for windows and ```source venv/bin/activate``` for Mac/linux
2. Install the dependancies ```pip install -r requirements.txt```
3. Run the Shakespeare_model.py by running ```python Shakespeare_model.py ```


## Approach and methodologies

### 1. Dataset Selection and Preprocessing

- **Dataset Selection:**  
  - I researched and chose a Shakespeare text file ```shakespeare.txt``` containing every single one of his complete works was used as the corpus.
  - I chose this dataset because it provided me with an extensive sample of shapespeares writing for the best text generation covering all aspects of his writing based on historical data.

- **Text Preprocessing:**  
  - In order to prepare my data for generation and n-gram sampling, the text file was converted to lowercase to ensure a more uniform text to process.
  - I removed all Punctuation to simplify the tokenization process and ensure accuracy.
  - I then Tokenized the text using NLTK’s `word_tokenize()` function to split the text into individual words.

---

### 2. Bigram Model Implementation

- **Bigram Generation:**  
  - I Extracted all possible Bigrams from the text using `nltk.bigrams(tokens)`.

- **Bigram Frequency Count:**  
  - I created a dictionary where each bigram is a key, and the count of words that follow the bigram is stored.
  - I implemented this dictionary: `from_bigram_to_next_token_counts` using `defaultdict(int)`.

- **Bigram Probability Computation:**  
  - What I did to compute probabilites was convert the frequency counts of each proceeding word of a bigram into probabilities by normalizing the counts.
  - This resulted in `from_bigram_to_next_token_probs`, which maps each bigram to a dictionary of next-word probabilities.

- **Bigram Sampling:**  
  - I implemented A function called `sample_next_token()`, to randomly sample the next word based on its previously calculated probability distribution.
  - This ensured that words with higher frequency were more likely to be chosen for generation.

- **Bigram Text Generation:**  
  - I implemented the function `generate_text_from_bigram()` to generate text starting from a given bigram.
  - It iteratively selects the next word using `sample_next_token()` until the predetermined word count is reached.

---

## 3. Extending to Trigrams

- **Trigram Generation:**  
  - The same process used for bigrams was extended to trigrams, jeeping the same logic slightly adjusted
  - `nltk.trigrams(tokens)` was used to extract trigrams.

- **Trigram Frequency Count:**  
  - A dictionary, `from_trigram_to_next_token_counts`, was created to map each trigram to its next-word counts.

- **Trigram Probability Computation:**  
  - The frequency counts were converted into probabilities in `from_trigram_to_next_token_probs`.

- **Trigram Text Generation:**  
  - The `generate_text_from_trigram()` function was implemented to generate text using trigrams.
  - The sampling logic remained the same but now considers three-word contexts.

---

## 4. Extending to Quadgrams

- **Quadgram Generation:**  
  - The model was extended further to extract four-word sequences using `nltk.ngrams(tokens, 4)`.

- **Quadgram Frequency Count:**  
  - `from_quadgram_to_next_token_counts` was created to store quadgram-based next-word counts.

- **Quadgram Probability Computation:**  
  - Normalization was applied to generate probabilities in `from_quadgram_to_next_token_probs`.

- **Quadgram Text Generation:**  
  - The `generate_text_from_quadgram()` function was implemented similarly to the bigram and trigram functions.
  - This allowed for more contextualized text generation which I later assessed in my human evaluation.

---

## 5. Testing and Validation

- **Automated Test Cases:**  
  - A separate file, `test.py`, was created to systematically test each function.
  - These tests were made one by one with each function to ensure that they worked properly as I wrote more and more code, the test cases were created with sample and simple text just to ensure the most basic functionality, they were then asserted and all of the following passed:
    - `preprocess_text()`: Ensures text is correctly tokenized.
    - `generate_bigrams()`, `generate_trigrams()`, `generate_quadgrams()`: Ensures proper n-gram extraction.
    - `create_bigram_counts()`, `create_trigram_counts()`, `create_quadgram_counts()`: Validates frequency counts.
    - `compute_bigram_probabilities()`, `compute_trigram_probabilities()`, `compute_quadgram_probabilities()`: Ensures correct probability calculations.
    - `sample_next_token()`: Ensures next-word selection follows the probability distribution.
    - `generate_text_from_bigram()`, `generate_text_from_trigram()`, `generate_text_from_quadgram()`: Ensures text is generated correctly.

- **Debugging and Improvements:**  
  - Debugging was performed to ensure text generation continued beyond short phrases, also to assess how each N-gram generated text looked.
  - Adjustments were made to handle errors such as the probabilities being incorrect and to ensure smooth generation.

---

## 6. Evaluation and Analysis

- **Comparison of Bigram, Trigram, and Quadgram Models:**  
  - Bigrams often resulted in either repetitive sequences or completely gramatically wrong due to limited context.
  - Trigrams improved coherence by considering an additional word in the sequence.
  - Quadgrams produced the most structured outputs and I understand that they require larger data to avoid rare occurrences.
  - Due to the large and accurate shakespearean corpus, all of the generated texts correctly used words and had some loose resemblance to his writing. 

- **Human Evaluation:**  
  - A qualitative evaluation was performed by analyzing the generated text through a survey, I first showed participants the code and the skakespeare.txt file then had them fill out the survey.
  - I then asked first questions to guage a participants knowledge of shakespeares works by giving them a sample from shakespeare.txt and asking if it they thought it was shakespeare who wrote it.
  - The quality of generated text was then assessed for fluency and Shakespearean style imitation by asking the participants if they thought that text generated through bigrams, trigrams and quadgrams sounded like it was written by him.
  - The results showcased a similar trend of people who knew him farely well saying that the bigrams were not too similar, the trigrams were half half and the quadgrams were almost completely him, with the occasional participant seeing through the generated text and immediately noticing that it wasn't shakespeares writing.

---

## 7. Conclusion

- **Final Model:**  
  - A functional text generation model using bigrams, trigrams, and quadgrams was successfully implemented.
  - The **quadgram model produced the best results** in terms of readability and coherence.
  - The **bigram model worked best when only short sequences were needed**.
  - The **trigram model balanced coherence and variety** effectively.

---