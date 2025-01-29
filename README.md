# Shakespeare-Text-Generator
A text generation model that works with 2-grams/3-grams/4-grams to imitate the style of William Shakespeare. This will be done by implement various functions that will help in generating text based on bigrams, trigrams and quadgrams.

## Setting up the environment
1. Create a Virtual Environment from terminal ```python -m venv venv``` and activate with ```venv\Scripts\activate``` for windows and ```source venv/bin/activate``` for Mac/linux
2. Install the dependancies ```pip install -r requirements.txt```


## Approach and methodologies
After task 1 was completed, i went through the text and assessed the output of each bigram in the dictionary compared to its "next word count" to ensure correctness of the preprocessing steps and found 100 percent accuracy when analyzing the text that had been successfully transformed into lower case, no punctuation and tokenized word by word. 