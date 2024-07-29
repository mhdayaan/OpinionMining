#!/usr/bin/env python
# coding: utf-8
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
import string

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Set up stopwords and punctuation
stop = set(stopwords.words('english'))
punc = set(string.punctuation)

# Initialize the lemmatizer
lemma = WordNetLemmatizer()

def clean_text(text):
    # Convert the text into lowercase
    text = text.lower()
    # Split into list
    wordList = text.split()
    # Remove punctuation
    wordList = ["".join(x for x in word if (x=="'")|(x not in punc)) for word in wordList]
    # Remove stopwords
    wordList = [word for word in wordList if word not in stop]
    # Lemmatisation
    wordList = [lemma.lemmatize(word) for word in wordList]
    return " ".join(wordList)

vocab_size = 5000
embedding_dim = 300
max_length = 1450
trunc_type='post'
oov_tok = "<OOV>"


# load weights into new model
classifier = load_model('CNN_LSTM_W2V_c1.h5')
print("Loading saved model...")
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)

def predict_response(response):
    response = list(map(str, response))
    response = list(map(clean_text, response))
    tokenizer.fit_on_texts(response)
    response = tokenizer.texts_to_sequences(response)
    response = pad_sequences(response, maxlen=max_length)
    result = classifier.predict(response)
    pred = np.argmax(result)
    
    if pred == 0:
        return 'NEGATIVE'
    elif pred == 1:
        return 'POSITIVE'






