#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import pandas as pd
from pandas import read_excel
import numpy as np
#import re
from re import sub
#import multiprocessing
from unidecode import unidecode
#import os
from time import time 
#import tensorflow as tf
#import keras
"""from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout,Activation,Embedding,Flatten,Bidirectional,MaxPooling2D, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import regularizers"""
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
#import h5py
#import csv
##import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import StratifiedKFold
import nltk
from nltk import tokenize
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
#from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
#import string


# In[10]:


vocab_size = 5000
embedding_dim = 300
max_length = 100
trunc_type='post'
oov_tok = "<OOV>"


# In[22]:


from keras.models import load_model
from keras.preprocessing import sequence
import numpy as np
# load weights into new model
classifier = load_model('CNN_LSTM.h5')
print("Loading saved model...")
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)

def predict_response(response):
    '''
    0 - very negative
    1 - somewhat negative
    2 - neutral
    3 - somewhat positive
    4 - very positive
    '''
    print(f'You said: \n{response}\n')
    tokenizer.fit_on_texts(response)
    word_index = tokenizer.word_index
    print(len(word_index))
    response = tokenizer.texts_to_sequences(response)
    response = pad_sequences(response, maxlen=max_length)
    result = classifier.predict(response)
    pred = np.argmax(result)
    print(pred)
    print(result)
    if pred == 0:
        return 'Negative'
    elif pred == 1:
        return 'Positive'


# In[ ]:




