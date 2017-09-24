import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
import string


# fill out the function below that transforms the input series
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    index = 0
    while window_size < len(series):
        X.append(series[index:window_size])
        y.append(series[window_size])
        window_size += 1
        index += 1
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    return model


### return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    # use ascii letters
    ascii_letters = string.ascii_letters
    punctuation = ['!', ',', '.', ':', ';', '?']
    # find all unique characters in the text
    unique_char = ''.join(set(text))
    # remove the non-english characters from text
    for char in unique_char:
        if char not in ascii_letters and char not in punctuation:
            text = text.replace(char, ' ')
    return text

### fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    index = 0
    while index + window_size < len(text):
        inputs.append(text[index:index+window_size])
        outputs.append(text[index+window_size])
        index += step_size
    return inputs,outputs

# build the required RNN model:
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars, activation='softmax'))
    return model

