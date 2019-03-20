#Creating a predictive model to predict the price of wine.
#Using Keras with a TF backend as approach.
#-----------------------------------------

#---------------imports---------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense, Activation

print(tf.__version__)
print("-------------------")
print("")

def data_summary(X_train, y_train, X_test, y_test):
    """Summarize current state of dataset"""
    print('Train images shape:', X_train.shape)
    print('Train labels shape:', y_train.shape)
    print('Test images shape:', X_test.shape)
    print('Test labels shape:', y_test.shape)
    #print('Train labels:', y_train)
    #print('Test labels:', y_test)

concept = "/concept"

#load data
train_data_raw = pd.read_csv('../data/training_data_raw.csv')
test_data_raw = pd.read_csv('../data/test_data_raw.csv')
data_raw = pd.read_csv('../data/winemag-data-130k-v2.csv')

#method used is adapted from:
#https://cloud.google.com/blog/products/gcp/intro-to-text-classification-with-keras-automatically-tagging-stack-overflow-posts

#---------------Data encoding---------------

#Encoding for 'Description' based on the alphabet
#.create Instance
#.fit the alphabet
#.encode train data
tokenize = Tokenizer()
tokenize.fit_on_texts(data_raw['description'][:data_raw.shape[0]])
train_data_description = tokenize.texts_to_matrix(train_data['description'][:train_data.shape[0]])

#Encode countries to OneHot.
#.creating labels from 'countries'
encoder = LabelEncoder()
encoder.fit(data_raw['country'][:data_raw.shape[0]].tolist())
all_countries = encoder.transform(data_raw['country'][:data_raw.shape[0]].tolist())

#.create one-hot-encoder instance
#.fit dataset of all countries
#.encode train data
one_hot_encoder = OneHotEncoder()
one_hot_encoder.fit(all_countries.reshape(len(all_countries), 1))
train_data_countries = one_hot_encoder.transform(train_data_countries.reshape(len(train_data_countries), 1))

#Encode next feature....

#--------------create X_train and Y_train---------------
#.append all sub matrixes to a x_train


#create y_train from the 'price'
#encode the price to 0 to 1
#Using standardscaler for normalizing 'price'
price_encoder = StandardScaler()
price_encoder.fit(....)

#---------------model creation---------------
#sequential model
model = Sequential()
model.add(Dense(512, input_shape=(x_train.shape[1],)))
model.add(Activation('sigmoid'))
model.add(Dense(256, activation='sigmoid'))
#output layer
model.add(Dense(1), activation='sigmoid'))
#---------------compile and train model---------------

batch_size = 32
no_of_epochs = 5

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=no_of_epochs,
                    verbose=1)

#---------------Testing---------------
#prepare test sets
#same approach as for training data
#use created encoder instances and transform test_data

#test the model
score = model.evaluate(test_x, test_y,
                     verbose=1)

print('Test score:', score[0])
print('Test accuracy:', score[1])

#save test scores
score_pd = pd.read_csv('base_score.csv')
new_scores_pd = pd.DataFrame({'TestScore': [score[0]], 'Test accuracy': [score[1]], 'Method': [concept], 'Hyperparameters': ["Batch Size = " + str(batch_size) + "; Number of Epochs = " + str(no_of_epochs)]})
score_pd = score_pd.append(new_scores_pd, sort=False)

score_pd.to_csv('base_score.csv', index=False)
