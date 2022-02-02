# %%
#importing the required libraries
import numpy as np
import pandas as pd
import pickle
from models import matrix_factorization_utilities
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split

#import libraries
import keras
from keras.layers import Embedding, Reshape, concatenate
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

# %%
# Reading the ratings data
ratings = pd.read_csv('Dataset/ratings.csv')

# %%
#Just taking the required columns
ratings = ratings[['userId', 'movieId','rating']]

# %%
#reading the movies dataset
movie_list = pd.read_csv('Dataset/movies.csv')

# %%
# Couting no of unique users and movies
len(ratings.userId.unique()), len(ratings.movieId.unique())

# %%
# Assigning a unique value to each user and movie in range 0,no_of_users and 0,no_of_movies respectively.
ratings.userId = ratings.userId.astype('category').cat.codes.values
ratings.movieId = ratings.movieId.astype('category').cat.codes.values

# %%
# Splitting the data into train and test.
train, test = train_test_split(ratings, test_size=0.2)

# %%
train.head

# %%
test.head

# %%
n_users, n_movies = len(ratings.userId.unique()), len(ratings.movieId.unique())

# %%
# Returns a neural network model which does recommendation
#def neural_network_model(n_latent_factors_user, n_latent_factors_movie):
    
movie_input = keras.layers.Input(shape=[1],name='Item')
movie_embedding = keras.layers.Embedding(n_movies + 1, 50, name='Movie-Embedding')(movie_input)
# 13 yerinen_latent_factors_movie
movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)
movie_vec = keras.layers.Dropout(0.2)(movie_vec)


user_input = keras.layers.Input(shape=[1],name='User')
user_vec = keras.layers.Flatten(name='FlattenUsers')(keras.layers.Embedding(n_users + 1, 20,name='User-Embedding')(user_input))
#10 yerine  n_latent_factors_user
user_vec = keras.layers.Dropout(0.2)(user_vec)


concat = keras.layers.concatenate([movie_vec, user_vec],name='Concat')
concat_dropout = keras.layers.Dropout(0.2)(concat)
dense = keras.layers.Dense(100,name='FullyConnected')(concat)
dropout_1 = keras.layers.Dropout(0.2,name='Dropout')(dense)
dense_2 = keras.layers.Dense(50,name='FullyConnected-1')(concat)
dropout_2 = keras.layers.Dropout(0.2,name='Dropout')(dense_2)
dense_3 = keras.layers.Dense(20,name='FullyConnected-2')(dense_2)
dropout_3 = keras.layers.Dropout(0.2,name='Dropout')(dense_3)
dense_4 = keras.layers.Dense(10,name='FullyConnected-3', activation='relu')(dense_3)


result = keras.layers.Dense(1, activation='relu',name='Activation')(dense_4)
adam = Adam(lr=0.001)
model = keras.Model([user_input, movie_input], result)
model.compile(optimizer=adam,loss= 'mean_absolute_error')

# %%
model.summary()

# %%
history_neural_network = model.fit([train.userId, train.movieId], train.rating, epochs=200, validation_data=0.1,verbose=1)

# %%
y_hat = np.round(model.predict([test.userId, test.movieId]),0)
y_true = test.rating

# %%
mean_absolute_error(y_true, y_hat)


