# %% [markdown]
# # 1.1-) import libraries

# %%
# import essential basic libraries 
import pandas as pd
import numpy as np
from math import sqrt
from scipy.sparse.linalg import svds

# import matrix_factorization_utilities.py
from models import matrix_factorization_utilities

# import machine learning libraries 
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# import visualization  libraries 
import matplotlib.pyplot as plt
%matplotlib inline

# import deep learning libraries 
import keras
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Activation, Flatten ,Embedding, Reshape, concatenate

# import warnings libraries for close warnings
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# #  1.2-) Dataset import and config

# %%
# Reading the ratings data
ratings = pd.read_csv('Dataset/ratings.csv')
#Just taking the required columns
ratings = ratings[['userId', 'movieId','rating']]
#reading the movies dataset
movie_list = pd.read_csv('Dataset/movies.csv')

# %%
#get ordered list of movieIds
item_indices = pd.DataFrame(sorted(list(set(ratings['movieId']))),columns=['movieId'])
#add in data frame index value to data frame
item_indices['movie_index']=item_indices.index
#inspect data frame
item_indices.head()

# %%
#get ordered list of movieIds
user_indices = pd.DataFrame(sorted(list(set(ratings['userId']))),columns=['userId'])
#add in data frame index value to data frame
user_indices['user_index']=user_indices.index
#inspect data frame
user_indices.head()

# %%
#join the movie indices
df_with_index = pd.merge(ratings,item_indices,on='movieId')
#join the user indices
df_with_index=pd.merge(df_with_index,user_indices,on='userId')
#inspec the data frame
df_with_index.head()

# %%
# Assigning a unique value to each user and movie in range 0,no_of_users and 0,no_of_movies respectively.
ratings.userId = ratings.userId.astype('category').cat.codes.values
ratings.movieId = ratings.movieId.astype('category').cat.codes.values

# %%
#take 80% as the training set and 20% as the test set
train, test= train_test_split(ratings,test_size=0.2)
train.shape,test.shape

# %%
train.head

# %%
train.movieId.max()

# %%
test.head

# %%
n_users = ratings.userId.unique().shape[0]
n_items = ratings.movieId.unique().shape[0]
n_users, n_items

# %%
#Create two user-item matrices, one for training and another for testing
train_data_matrix = np.zeros((n_users, n_items))
    #for every line in the data
for line in train.itertuples():
    #set the value in the column and row to 
    #line[1] is userId, line[2] is movieId and line[3] is rating, line[4] is movie_index and line[5] is user_index
    train_data_matrix[line[1]-1,line[2]-1] = line[3]
train_data_matrix.shape

# %%
#Create two user-item matrices, one for training and another for testing
test_data_matrix = np.zeros((n_users, n_items))
    #for every line in the data
for line in test.itertuples():
    #set the value in the column and row to 
    #line[1] is userId, line[2] is movieId and line[3] is rating, line[4] is movie_index and line[5] is user_index
    #print(line[2])
    test_data_matrix[line[1]-1,line[2]-1] = line[3]
test_data_matrix.shape

# %%
pd.DataFrame(train_data_matrix).head()

# %%
train['rating'].max()

# %% [markdown]
# # 2-) svd to the test data and calculate rmse score of matrix factorisation for find best n_latent

# %%
# accuracy bulunacak

# %%
#Calculate the rmse sscore of SVD using different values of k (latent features)
rmse_list = []
mae_list = []
accuracy=[]
pred_list=[]
a=[1,2,5]
for i in a:
    #apply svd to the test data
    u,s,vt = svds(train_data_matrix,k=i)
    #get diagonal matrix
    s_diag_matrix=np.diag(s)
    #predict x with dot product of u s_diag and vt
    X_pred = np.dot(np.dot(u,s_diag_matrix),vt)
    pred_list.append(X_pred)
    #calculate rmse score of matrix factorisation predictions
    mse =mean_squared_error(test_data_matrix, X_pred)
    
    rmse_score = sqrt(mse)
    rmse_list.append(round(rmse_score,4))
    mae_score = mean_absolute_error(test_data_matrix,X_pred)
    mae_list.append(round(mae_score,4))

# %%
min_value_rmse=min(rmse_list)
min_index_rmse=rmse_list.index(min_value_rmse)
min_value_mae=min(mae_list)
min_index_mae=mae_list.index(min_value_mae)

print("Matrix Factorisation with " + str(a[min_index_rmse]) +" latent features has a RMSE of " + str(min_value_rmse))
print("Matrix Factorisation with " + str(a[min_index_mae]) +" latent features has a MAE of " + str(min_value_mae))

# %%
import keras
from keras.layers import Embedding, Reshape
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

# %%
movie_input = keras.layers.Input(shape=[1],name='Item')
movie_embedding = keras.layers.Embedding(14026 + 1, 1, name='Movie-Embedding')(movie_input)
movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)
user_input = keras.layers.Input(shape=[1],name='User')
user_vec = keras.layers.Flatten(name='FlattenUsers')(keras.layers.Embedding(7120 + 1, 1,name='User-Embedding')(user_input))
prod = keras.layers.concatenate([movie_vec, user_vec],name='DotProduct')
result = keras.layers.Dense(1, activation='relu',name='Activation')(prod)
adam = Adam(lr=0.005)
model = keras.Model([user_input, movie_input], result)
model.compile(optimizer=adam,loss= 'mean_absolute_error')

# %%
model.summary()

# %%
history = model.fit([train.userId, train.movieId], train.rating,  epochs=10,batch_size=64,verbose=1, validation_split=0.1)

# %%
y_hat = np.round(model.predict([test.userId, test.movieId]),0)
y_true = test.rating

# %%
mean_absolute_error(y_true, y_hat)

# %%
def accuracy(y_true,y_hat):
    errors=mean_absolute_error(y_true, y_hat)
    mape = 100 * (errors / y_true)
    accuracy = 100 - np.mean(mape)
    return accuracy
print("(MAE)Mean Absolute Error:",round(mean_absolute_error(y_true, y_hat),4))
print("(RMSE)Root Mean Square Error:",round(np.sqrt(mean_squared_error(y_true,y_hat)) ,4))
print(' Accuracy:', round(accuracy(y_true,y_hat), 2), '%.')

# %%
model.save("models/matrix_factorisation_model_with_n_latent_factors.h5")

# %%
plt.plot(np.array(history.history['loss']))
plt.plot(np.array(history.history['val_loss']))
plt.ylabel('Mean Absolute Error')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.title('Mean Absolute Error Over Epochs')

# %% [markdown]
# ## 2.1-) matrix_factorisation_model_with_n_latent_factors recommendation

# %%
mlp_user_embedding_weights = (next(iter(filter(lambda x: x.name == 'User-Embedding', model.layers))).get_weights())

# get the latent embedding for your desired user
user_latent_matrix = mlp_user_embedding_weights[0]

desired_user_id = 38
one_user_vector = user_latent_matrix[desired_user_id,:]
one_user_vector = np.reshape(one_user_vector, (1,a[min_index_rmse]))

from sklearn.cluster import KMeans

print('\nPerforming kmeans to find the nearest users...')
# get similar users
kmeans = KMeans(n_clusters=20, random_state=0, verbose=0).fit(user_latent_matrix)
desired_user_label = kmeans.predict(one_user_vector)
user_label = kmeans.labels_
neighbors = []
for user_id, user_label in enumerate(user_label):
    if user_label == desired_user_label:
        neighbors.append(user_id)
print('Found {0} neighbor users.'.format(len(neighbors)))

# %%
    # get the movies in 3 similar users' movies
    movies = []
    result_list=[]
    for user_id in neighbors:
        movies += list(ratings[ratings['userId'] == int(user_id)]['movieId'])
    movies = list(set(movies))
    result_list.append('Found {0} neighbor movies from these users.'.format(len(movies)))

    users = np.full(len(movies), desired_user_id, dtype='int32')
    items = np.array(movies, dtype='int32')

    result_list.append('Ranking most likely tracks using the NeuMF model...')
    # and predict movies for my user
    results = model.predict([users,items],batch_size=10, verbose=0) 
    results = results.tolist()
    result_list.append('Ranked the movies!')

    results = pd.DataFrame(results, columns=['pre_rating']).astype("float")
    items = pd.DataFrame(items, columns=['movieId'])
    results = pd.concat([items, results], ignore_index=True, sort=False, axis=1)
    results.columns =['movieId', 'pre_rating'] 
    results_df = pd.DataFrame(np.nan, index=range(len(results)), columns=['pre_rating','movieId'])
    for index, row in results.iterrows():
        results_df.loc[index] = [row['pre_rating'], ratings[ratings['movieId'] == row['movieId']].iloc[0]['movieId']]

    results_df= results_df.sort_values(by=['pre_rating'], ascending=False)
    results_df["movieId"]=results_df["movieId"].astype(int)
    results_df=pd.merge(results_df,movie_list,on="movieId")[:10]


# %%
results_df

# %% [markdown]
# ## 2.2-)svd matrix factorisation recommendation

# %%
#Convert predictions to a DataFrame
mf_pred = pd.DataFrame(X_pred, )
mf_pred.head()

# %%
df_names = pd.merge(ratings,movie_list,on='movieId')
df_names.head()

# %%
#choose a user ID
user_id = 1
#get movies rated by this user id
users_movies = df_names.loc[df_names["userId"]==user_id]
#print how many ratings user has made 
print("User ID : " + str(user_id) + " has already rated " + str(len(users_movies)) + " movies")
#list movies that have been rated
users_movies.head()

# %%
# fonksiyon yazılacak
user_index = train.loc[train["userId"]==user_id]['userId'][:1].values[0]
#get movie ratings predicted for this user and sort by highest rating prediction
sorted_user_predictions = pd.DataFrame(mf_pred.iloc[user_index].sort_values(ascending=False))
#rename the columns
sorted_user_predictions.columns=['ratings']
#save the index values as movie id
sorted_user_predictions['movieId']=sorted_user_predictions.index
print("Top 10 predictions for User " + str(user_id))
#display the top 10 predictions for this user
pd.merge(sorted_user_predictions,movie_list, on = 'movieId')[:10]

# %% [markdown]
# # 3-) Neural network model 

# %%
n_users, n_movies = len(ratings.userId.unique()), len(ratings.movieId.unique())

# %%
n_users

# %%
n_movies

# %%
# Returns a neural network model which does recommendation
    
movie_input = keras.layers.Input(shape=[1],name='Item')
movie_embedding = keras.layers.Embedding(14026 + 1, 20, name='Movie-Embedding')(movie_input)
# 13 yerinen_latent_factors_movie
movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)
movie_vec = keras.layers.Dropout(0.2)(movie_vec)


user_input = keras.layers.Input(shape=[1],name='User')
user_vec = keras.layers.Flatten(name='FlattenUsers')(keras.layers.Embedding(7120 + 1, 50,name='User-Embedding')(user_input))
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
adam = Adam(lr=0.005)
model = keras.Model([user_input, movie_input], result)
model.compile(optimizer=adam,loss= 'mean_absolute_error')

# %%
model.summary()

# %%
history_neural_network = model.fit([train.userId, train.movieId], train.rating, epochs=10,batch_size=64,verbose=1, validation_split=0.1)

# %%
plt.plot(np.array(history_neural_network.history['loss']))
plt.plot(np.array(history_neural_network.history['val_loss']))
plt.ylabel('Mean Absolute Error')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.title('Mean Absolute Error Over Epochs')

# %%
y_hat = model.predict([test.userId, test.movieId])
y_true = test.rating

# %%
def accuracy(y_true,y_hat):
    errors=mean_absolute_error(y_true, y_hat)
    mape = 100 * (errors / y_true)
    accuracy = 100 - np.mean(mape)
    return accuracy

# %%
print("(MAE)Mean Absolute Error:",round(mean_absolute_error(y_true, y_hat),4))
print("(RMSE)Root Mean Square Error:",round(np.sqrt(mean_squared_error(y_true,y_hat)) ,4))
print(' Accuracy:', round(accuracy(y_true,y_hat), 2), '%.')

# %% [markdown]
# ## 3.1-) Neural network model recommendation

# %%
mlp_user_embedding_weights = (next(iter(filter(lambda x: x.name == 'User-Embedding', model.layers))).get_weights())

# get the latent embedding for your desired user
user_latent_matrix = mlp_user_embedding_weights[0]

desired_user_id = 38 # User: Xinyue Liu 
one_user_vector = user_latent_matrix[desired_user_id,:]
one_user_vector = np.reshape(one_user_vector, (1,50))

from sklearn.cluster import KMeans

print('\nPerforming kmeans to find the nearest users...')
# get similar users
kmeans = KMeans(n_clusters=20, random_state=0, verbose=0).fit(user_latent_matrix)
desired_user_label = kmeans.predict(one_user_vector)
user_label = kmeans.labels_
neighbors = []
for user_id, user_label in enumerate(user_label):
    if user_label == desired_user_label:
        neighbors.append(user_id)
print('Found {0} neighbor users.'.format(len(neighbors)))

# %%
# get the movies in 3 similar users' movies
movies = []
for user_id in neighbors:
    movies += list(ratings[ratings['userId'] == int(user_id)]['movieId'])
movies = list(set(movies))
print('Found {0} neighbor movies from these users.'.format(len(movies)))

# %%
users = np.full(len(movies), desired_user_id, dtype='int32')
items = np.array(movies, dtype='int32')

print('\nRanking most likely tracks using the NeuMF model...')
# and predict movies for my user
results = model.predict([users,items],batch_size=10, verbose=0) 
results = results.tolist()
print('Ranked the movies!')

# %%
results = pd.DataFrame(results, columns=['pre_rating']).astype("float")
items = pd.DataFrame(items, columns=['movieId'])
results = pd.concat([items, results], ignore_index=True, sort=False, axis=1)
results.columns =['movieId', 'pre_rating'] 
results_df = pd.DataFrame(np.nan, index=range(len(results)), columns=['pre_rating','movieId'])
for index, row in results.iterrows():
    results_df.loc[index] = [row['pre_rating'], ratings[ratings['movieId'] == row['movieId']].iloc[0]['movieId']]
   
results_df= results_df.sort_values(by=['pre_rating'], ascending=False)
results_df["movieId"]=results_df["movieId"].astype(int)
results_df=pd.merge(results_df,movie_list,on="movieId")

results_df.head(10)

# %%
model.save("models/neural_network_model.h5")

# %% [markdown]
# # 4-) Neural collaborative filtering model

# %%
y_true = test.rating 

# %%
n_users_base,n_items_base   = len(ratings.userId.unique()), len(ratings.movieId.unique())

# %%
n_users_base,n_items_base  

# %%
n_latent_factors_user = 50
n_latent_factors_movie = 25
n_latent_factors_mf = 3
n_users = ratings['userId'].unique().max()
n_movies = ratings['movieId'].unique().max()

# %%
n_users,n_movies

# %%
movie_input = keras.layers.Input(shape=[1],name='Item')
movie_embedding_mlp = keras.layers.Embedding(14026 + 1, 25, name='Movie-Embedding-MLP')(movie_input)
movie_vec_mlp = keras.layers.Flatten(name='FlattenMovies-MLP')(movie_embedding_mlp)
movie_vec_mlp = keras.layers.Dropout(0.2)(movie_vec_mlp)

movie_embedding_mf = keras.layers.Embedding(14026 + 1, 3, name='Movie-Embedding-MF')(movie_input)
movie_vec_mf = keras.layers.Flatten(name='FlattenMovies-MF')(movie_embedding_mf)
movie_vec_mf = keras.layers.Dropout(0.2)(movie_vec_mf)

user_input = keras.layers.Input(shape=[1],name='User')
user_vec_mlp = keras.layers.Flatten(name='FlattenUsers-MLP')(keras.layers.Embedding(7120 + 1, 50,name='User-Embedding-MLP')(user_input))
user_vec_mlp = keras.layers.Dropout(0.2)(user_vec_mlp)

user_vec_mf = keras.layers.Flatten(name='FlattenUsers-MF')(keras.layers.Embedding(7120 + 1, 3,name='User-Embedding-MF')(user_input))
user_vec_mf = keras.layers.Dropout(0.2)(user_vec_mf)

concat = keras.layers.concatenate([movie_vec_mlp, user_vec_mlp], axis=1, name='Concat')

concat_dropout = keras.layers.Dropout(0.2)(concat)
dense = keras.layers.Dense(200,name='FullyConnected', activation='relu')(concat_dropout)
dense_batch = keras.layers.BatchNormalization(name='Batch')(dense)
dropout_1 = keras.layers.Dropout(0.2,name='Dropout-1')(dense_batch)
dense_2 = keras.layers.Dense(100,name='FullyConnected-1',activation='relu')(dropout_1)
dense_batch_2 = keras.layers.BatchNormalization(name='Batch-2')(dense_2)

dropout_2 = keras.layers.Dropout(0.2,name='Dropout-2')(dense_batch_2)
dense_3 = keras.layers.Dense(50,name='FullyConnected-2', activation='relu')(dropout_2)
dense_4 = keras.layers.Dense(20,name='FullyConnected-3', activation='relu')(dense_3)

pred_mf = keras.layers.Multiply()([movie_vec_mf, user_vec_mf])
pred_mlp = keras.layers.Dense(1, activation='relu',name='Activation')(dense_4)
combine_mlp_mf = keras.layers.concatenate([pred_mf, pred_mlp],name='Concat-MF-MLP')

result_combine = keras.layers.Dense(100,name='Combine-MF-MLP')(combine_mlp_mf)
deep_combine = keras.layers.Dense(100,name='FullyConnected-4')(result_combine)

result = keras.layers.Dense(1,name='Prediction', activation='relu')(result_combine)
model_2 = keras.Model([user_input, movie_input], result)
opt = keras.optimizers.Adam(lr =0.01)
model_2.compile(optimizer='adam',loss= 'mean_absolute_error')

# %%
model_2.summary()

# %%
history = model_2.fit([train.userId, train.movieId], train.rating, epochs=10, verbose=1,batch_size=64, validation_split=0.1)

# %%
def plot_acc(history):
    plt.plot(np.array(history.history['loss']))
    plt.plot(np.array(history.history['val_loss']))
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'])
    plt.title('Mean Absolute Error Over Epochs')
    
plot_acc(history)

# %%
print('MAE for testing: {}'.format(round(mean_absolute_error(y_true, model_2.predict([test.userId, test.movieId])),4)))
rmse = np.sqrt(mean_squared_error(y_true,model_2.predict([test.userId, test.movieId]))) 
print("Root Mean Square Error:",round(rmse,4))
print('Accuracy:', round(accuracy(y_true, model_2.predict([test.userId, test.movieId])), 2), '%.')

# %%
model_2.save("models/neural_collaborative_filtering.h5")

# %% [markdown]
# ## 4.1-) # 4-) Neural collaborative filtering model recommendation

# %%
mlp_user_embedding_weights = (next(iter(filter(lambda x: x.name == 'User-Embedding-MLP', model_2.layers))).get_weights())

# get the latent embedding for your desired user
user_latent_matrix = mlp_user_embedding_weights[0]

desired_user_id = 38
one_user_vector = user_latent_matrix[desired_user_id,:]
one_user_vector = np.reshape(one_user_vector, (1,50))

from sklearn.cluster import KMeans

print('\nPerforming kmeans to find the nearest users...')
# get similar users
kmeans = KMeans(n_clusters=20, random_state=0, verbose=0).fit(user_latent_matrix)
desired_user_label = kmeans.predict(one_user_vector)
user_label = kmeans.labels_
neighbors = []
for user_id, user_label in enumerate(user_label):
    if user_label == desired_user_label:
        neighbors.append(user_id)
print('Found {0} neighbor users.'.format(len(neighbors)))

# %%
movies = []
for user_id in neighbors:
    movies += list(ratings[ratings['userId'] == int(user_id)]['movieId'])
movies = list(set(movies))
print('Found {0} neighbor movies from these users.'.format(len(movies)))

# %%
users = np.full(len(movies), desired_user_id, dtype='int32')
items = np.array(movies, dtype='int32')

print('\nRanking most likely tracks using the NeuMF model...')
# and predict movies for my user
results = model_2.predict([users,items],batch_size=10, verbose=0) 
results = results.tolist()
print('Ranked the movies!')

# %%
results = pd.DataFrame(results, columns=['pre_rating']).astype("float")
items = pd.DataFrame(items, columns=['movieId'])
results = pd.concat([items, results], ignore_index=True, sort=False, axis=1)
results.columns =['movieId', 'pre_rating'] 

results_df = pd.DataFrame(np.nan, index=range(len(results)), columns=['pre_rating','movieId'])

for index, row in results.iterrows():
    results_df.loc[index] = [row['pre_rating'], ratings[ratings['movieId'] == row['movieId']].iloc[0]['movieId']]
   
# output
results_df= results_df.sort_values(by=['pre_rating'], ascending=False)
results_df["movieId"]=results_df["movieId"].astype(int)
results_df=pd.merge(results_df,movie_list,on="movieId")

results_df.head(10)

# %%



