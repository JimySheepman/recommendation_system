# %%
import warnings
warnings.filterwarnings("ignore")

# %%
import numpy as np 
import pandas as pd

# %%
rs_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
rs_cols1=['movie_id', 'movie_name','genre']

ratings= pd.read_csv('Dataset/ml-1m/ratings.dat', sep='::', names=rs_cols, encoding='latin-1')
movies= pd.read_csv('Dataset/ml-1m/movies.dat', sep='::', names=rs_cols1, encoding='latin-1')

# %%
ratings = pd.merge(ratings, movies, on= 'movie_id')

# %%
ratings

# %%
ratings = ratings.iloc[:,].reset_index(drop=True)
ratings= ratings.drop(["unix_timestamp"],axis=1).drop(["genre"], axis=1)

# %%
ratings

# %%
rating_t = ratings.copy()


# %%
rating_t

# %%

rating_stack = rating_t

# %%

rating_stack2 = rating_stack.drop(rating_stack[rating_stack['rating']=='0'].index, axis=1) 
rating_stack2

# %%
uniques = sorted(rating_stack2['rating'].unique())
print('Unique values in rating: {}'.format(uniques))
#rating_stack2.rating.replace('6','5',inplace=True)

# %%

import matplotlib.pyplot as plt
%matplotlib inline

plt.hist(rating_stack2.rating, bins=10)
plt.title('Histogram of Movie Ratings')
plt.show()

# %%
from sklearn.model_selection import train_test_split
train, test = train_test_split(ratings, test_size=0.2)

# %%
train


# %%
test

# %%
y_true = test.rating 

# %%
y_true

# %%
import keras

n_latent_factors_user = 20
n_latent_factors_movie = 50
n_latent_factors_mf = 5
n_users = ratings['user_id'].unique().max()
n_movies = ratings['movie_id'].unique().max()

# %%
n_users

# %%
n_movies

# %%
movie_input = keras.layers.Input(shape=[1],name='Item')
movie_embedding_mlp = keras.layers.Embedding(n_movies + 1, n_latent_factors_movie, name='Movie-Embedding-MLP')(movie_input)
movie_vec_mlp = keras.layers.Flatten(name='FlattenMovies-MLP')(movie_embedding_mlp)
movie_vec_mlp = keras.layers.Dropout(0.2)(movie_vec_mlp)

# %%
movie_embedding_mf = keras.layers.Embedding(n_movies + 1, n_latent_factors_mf, name='Movie-Embedding-MF')(movie_input)
movie_vec_mf = keras.layers.Flatten(name='FlattenMovies-MF')(movie_embedding_mf)
movie_vec_mf = keras.layers.Dropout(0.2)(movie_vec_mf)

# %%
user_input = keras.layers.Input(shape=[1],name='User')
user_vec_mlp = keras.layers.Flatten(name='FlattenUsers-MLP')(keras.layers.Embedding(n_users + 1, n_latent_factors_user,name='User-Embedding-MLP')(user_input))
user_vec_mlp = keras.layers.Dropout(0.2)(user_vec_mlp)

# %%
user_vec_mf = keras.layers.Flatten(name='FlattenUsers-MF')(keras.layers.Embedding(n_users + 1, n_latent_factors_mf,name='User-Embedding-MF')(user_input))
user_vec_mf = keras.layers.Dropout(0.2)(user_vec_mf)

# %%
concat = keras.layers.concatenate([movie_vec_mlp, user_vec_mlp], axis=1, name='Concat')

# %%
concat_dropout = keras.layers.Dropout(0.2)(concat)
dense = keras.layers.Dense(200,name='FullyConnected', activation='relu')(concat_dropout)
dense_batch = keras.layers.BatchNormalization(name='Batch')(dense)
dropout_1 = keras.layers.Dropout(0.2,name='Dropout-1')(dense_batch)
dense_2 = keras.layers.Dense(100,name='FullyConnected-1',activation='relu')(dropout_1)
dense_batch_2 = keras.layers.BatchNormalization(name='Batch-2')(dense_2)


# %%
dropout_2 = keras.layers.Dropout(0.2,name='Dropout-2')(dense_batch_2)
dense_3 = keras.layers.Dense(50,name='FullyConnected-2', activation='relu')(dropout_2)
dense_4 = keras.layers.Dense(20,name='FullyConnected-3', activation='relu')(dense_3)

# %%
pred_mf = keras.layers.Multiply()([movie_vec_mf, user_vec_mf])
pred_mlp = keras.layers.Dense(1, activation='relu',name='Activation')(dense_4)
combine_mlp_mf = keras.layers.concatenate([pred_mf, pred_mlp],name='Concat-MF-MLP')

# %%
result_combine = keras.layers.Dense(100,name='Combine-MF-MLP')(combine_mlp_mf)
deep_combine = keras.layers.Dense(100,name='FullyConnected-4')(result_combine)

# %%
result = keras.layers.Dense(1,name='Prediction', activation='relu')(result_combine)
model = keras.Model([user_input, movie_input], result)
opt = keras.optimizers.Adam(lr =0.01)
model.compile(optimizer='adam',loss= 'mean_absolute_error')

# %%
model.summary()

# %%
history = model.fit([train.user_id, train.movie_id], train.rating, epochs=5, batch_size=128,verbose=1, validation_split=0.1)

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
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
y_true = test.rating

x_pred=np.round(model.predict([test.user_id, test.movie_id]),0)

# %%

rmse = np.sqrt(mean_squared_error(y_true, x_pred))
errors = mean_absolute_error(y_true, x_pred)
mape = 100 * (errors / y_true)
accuracy = 100 - np.mean(mape)
print('MAE for testing: {}'.format(round(mean_absolute_error(y_true, x_pred)), 4))
print("Root Mean Square Error: {} ".format(round(rmse, 4)))
print('Accuracy: {} %.'.format(round(accuracy, 2)))

# %%
mlp_user_embedding_weights = (next(iter(filter(lambda x: x.name == 'User-Embedding-MLP', model.layers))).get_weights())

# get the latent embedding for your desired user
user_latent_matrix = mlp_user_embedding_weights[0]

desired_user_id = 38 # User: Xinyue Liu 
one_user_vector = user_latent_matrix[desired_user_id,:]
one_user_vector = np.reshape(one_user_vector, (1,10))

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
    movies += list(ratings[ratings['user_id'] == int(user_id)]['movie_id'])
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
results = pd.DataFrame(results, columns=['Rating']).astype("float")
items = pd.DataFrame(items, columns=['movie_id'])
results = pd.concat([items, results], ignore_index=True, sort=False, axis=1)
results.columns =['movie_id', 'Rating'] 

results_df = pd.DataFrame(np.nan, index=range(len(results)), columns=['Rating','Movie'])

# loop through and get the ratings (of being interested by desired user according to my model)
for index, row in results.iterrows():
    results_df.loc[index] = [row['Rating'], ratings[ratings['movie_id'] == row['movie_id']] #.iloc[0]['movie_id']]
                         
results_df = results_df.sort_values(by=['Rating'], ascending=False)

results_df.head(5)

# %%
results = pd.DataFrame(results, columns=['Rating']).astype("float")
items = pd.DataFrame(items, columns=['movie_id'])
results = pd.concat([items, results], ignore_index=True, sort=False, axis=1)
results.columns =['movie_id', 'Rating'] 

# %%
results

# %%
items

# %%
results

# %%
results

# %%
results_df = pd.DataFrame(np.nan, index=range(len(results)), columns=['Rating','Movie'])

# %%
results_df

# %%
for index, row in results.iterrows():
    results_df.loc[index] = [row['Rating'], ratings[ratings['movie_id'] == row['movie_id']].iloc[0]['movie_id']]
                         
results_df = results_df.sort_values(by=['Rating'], ascending=False)

results_df.head(5)


