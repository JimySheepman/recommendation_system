# %%
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error

import math

# %%
rs_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=rs_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.base', sep='\t', names=rs_cols, encoding='latin-1')

# %%
n_users_base = ratings_base['user_id'].unique().max()
n_items_base = ratings_base['movie_id'].unique().max()

n_users_base,n_items_base

# %%
n_users_test = ratings_test['user_id'].unique().max()
n_items_test = ratings_test['movie_id'].unique().max()
n_users_test,n_items_test

# %%
train_matrix = np.zeros((n_users_base, n_items_base))
test_matrix = np.zeros((n_users_test, n_items_test))
for line in ratings_base.itertuples():
    train_matrix[line[1]-1,line[2]-1] = line[3]

# %%
test_matrix

# %%
train_matrix

# %%
arr=np.array([[1,2,3,],[4,5,6],[7,8,9]])#train matrix
arr2=np.array([[0,0,0],[0,0,0],[0,0,0]])#test matrix
arr2[0]=arr[0]
arr

# %%
arr2

# %%
arr[0]=0
arr

# %%
arr2

# %%
n_users_base

# %%
random_users_id_list=[]
while True:
    rnd=np.random.randint(943)
    if len(random_users_id_list) < 94:
        if rnd in random_users_id_list:
            continue
        else:
            random_users_id_list.append(rnd)
    else:
        break

random_users_id_list.sort()
random_users_id_list

# %%
users_first_rating_movie=[]
for user in random_users_id_list:
    for item in range(n_items_base):
        if train_matrix[user][item] == 0:
            continue
        else:
            users_first_rating_movie.append(item)
            break

print(random_users_id_list)
print(users_first_rating_movie)

# %%
for i in range(94):
    print(train_matrix[random_users_id_list[i]][users_first_rating_movie[i]], end=", ")

# %%
for i in range(len(random_users_id_list)):
    test_matrix[random_users_id_list[i]][users_first_rating_movie[i]]=train_matrix[random_users_id_list[i]][users_first_rating_movie[i]]
    train_matrix[random_users_id_list[i]][users_first_rating_movie[i]]=0

print(random_users_id_list)


# %%
for i in random_users_id_list:
    print(np.sum(test_matrix[i]), end=", ")

# %%
for i in range(94):
    print(train_matrix[random_users_id_list[i]][users_first_rating_movie[i]], end=", ")

# %%
# 1-user_similart
user_similarity = pairwise_distances(train_matrix, metric='correlation')
user_similarity=1-user_similarity
print('shape: ',user_similarity.shape)
print('type: ',type(user_similarity))
user_similarity

# %%
len(user_similarity)
correlationjaccard

for i in range(len(user_similarity)):
    for j in range(len(user_similarity)):
        if i == j:
            user_similarity[i][j]=0
        else:
            continue
            

# %%
for i in range(len(user_similarity)):
    print(user_similarity[i][i], end=" ,")

user_similarity.sort

# %%
a=user_similarity.argsort()
a[0][-11:]

# %%
similar_n = user_similarity.argsort()[:,-11:][:,::-1]

# %%
similar_n.shape

# %%
user_similarity[0][915]

# %%
user_similarity[0][681]

# %%
user_similarity[1][930]

# %%
ml=[1,2,5,8,9,123,62,1231,11]

# %%
arr

# %%
arr=np.array([[4,5,6],[3,2,1],[7,8,9]])
arr

# %%
arr.argsort()

# %%
st=arr.argsort()[:,1:][:,::-1]# doğru olan

# %%
st

# %%
def predict_user_user(train_matrix, user_similarity, n_similar = 10):
    # sort edip ters indexlerini ters çeviriyor seçilen en yakın user sayısına göre
    similar_n = user_similarity.argsort()[:,n_similar:][:,::-1]# kontrol et
    # pred matrix'i oluşturuluyor
    pred = np.zeros((n_users_base,n_items_base))
    
    for i,users in enumerate(similar_n):# pred yapılıyor
        similar_users_indexes = users
        similarity_n = user_similarity[i,similar_users_indexes]
        matrix_n = train_matrix[similar_users_indexes,:]
        rated_items = similarity_n[:,np.newaxis].T.dot(matrix_n - matrix_n.mean(axis=1)[:,np.newaxis])/ similarity_n.sum()
        pred[i,:]  = rated_items
        
    return pred

# %%
predictions = predict_user_user(train_matrix,user_similarity, 10) + train_matrix.mean(axis=1)[:, np.newaxis]# 30 olması daha iyi
print('predictions shape ',predictions.shape)

predictions

# %%
predicted_ratings = predictions[test_matrix.nonzero()]
test_truth = test_matrix[test_matrix.nonzero()]
math.sqrt(mean_squared_error(predicted_ratings,test_truth))

# %%
test_truth

# %%
predicted_ratings[0],predicted_ratings[1]

# %%
test_truth[0],test_truth[1]

# %%
predicted_ratings

# %%
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
import math

# %%
rs_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv('ml-1m/ratings.dat', sep='::', names=rs_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-1m/ratings.dat', sep='::', names=rs_cols, encoding='latin-1')

# %%
ratings_base.head()

# %%
n_users_base = ratings_base['user_id'].unique().max()
n_items_base = ratings_base['movie_id'].unique().max()

n_users_base,n_items_base

# %%
n_users_test = ratings_test['user_id'].unique().max()
n_items_test = ratings_test['movie_id'].unique().max()
n_users_test,n_items_test

# %%
train_matrix = np.zeros((n_users_base, n_items_base))
test_matrix = np.zeros((n_users_test, n_items_test))

# %%
for line in ratings_base.itertuples():
    train_matrix[line[1]-1,line[2]-1] = line[3]

test_matrix

# %%
train_matrix

# %%
n_users_base,n_items_base

# %%
random_users_id_list=[]
while True:
    rnd=np.random.randint(6040)
    if len(random_users_id_list) < 604:
        if rnd in random_users_id_list:
            continue
        else:
            random_users_id_list.append(rnd)
    else:
        break

random_users_id_list.sort()
random_users_id_list

# %%
users_first_rating_movie=[]
for user in random_users_id_list:
    for item in range(n_items_base):
        if train_matrix[user][item] == 0:
            continue
        else:
            users_first_rating_movie.append(item)
            break

print(random_users_id_list)

# %%
print(users_first_rating_movie)

# %%
for i in range(94):
    print(train_matrix[random_users_id_list[i]][users_first_rating_movie[i]], end=", ")

# %%
for i in range(len(random_users_id_list)):
    test_matrix[random_users_id_list[i]][users_first_rating_movie[i]]=train_matrix[random_users_id_list[i]][users_first_rating_movie[i]]
    train_matrix[random_users_id_list[i]][users_first_rating_movie[i]]=0

print(random_users_id_list)

# %%
for i in random_users_id_list:
    print(np.sum(test_matrix[i]), end=", ")

# %%
for i in range(94):
    print(train_matrix[random_users_id_list[i]][users_first_rating_movie[i]], end=", ")

# %%
# 1-user_similart
user_similarity = pairwise_distances(train_matrix, metric='cosine')
user_similarity=1-user_similarity
print('shape: ',user_similarity.shape)
print('type: ',type(user_similarity))
user_similarity

# %%
for i in range(len(user_similarity)):
    for j in range(len(user_similarity)):
        if i == j:
            user_similarity[i][j]=0
        else:
            continue

for i in range(len(user_similarity)):
    print(user_similarity[i][i], end=" ,")


# %%
def predict_user_user(train_matrix, user_similarity, n_similar=30):
    # sort edip ters indexlerini ters çeviriyor seçilen en yakın user sayısına göre
    similar_n = user_similarity.argsort()[:,n_similar:][:,::-1]# kontrol et
    # pred matrix'i oluşturuluyor
    pred = np.zeros((n_users_base,n_items_base))
    
    for i,users in enumerate(similar_n):# pred yapılıyor
        similar_users_indexes = users
        similarity_n = user_similarity[i,similar_users_indexes]
        matrix_n = train_matrix[similar_users_indexes,:]
        rated_items = similarity_n[:,np.newaxis].T.dot(matrix_n - matrix_n.mean(axis=1)[:,np.newaxis])/ similarity_n.sum()
        pred[i,:]  = rated_items
        
    return pred

# %%
predictions = predict_user_user(train_matrix,user_similarity, 30) + train_matrix.mean(axis=1)[:, np.newaxis]# 30 olması daha iyi
print('predictions shape ',predictions.shape)

predictions

# %%
predicted_ratings = predictions[test_matrix.nonzero()]
test_truth = test_matrix[test_matrix.nonzero()]
math.sqrt(mean_squared_error(predicted_ratings,test_truth))

# %%
test_truth

# %%
predicted_ratings

# %%
predicted_ratings[0],predicted_ratings[1]

# %%
test_truth[0],test_truth[1]


