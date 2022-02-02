import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
import math

rs_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('Dataset/ml-100k/ua.base', sep='\t', names=rs_cols, encoding='latin-1')
ratings_test = pd.read_csv('Dataset/ml-100k/ua.base', sep='\t', names=rs_cols, encoding='latin-1')
n_users_base = ratings_base['user_id'].unique().max()
n_items_base = ratings_base['movie_id'].unique().max()
n_users_test = ratings_test['user_id'].unique().max()
n_items_test = ratings_test['movie_id'].unique().max()
train_matrix = np.zeros((n_users_base, n_items_base))
test_matrix = np.zeros((n_users_test, n_items_test))

for line in ratings_base.itertuples():
    train_matrix[line[1]-1,line[2]-1] = line[3]

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
print(random_users_id_list)

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

for i in range(94):
    print(train_matrix[random_users_id_list[i]][users_first_rating_movie[i]], end=", ")

for i in range(len(random_users_id_list)):
    test_matrix[random_users_id_list[i]][users_first_rating_movie[i]]=train_matrix[random_users_id_list[i]][users_first_rating_movie[i]]
    train_matrix[random_users_id_list[i]][users_first_rating_movie[i]]=0

print(random_users_id_list)
for i in random_users_id_list:
    print(np.sum(test_matrix[i]), end=", ")

for i in range(94):
    print(train_matrix[random_users_id_list[i]][users_first_rating_movie[i]], end=", ")

user_similarity = pairwise_distances(train_matrix, metric='correlation')
user_similarity=1-user_similarity
print('shape: ',user_similarity.shape)
print('type: ',type(user_similarity))

for i in range(len(user_similarity)):
    for j in range(len(user_similarity)):
        if i == j:
            user_similarity[i][j]=0
        else:
            continue

for i in range(len(user_similarity)):
    print(user_similarity[i][i], end=" ,")

def predict_user_user(train_matrix, user_similarity, n_similar = 10):
    similar_n = user_similarity.argsort()[:,n_similar:][:,::-1]# kontrol et
    pred = np.zeros((n_users_base,n_items_base))
    
    for i,users in enumerate(similar_n):
        similar_users_indexes = users
        similarity_n = user_similarity[i,similar_users_indexes]
        matrix_n = train_matrix[similar_users_indexes,:]
        rated_items = similarity_n[:,np.newaxis].T.dot(matrix_n - matrix_n.mean(axis=1)[:,np.newaxis])/ similarity_n.sum()
        pred[i,:]  = rated_items
        
    return pred

predictions = predict_user_user(train_matrix,user_similarity, 10) + train_matrix.mean(axis=1)[:, np.newaxis]# 30 olması daha iyi
print('predictions shape ',predictions.shape)

predicted_ratings = predictions[test_matrix.nonzero()]
test_truth = test_matrix[test_matrix.nonzero()]
math.sqrt(mean_squared_error(predicted_ratings,test_truth))