# %% [markdown]
# ## Cosinüs 

# %%
import numpy as np
from scipy import io
import warnings
from sklearn.metrics.pairwise import pairwise_distances
warnings.filterwarnings("ignore")

# work .mat file
mat = io.loadmat('matrix.mat')
matrix=mat["matrix"]
sim_list=[]
pre_list=[]
sim_list.append(matrix[0][0])
sim_list.append(matrix[13][6])
matrix[0][0]=0

def user_rating_average(mat):
    sum1=0
    average_list=[]
    for x in range(len(mat)):
        for i in mat[x]:
            sum1=sum1+i
        sum1=sum1/len(mat[x].nonzero()[0])
        average_list.append(sum1)
        sum1=0
    return average_list

user_average_list=user_rating_average(matrix)
user_average_list=np.nan_to_num(user_average_list, nan=0)

user_similarity_list = pairwise_distances(matrix, metric='cosine')
similar_n = user_similarity_list[0].argsort()

def user_rating_pred(avr_list,mat_row,user_1_rating,pre_id):
    result=0
    pay=0
    for usr in range(len(avr_list)):
        if usr == pre_id:
            continue
        else:
            result+=mat_row[usr]*(user_1_rating[usr][pre_id]-avr_list[usr])
            pay+=mat_row[usr]
    return avr_list[pre_id]+(result/pay)
# tüm oy vermeyenleri çıkart
# sim ilk hesapla
telsim=user_rating_pred(user_average_list,user_similarity_list[0],matrix,similar_n[0])
matrix[13][6]=0
similar_n1 = user_similarity_list[13].argsort()
telsim1=user_rating_pred(user_average_list,user_similarity_list[13],matrix,similar_n1[0])
pre_list.append(telsim)
pre_list.append(telsim1)
print("User-{} prediction rating:".format(similar_n[0]),telsim)
print("User-{} prediction rating:".format(similar_n1[0]),telsim1)

def calulate_error_avg(list1,list2):
    total=0
    for i in range(len(list1)):
        total=total+(list1[i]-list2[i])
    return total/len(list1)
a=calulate_error_avg(sim_list,pre_list)
print("Error:",a)

# %%
import numpy as np
from scipy import io
import warnings
from sklearn.metrics.pairwise import pairwise_distances
warnings.filterwarnings("ignore")

mat = io.loadmat('matrix.mat')
matrix=mat["matrix"]
user_similarity_list = pairwise_distances(matrix, metric='cosine')
similar_n = user_similarity_list[13].argsort()


# %%
similar_n

# %%

user_similarity_list = pairwise_distances(matrix, metric='cosine')
similar_n = user_similarity_list[0].argsort()

# %%
similar_n

# %%
similar_n

# %% [markdown]
# # Pearson

# %%
import numpy as np
from scipy import stats, io
import warnings
warnings.filterwarnings("ignore")

# work .mat file
mat = io.loadmat('matrix.mat')
matrix=mat["matrix"]
sim_list=[]
pre_list=[]
sim_list.append(matrix[0][6])
sim_list.append(matrix[13][6])
matrix[0][0]=0


def user_rating_average(mat):
    sum1=0
    average_list=[]
    for x in range(len(mat)):
        for i in mat[x]:
            sum1=sum1+i
        sum1=sum1/len(mat[x].nonzero()[0])
        average_list.append(sum1)
        sum1=0
    return average_list

user_average_list=user_rating_average(matrix)
user_average_list=np.nan_to_num(user_average_list, nan=0)

def pearson_similarity(mat):
    arr=np.zeros((20, 20))
    for i in range(len(mat)):
        for j in range(len(mat)):
            pearson=stats.pearsonr(mat[i],mat[j])
            arr[i][j]=pearson[0]
    return arr

user_similarity_list=pearson_similarity(matrix)

user_similarity_list=np.nan_to_num(user_similarity_list, nan=0)

similar_n = user_similarity_list[0].argsort()

def cal_pay(avr_list,mat_row,user_1_rating,pre_id):
    result=0
    pay=0
    for item in range(len(avr_list)):
        if item == pre_id:
            continue
        else:
            result+=mat_row[item]*(user_1_rating[item][pre_id]-avr_list[item])
            pay+=mat_row[item]
    return avr_list[pre_id]+(result/pay)

# tüm oy vermeyenleri çıkart
# sim ilk hesapla
telsim=cal_pay(user_average_list,user_similarity_list[0],matrix,similar_n[-1])
matrix[13][6]=0
similar_n1 = user_similarity_list[13].argsort()
telsim1=cal_pay(user_average_list,user_similarity_list[13],matrix,similar_n1[-1])
pre_list.append(telsim)
pre_list.append(telsim1)
print("User-{} prediction rating:".format(similar_n[-1]),telsim)
print("User-{} prediction rating:".format(similar_n1[-1]),telsim1)
def calulate_error_avg(list1,list2):
    total=0
    for i in range(len(list1)):
        total=total+(list1[i]-list2[i])
    return total/len(list1)
a=calulate_error_avg(sim_list,pre_list)
print("Error:",a)

# %%
sim_list

# %%
pre_list

# %% [markdown]
# # ml-100k dataset

# %%
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
import math

# %%
rs_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('Dataset/ml-100k/ua.base', sep='\t', names=rs_cols, encoding='latin-1')
ratings_test = pd.read_csv('Dataset/ml-100k/ua.base', sep='\t', names=rs_cols, encoding='latin-1')

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

# %%
test_matrix

# %%
train_matrix

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

# %%
random_users_id_list.sort()

# %%
print(random_users_id_list)

# %% [markdown]
# ## Finding and keeping the first items and ratings that 10% users voted for 

# %%
users_first_rating_movie=[]

# %%
for user in random_users_id_list:
    for item in range(n_items_base):
        if train_matrix[user][item] == 0:
            continue
        else:
            users_first_rating_movie.append(item)
            break

# %%
print(random_users_id_list)

# %%
print(users_first_rating_movie)

# %%
for i in range(94):
    print(train_matrix[random_users_id_list[i]][users_first_rating_movie[i]], end=", ")

# %% [markdown]
# ## create a test matrix 

# %%
for i in range(len(random_users_id_list)):
    test_matrix[random_users_id_list[i]][users_first_rating_movie[i]]=train_matrix[random_users_id_list[i]][users_first_rating_movie[i]]
    train_matrix[random_users_id_list[i]][users_first_rating_movie[i]]=0

# %%
print(random_users_id_list)

# %%
for i in random_users_id_list:
    print(np.sum(test_matrix[i]), end=", ")

# %%
for i in range(94):
    print(train_matrix[random_users_id_list[i]][users_first_rating_movie[i]], end=", ")

# %% [markdown]
# ## creates test matrix and train matrix 

# %%
user_similarity = pairwise_distances(train_matrix, metric='correlation')
user_similarity=1-user_similarity
print('shape: ',user_similarity.shape)
print('type: ',type(user_similarity))
user_similarity

# %% [markdown]
# ## make the diogones zero 

# %%
len(user_similarity)

# %%
for i in range(len(user_similarity)):
    for j in range(len(user_similarity)):
        if i == j:
            user_similarity[i][j]=0
        else:
            continue

# %%
for i in range(len(user_similarity)):
    print(user_similarity[i][i], end=" ,")

# %%
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

# %%
predictions = predict_user_user(train_matrix,user_similarity, 10) + train_matrix.mean(axis=1)[:, np.newaxis]# 30 olması daha iyi
print('predictions shape ',predictions.shape)
predictions

# %%
predicted_ratings = predictions[test_matrix.nonzero()]
test_truth = test_matrix[test_matrix.nonzero()]

# %%
math.sqrt(mean_squared_error(predicted_ratings,test_truth))

# %% [markdown]
# # ml-1m dataset

# %%
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
import math

# %%
rs_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('Dataset/ml-1m/ratings.dat', sep='::', names=rs_cols, encoding='latin-1')
ratings_test = pd.read_csv('Dataset/ml-1m/ratings.dat', sep='::', names=rs_cols, encoding='latin-1')

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

# %%
test_matrix

# %%
train_matrix

# %%
n_users_base, n_items_base

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

# %%
random_users_id_list.sort()

# %%
print(random_users_id_list)

# %%
users_first_rating_movie=[]

# %%
for user in random_users_id_list:
    for item in range(n_items_base):
        if train_matrix[user][item] == 0:
            continue
        else:
            users_first_rating_movie.append(item)
            break

# %%
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

# %%
print(random_users_id_list)

# %%
for i in random_users_id_list:
    print(np.sum(test_matrix[i]), end=", ")

# %%
for i in range(94):
    print(train_matrix[random_users_id_list[i]][users_first_rating_movie[i]], end=", ")

# %%
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

# %%
for i in range(len(user_similarity)):
    print(user_similarity[i][i], end=" ,")

# %%
def predict_user_user(train_matrix, user_similarity, n_similar=30):
    similar_n = user_similarity.argsort()[:,n_similar:][:,::-1]
    pred = np.zeros((n_users_base,n_items_base))
    
    for i,users in enumerate(similar_n):
        similar_users_indexes = users
        similarity_n = user_similarity[i,similar_users_indexes]
        matrix_n = train_matrix[similar_users_indexes,:]
        rated_items = similarity_n[:,np.newaxis].T.dot(matrix_n - matrix_n.mean(axis=1)[:,np.newaxis])/ similarity_n.sum()
        pred[i,:]  = rated_items
        
    return pred

# %%
predictions = predict_user_user(train_matrix,user_similarity, 30) + train_matrix.mean(axis=1)[:, np.newaxis]
print('predictions shape ',predictions.shape)

# %%
predicted_ratings = predictions[test_matrix.nonzero()]
test_truth = test_matrix[test_matrix.nonzero()]

# %%
math.sqrt(mean_squared_error(predicted_ratings,test_truth))


