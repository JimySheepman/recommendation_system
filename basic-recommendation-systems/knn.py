# %%
import numpy as np
import pandas as pd
import scipy
from scipy.stats import pearsonr
from sklearn.preprocessing import normalize
import math

# %%
rs_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

# %%
ratings = pd.read_csv('ml-100k/ua.base', sep='\t', names=rs_cols, encoding='latin-1')

# %%
ratings

# %%
user_ratings = ratings.groupby('user_id')

# %%
user_ratings = {}
for k, v in ratings.groupby('user_id'):
    user_ratings[k] = dict(zip(v['movie_id'].values, v['rating'].values))
user_ratings_test = dict(list(user_ratings.items())[:1000])

# %%
class userUserBasedCF(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.means = {}
        for user, user_ratings in self.dataset.items():
            mean = np.mean(np.array(list(user_ratings.values())))
            self.means[user] = mean

    def pearson_correlation(self, user1, user2):
        """
        user1, user2: dictionaries
        """
        common_movies = sorted(set(user1).intersection(set(user2)))
        if len(common_movies) != 0 and len(common_movies) != 1:
            user1_ratings = np.squeeze(
                normalize(np.array([user1[movie] for movie in common_movies])[np.newaxis, :]))
            user2_ratings = np.squeeze(
                normalize(np.array([user2[movie] for movie in common_movies])[np.newaxis, :]))
            corr = pearsonr(user1_ratings, user2_ratings)[0]
        else:
            corr = 0
            #print("No common movies")
        return corr

    def knn(self, user, k):
        """
        user: user_id
        k: number of KNN
        """
        neighbours = {}
        i = 0
        for user_id, user_data in self.dataset.items():
            if user_id == user:
                continue
            corr = self.pearson_correlation(self.dataset[user], user_data)
            neighbours[user_id] = corr
            i += 1
        sort = sorted(neighbours.items(), key=lambda x: x[1], reverse=True)
        knn = sort[:k]
        knn_user_ids = [user_id for user_id, user_corr in knn]
        print("KNN")
        return knn_user_ids

    def predict(self, user, movie_id, knn):
        """
        user: user_id
        movie_id: movie_id
        knn: knn_user_ids

        prediction = mean_rating_of_active_user + sum_over_knn(user_rating_for_i * pearson(user, active_user))/sum_over_knn(pearson(user, active_user))
        """
        mean_user_rating = self.means[user]
        print("Mean user rating for the user is ", mean_user_rating)
        iter_rating = 0.0
        pear_corr = 0.0
        for i, element in enumerate(knn):
            temp_corr = self.pearson_correlation(
                self.dataset[user], self.dataset[element])
            if math.isnan(temp_corr):
                continue
            if movie_id in self.dataset[element].keys():
                iter_rating += (self.dataset[element]
                                [movie_id]-self.means[element]) * temp_corr
            else:
                iter_rating += 0
            pear_corr += temp_corr

        pred = mean_user_rating + iter_rating/pear_corr
        return pred


# %%
cf = userUserBasedCF(user_ratings_test)

# %%
import warnings
warnings.filterwarnings('ignore')
knn = cf.knn(3, 100)
cf.predict(3, 45, knn)

# %%
knn

# %%
ratings_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=rs_cols, encoding='latin-1')

# %%
n_users_base = ratings_base['user_id'].unique().max()
n_items_base = ratings_base['movie_id'].unique().max()

n_users_base,n_items_base

# %%
train_matrix = np.zeros((n_users_base, n_items_base))
for line in ratings_base.itertuples():
    train_matrix[line[1]-1,line[2]-1] = line[3]

# %%
train_matrix[3,45]

# %%
user_ratings

# %%
user_ratings_test

# %%
len(user_ratings_test.keys())


