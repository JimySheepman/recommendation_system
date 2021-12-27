from flask import Flask, render_template, request
# import essential basic libraries
import pandas as pd
import numpy as np
from math import sqrt
from scipy.sparse.linalg import svds
from scipy.optimize import fmin_cg
# import matrix_factorization_utilities.py
from models import matrix_factorization_utilities
# import machine learning libraries
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
# import deep learning libraries
import keras
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Activation, Flatten, Embedding, Reshape, concatenate
# import warnings libraries for close warnings
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)


def load_data(dataset_id):
        if dataset_id == '1':
            rs_cols = ['userId', 'movieId', 'rating', 'unix_timestamp']
            movie_cols = ['movieId', 'title', 'release_date',
                          'video_release_date', 'imdb_url']
            movies = pd.read_csv('Dataset/ml-100k/u.item', sep='|',
                                 names=movie_cols, usecols=range(5), encoding='latin-1')
            ratings_base = pd.read_csv(
                'Dataset/ml-100k/ua.base', sep='\t', names=rs_cols, encoding='latin-1')
            ratings_test = pd.read_csv(
                'Dataset/ml-100k/ua.test', sep='\t', names=rs_cols, encoding='latin-1')
            ratings_base.drop('unix_timestamp', inplace=True, axis=1)
            ratings_test.drop('unix_timestamp', inplace=True, axis=1)
            movies.drop('release_date', inplace=True, axis=1)
            movies.drop('video_release_date', inplace=True, axis=1)
            movies.drop('imdb_url', inplace=True, axis=1)

            links = pd.read_csv('Dataset/movies.dat',
                                sep='\t', encoding='latin-1')
            for i in links.columns:
                if i == 'id'or i == 'title'or i == 'year' or i == 'rtPictureURL':
                    continue
                else:
                    links.drop(i, inplace=True, axis=1)

            links.columns = ['movieId', 'title_old', 'year', 'rtPictureURL']
            links['title'] = links.apply(lambda x: '%s (%s)' % (
                x['title_old'], x['year']), axis=1)
            links.drop('title_old', inplace=True, axis=1)
            links.drop('year', inplace=True, axis=1)
            links = links[['movieId', 'title', 'rtPictureURL']]
            movie_list = pd.merge(movies, links, on='movieId')
            movie_list.drop('title_y', inplace=True, axis=1)
            movie_list.columns = ['movieId', 'title', 'url']
            ratings_base.userId = ratings_base.userId.astype(
                'category').cat.codes.values
            ratings_base.movieId = ratings_base.movieId.astype(
                'category').cat.codes.values
            ratings_test.userId = ratings_test.userId.astype(
                'category').cat.codes.values
            ratings_test.movieId = ratings_test.movieId.astype(
                'category').cat.codes.values
            train = ratings_base
            test = ratings_test
            ratings = pd.concat([ratings_base, ratings_test])

            return train, test, ratings, movie_list

        elif dataset_id == '2':
            # Reading the ratings data
            ratings = pd.read_csv('Dataset/ratings.csv')
            # Just taking the required columns
            ratings = ratings[['userId', 'movieId', 'rating']]
            # reading the movies dataset
            movies = pd.read_csv('Dataset/movies.csv')

            links = pd.read_csv('Dataset/movies.dat',
                                sep='\t', encoding='latin-1')
            for i in links.columns:
                if i == 'id'or i == 'title'or i == 'year' or i == 'rtPictureURL':
                    continue
                else:
                    links.drop(i, inplace=True, axis=1)

            links.columns = ['movieId', 'title_old', 'year', 'rtPictureURL']
            links['title'] = links.apply(lambda x: '%s (%s)' % (
                x['title_old'], x['year']), axis=1)
            links.drop('title_old', inplace=True, axis=1)
            links.drop('year', inplace=True, axis=1)
            links = links[['movieId', 'title', 'rtPictureURL']]
            movie_list = pd.merge(movies, links, on='movieId')
            movie_list.drop('title_y', inplace=True, axis=1)
            movie_list.columns = ['movieId', 'title', 'genres', 'url']

            ratings.userId = ratings.userId.astype('category').cat.codes.values
            ratings.movieId = ratings.movieId.astype(
                'category').cat.codes.values

            # take 80% as the training set and 20% as the test set
            train, test = train_test_split(ratings, test_size=0.2)
            return train, test, ratings, movie_list


def matrix_factorisation_predictions(usr,dataset_id):
    train, test, ratings,movie_list =load_data(dataset_id)
    n_users = train.userId.unique().shape[0]
    n_items = train.movieId.max()+1
    train_data_matrix = np.zeros((n_users, n_items))
    #for every line in the data
    for line in train.itertuples():
        train_data_matrix[line[1]-1,line[2]-1] = line[3]
            
    #Create two user-item matrices, one for training and another for testing
    test_data_matrix = np.zeros((n_users, n_items))
    #for every line in the data
    for line in test.itertuples():
        test_data_matrix[line[1]-1,line[2]-1] = line[3]
            
    rmse_list = []
    mae_list = []
    result_list=[]
    a=[1,2,5,10,20]
    for i in a:
        #apply svd to the test data
        u,s,vt = svds(train_data_matrix,k=i)
        #get diagonal matrix
        s_diag_matrix=np.diag(s)
        #predict x with dot product of u s_diag and vt
        X_pred = np.dot(np.dot(u,s_diag_matrix),vt)
        #calculate rmse score of matrix factorisation predictions
        mse =mean_squared_error(test_data_matrix, X_pred)
        rmse_score = sqrt(mse)
        rmse_list.append(round(rmse_score,4))
        mae_score = mean_absolute_error(test_data_matrix,X_pred)
        mae_list.append(round(mae_score,4))
   
    min_value_rmse=min(rmse_list)
    min_index_rmse=rmse_list.index(min_value_rmse)
    min_value_mae=min(mae_list)
    min_index_mae=mae_list.index(min_value_mae)
    y_true = test.rating
    errors=min_value_mae
    mape = 100 * (errors / y_true)
    accuracy = 100 - np.mean(mape)
    accuracy=round(accuracy, 2)
    result_list.append("Matrix Factorisation with " + str(a[min_index_rmse]) +" latent features has a Accuracy of "+str(accuracy)+ "%.")
    result_list.append("Matrix Factorisation with " + str(a[min_index_rmse]) +" latent features has a RMSE of " + str(min_value_rmse))
    result_list.append("Matrix Factorisation with " + str(a[min_index_mae]) +" latent features has a MAE of " + str(min_value_mae))
    mf_pred = pd.DataFrame(X_pred)
    df_names = pd.merge(ratings,movie_list,on='movieId')
    user_id = usr 
    users_movies = df_names.loc[df_names["userId"]==user_id]
    result_list.append("User ID : " + str(user_id) + " has already rated " + str(len(users_movies)) + " movies")
    user_index = user_id-1
    sorted_user_predictions = pd.DataFrame(mf_pred.iloc[user_index].sort_values(ascending=False))
    sorted_user_predictions.columns=['ratings']
    sorted_user_predictions['movieId']=sorted_user_predictions.index
    results_df=pd.merge(sorted_user_predictions,movie_list, on = 'movieId')[:10]

    return  results_df,result_list

def neural_network_predictions(usr,dataset_id):
    train, test, ratings,movie_list =load_data(dataset_id)
    if dataset_id =='1':
        result_list=[]
        model_2 = load_model('models/100k_nn.h5', compile = True)

        y_hat = model_2.predict([test.userId, test.movieId])
        y_true = test.rating
        result_list.append("(MAE)Mean Absolute Error: {}".format(round(mean_absolute_error(y_true, y_hat),4)))
        result_list.append("(RMSE)Root Mean Square Error: {}".format(round(np.sqrt(mean_squared_error(y_true,y_hat)) ,4)))
        errors=mean_absolute_error(y_true,y_hat)
        mape = 100 * (errors / y_true)
        accuracy = 100 - np.mean(mape)
        result_list.append(' Accuracy: {}'.format(round(accuracy, 2), '%.'))

        mlp_user_embedding_weights = (next(iter(filter(lambda x: x.name == 'User-Embedding', model_2.layers))).get_weights())
        # get the latent embedding for your desired user
        user_latent_matrix = mlp_user_embedding_weights[0]
        desired_user_id = usr
        one_user_vector = user_latent_matrix[desired_user_id,:]
        one_user_vector = np.reshape(one_user_vector, (-1, 10))
        result_list.append('Performing kmeans to find the nearest users...')
        # get similar users
        kmeans = KMeans(n_clusters=20, random_state=0, verbose=0).fit(user_latent_matrix)
        desired_user_label = kmeans.predict(one_user_vector)
        user_label = kmeans.labels_
        neighbors = []
        for user_id, user_label in enumerate(user_label):
            if user_label == desired_user_label:
                neighbors.append(user_id)
        result_list.append('Found {0} neighbor users.'.format(len(neighbors)))

        movies = []
        for user_id in neighbors:
            movies += list(ratings[ratings['userId'] == int(user_id)]['movieId'])
        movies = list(set(movies))
        result_list.append('Found {0} neighbor movies from these users.'.format(len(movies)))

        users = np.full(len(movies), desired_user_id, dtype='int32')
        items = np.array(movies, dtype='int32')
        result_list.append('Ranking most likely tracks using the NeuMF model...')
        # and predict movies for my user
        results = model_2.predict([users,items],batch_size=10, verbose=0) 
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

        return results_df , result_list

    elif dataset_id=='2':
        result_list=[]
        model_2 = load_model('models/1m_nn.h5', compile = True)

        y_hat = model_2.predict([test.userId, test.movieId])
        y_true = test.rating
        result_list.append("(MAE)Mean Absolute Error: {}".format(round(mean_absolute_error(y_true, y_hat),4)))
        result_list.append("(RMSE)Root Mean Square Error: {}".format(round(np.sqrt(mean_squared_error(y_true,y_hat)) ,4)))
        errors=mean_absolute_error(y_true,y_hat)
        mape = 100 * (errors / y_true)
        accuracy = 100 - np.mean(mape)
        result_list.append(' Accuracy: {}'.format(round(accuracy, 2), '%.'))

        mlp_user_embedding_weights = (next(iter(filter(lambda x: x.name == 'User-Embedding', model_2.layers))).get_weights())
        # get the latent embedding for your desired user
        user_latent_matrix = mlp_user_embedding_weights[0]
        desired_user_id = usr
        one_user_vector = user_latent_matrix[desired_user_id,:]
        one_user_vector = np.reshape(one_user_vector, (-1, 10))
        result_list.append('Performing kmeans to find the nearest users...')
        # get similar users
        kmeans = KMeans(n_clusters=20, random_state=0, verbose=0).fit(user_latent_matrix)
        desired_user_label = kmeans.predict(one_user_vector)
        user_label = kmeans.labels_
        neighbors = []
        for user_id, user_label in enumerate(user_label):
            if user_label == desired_user_label:
                neighbors.append(user_id)
        result_list.append('Found {0} neighbor users.'.format(len(neighbors)))

        movies = []
        for user_id in neighbors:
            movies += list(ratings[ratings['userId'] == int(user_id)]['movieId'])
        movies = list(set(movies))
        result_list.append('Found {0} neighbor movies from these users.'.format(len(movies)))

        users = np.full(len(movies), desired_user_id, dtype='int32')
        items = np.array(movies, dtype='int32')
        result_list.append('Ranking most likely tracks using the NeuMF model...')
        # and predict movies for my user
        results = model_2.predict([users,items],batch_size=10, verbose=0) 
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

        return results_df , result_list

def neural_collaborative_filtering_predictions(usr,dataset_id):
    train, test, ratings,movie_list =load_data(dataset_id)
    if dataset_id=='1':
        result_list=[]
        model_3 = load_model('models/100k_ncf.h5', compile = True)
        y_true = test.rating
        result_list.append('MAE for testing: {}'.format(round(mean_absolute_error(y_true, model_3.predict([test.userId, test.movieId])),4)))
        rmse = np.sqrt(mean_squared_error(y_true,model_3.predict([test.userId, test.movieId]))) 
        result_list.append("Root Mean Square Error: {} ".format(round(rmse,4)))
        errors=mean_absolute_error(y_true, model_3.predict([test.userId, test.movieId]))
        mape = 100 * (errors / y_true)
        accuracy = 100 - np.mean(mape)
        result_list.append('Accuracy: {} '.format( round(accuracy, 2), '%.'))

        mlp_user_embedding_weights = (next(iter(filter(lambda x: x.name == 'User-Embedding-MLP', model_3.layers))).get_weights())
        # get the latent embedding for your desired user
        user_latent_matrix = mlp_user_embedding_weights[0]
        desired_user_id = usr
        one_user_vector = user_latent_matrix[desired_user_id,:]
        one_user_vector = np.reshape(one_user_vector, (-1,10))
        result_list.append('For user id: {0} '.format(desired_user_id))
        # get similar users
        kmeans = KMeans(n_clusters=20, random_state=0, verbose=0).fit(user_latent_matrix)
        desired_user_label = kmeans.predict(one_user_vector)
        user_label = kmeans.labels_
        neighbors = []
        for user_id, user_label in enumerate(user_label):
            if user_label == desired_user_label:
                neighbors.append(user_id)
        result_list.append('Found {0} neighbor users.'.format(len(neighbors)))

        movies = []
        for user_id in neighbors:
            movies += list(ratings[ratings['userId'] == int(user_id)]['movieId'])
        movies = list(set(movies))
        result_list.append('Found {0} neighbor movies from these users.'.format(len(movies)))

        users = np.full(len(movies), desired_user_id, dtype='int32')
        items = np.array(movies, dtype='int32')

        # and predict movies for my user
        results = model_3.predict([users,items],batch_size=10, verbose=0) 
        results = results.tolist()
        
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
        return results_df , result_list

    elif dataset_id=='2':
        result_list=[]
        model_3 = load_model('models/1m_ncf.h5', compile = True)
        y_true = test.rating
        result_list.append('MAE for testing: {}'.format(round(mean_absolute_error(y_true, model_3.predict([test.userId, test.movieId])),4)))
        rmse = np.sqrt(mean_squared_error(y_true,model_3.predict([test.userId, test.movieId]))) 
        result_list.append("Root Mean Square Error: {} ".format(round(rmse,4)))
        errors=mean_absolute_error(y_true, model_3.predict([test.userId, test.movieId]))
        mape = 100 * (errors / y_true)
        accuracy = 100 - np.mean(mape)
        result_list.append('Accuracy: {} '.format( round(accuracy, 2), '%.'))

        mlp_user_embedding_weights = (next(iter(filter(lambda x: x.name == 'User-Embedding-MLP', model_3.layers))).get_weights())
        # get the latent embedding for your desired user
        user_latent_matrix = mlp_user_embedding_weights[0]
        desired_user_id = usr
        one_user_vector = user_latent_matrix[desired_user_id,:]
        one_user_vector = np.reshape(one_user_vector, (-1,50))
        result_list.append('For user id: {0} '.format(desired_user_id))
        # get similar users
        kmeans = KMeans(n_clusters=20, random_state=0, verbose=0).fit(user_latent_matrix)
        desired_user_label = kmeans.predict(one_user_vector)
        user_label = kmeans.labels_
        neighbors = []
        for user_id, user_label in enumerate(user_label):
            if user_label == desired_user_label:
                neighbors.append(user_id)
        result_list.append('Found {0} neighbor users.'.format(len(neighbors)))

        movies = []
        for user_id in neighbors:
            movies += list(ratings[ratings['userId'] == int(user_id)]['movieId'])
        movies = list(set(movies))
        result_list.append('Found {0} neighbor movies from these users.'.format(len(movies)))

        users = np.full(len(movies), desired_user_id, dtype='int32')
        items = np.array(movies, dtype='int32')

        # and predict movies for my user
        results = model_3.predict([users,items],batch_size=10, verbose=0) 
        results = results.tolist()

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
        return results_df , result_list



@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user = request.form["nm"]
        model_id = request.form["select"]
        dataset_id = request.form["select_dataset"]
        if model_id == "1":
            mf_pred, result = matrix_factorisation_predictions(int(user),dataset_id)
            top10 = mf_pred.drop('url', axis=1).values.tolist()
            url_list = mf_pred.url.tolist()
            title_list = mf_pred.title.tolist()
            headers = ['prediction Ratings', 'Movie Id', 'Title',]

        elif model_id == "2":
            nn_pred, result = neural_network_predictions(int(user),dataset_id)
            top10 = nn_pred.drop('url', axis=1).values.tolist()
            url_list = nn_pred.url.tolist()
            title_list = nn_pred.title.tolist()
            print(nn_pred)
            headers = ['prediction Ratings', 'Movie Id', 'Title', 'Genres']

        elif model_id == '3':
            ncf_pred, result = neural_collaborative_filtering_predictions(
                int(user),dataset_id)
            top10 = ncf_pred.drop('url', axis=1).values.tolist()
            url_list = ncf_pred.url.tolist()
            title_list = ncf_pred.title.tolist()
            headers = ['prediction Ratings', 'Movie Id', 'Title', 'Genres']

        else:
            return render_template("index.html")

        return render_template("index.html", user=user, headers=headers, top10=top10, url_list=url_list, result=result, title_list=title_list)
    else:
        return render_template("index.html")


if __name__ == "__manin__":
    app.run(debug=True)
