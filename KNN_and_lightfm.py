# %% [markdown]
# # KNN

# %%
import numpy as np
import pandas as pd
import scipy
from scipy.stats import pearsonr
from sklearn.preprocessing import normalize
import math

# %%
rs_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('Dataset/ml-100k/ua.base', sep='\t', names=rs_cols, encoding='latin-1')
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
            user1_ratings = np.squeeze(normalize(np.array([user1[movie] for movie in common_movies])[np.newaxis, :]))
            user2_ratings = np.squeeze(normalize(np.array([user2[movie] for movie in common_movies])[np.newaxis, :]))
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
        neighbours={}
        i = 0
        for user_id, user_data in self.dataset.items():
            if user_id == user:
                continue
            corr = self.pearson_correlation(self.dataset[user], user_data)
            neighbours[user_id] = corr
            i+=1
        sort = sorted(neighbours.items(), key=lambda x: x[1], reverse = True)
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
            temp_corr = self.pearson_correlation(self.dataset[user], self.dataset[element])
            if math.isnan(temp_corr):
                continue
            if movie_id in self.dataset[element].keys():
                iter_rating += (self.dataset[element][movie_id]-self.means[element]) * temp_corr
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
print(knn)

# %%
ratings_base = pd.read_csv('Dataset/ml-100k/ua.base', sep='\t', names=rs_cols, encoding='latin-1')

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

# %%
import warnings
warnings.simplefilter("ignore")

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import spatial #To cumopute distance between each pair of the two collections of inputs.
import operator #to perform comparisons between e.g it(a,b) is equal to a<b
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import math

# %% [markdown]
# # We add the tables and look at the data we have and combine the tables according to their relations. 

# %% [markdown]
# ## Let's add u.data and look at the top 10 

# %%
columns = ['userid', 'movieid', 'rating', 'timestamp']
ratings = pd.read_csv('Dataset/ml-100k/u.data', sep='\t',names= columns )
ratings.head(10)

# %%
ratings.shape

# %% [markdown]
# ## Let's add u.item and look at the top 10 

# %%
# importing the u.item dataset
movie_titles = ['movieid', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'action', 'adventure','animation','childrens', 'comedy','crime','documentary','drama','fantasy','film-nior','horror','musical','mystery','romance','sci-fi','thriler','war','westen']
titles = pd.read_csv('Dataset/ml-100k/u.item', sep='|',encoding='latin-1', names= movie_titles)
titles.head(10)

# %%
titles.shape

# %% [markdown]
# ## Let's add u.genre and look at the top 10

# %%
genType = ['type','genreid']
genre = pd.read_csv('Dataset/ml-100k/u.genre',sep='|',names=genType)
genre.head(10)

# %%
genre.shape

# %% [markdown]
# ## Let's add u.user and look at the top 10

# %%
user_info = ['userid', 'age', 'gender','occupations','zip code']
users = pd.read_csv('Dataset/ml-100k/u.user',sep='|',encoding='latin-1', names= user_info)
users.head(10)

# %%
users.shape

# %% [markdown]
# ## We combine u_data with u_user and sort 

# %%
ratings = pd.merge(ratings, users, on= 'userid')
ratings.head()

# %%
ratings.shape

# %% [markdown]
# ##  We merge and sort u_data with u_movies 

# %%
titles = pd.merge(titles,ratings, on='movieid')
titles.head()

# %%
titles.shape

# %%
# highest genre-based ratings 
titles.groupby('movie title')['rating'].mean().sort_values(ascending=False).head(10)

# %%
# counts of rates 
titles.groupby('movie title')['rating'].count().sort_values(ascending=False).head(10)

# %%
# find the mean value 
meanrating= pd.DataFrame(titles.groupby('movie title')['rating'].mean())
meanrating.round()

# %%
# add count to table 
meanrating['num of ratings'] = pd.DataFrame(titles.groupby('movie title')['rating'].count())
meanrating.round()

# %%
# mean rating with genre 
def get(x):
    moviegen = titles.iloc[:,x]==1
    moviegen = titles.iloc[moviegen]
    return moviegen

# %%
#importing u.genre dataset 
genType = ['type','genreid']
genre = pd.read_csv('Dataset/ml-100k/u.genre',sep='|',names=genType)
genre.drop(labels = 'genreid', axis = 1)
gens = list(genre['type'].values)
print(gens)

# %%
plt.figure(figsize=(10,4))
meanrating['rating'].hist(bins=70)

# %%
columns = ['userid', 'movieid', 'rating', 'timestamp']
df = pd.read_csv('Dataset/ml-100k/u.data', sep='\t',names= columns,index_col=0 )
newdf = df.drop(columns=['timestamp'])
newdf.head()

# %%
#Creating a new dataframe with the movie title and the rating from each user.
moviemat = titles.pivot_table(index='userid', columns ='movieid', values='rating')
moviemat.head(10)

# %%
moviemat.shape

# %%
#Replacing the missing values with the mean of the movie
newMatrix = moviemat.fillna(moviemat.mean()).round(2)

# %%
newMatrix.head(10)

# %%
features = ['userid','movieid','rating','timestamp']
traindataset = pd.read_csv('Dataset/ml-100k/u1.base', sep ='\t', names=features)
print(traindataset.head())

# %%
traindataset.isnull().sum()

# %%
traindataset.sum()

# %%
traindataset.shape

# %%
features = ['userid','movieid','rating','timestamp']
testdataset = pd.read_csv('Dataset/ml-100k/u1.test', sep ='\t', names=features)
print(testdataset.head())

# %%
testdataset.isnull().sum()

# %%
testdataset.sum()

# %%
testdataset.shape

# %%
#spliting datasets
X = traindataset.iloc[:,0:2]
y = traindataset.iloc[:,2]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size = 0.2)

# %%
#Feature Scaling 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# %%
math.sqrt(len(y_test))

# %%
#Defining the model
classifier = KNeighborsClassifier(n_neighbors=125, p=2, metric='cosine')

# %%
#fiting the model
classifier.fit(X_train, y_train)

# %%
#predicting the test set results
y_pred = classifier.predict(X_test)

# %%
y_pred

# %%
#evaluation model
cm = confusion_matrix(y_test, y_pred)
print(cm)


# %%
#f1 score
print(f1_score(y_test, y_pred,average='macro'))

# %%
#accuracy score
print(accuracy_score(y_test, y_pred))

# %%
#mean_squared_error
mean_squared_error(y_pred, y_test)

# %% [markdown]
# # U2.BASE DATASET AND U2.TEST DATASET

# %%
features = ['userid','movieid','rating','timestamp']
traindataset2 = pd.read_csv('Dataset/ml-100k/u2.base', sep ='\t', names=features)
print(traindataset2.head())

# %%
traindataset2.isnull().sum()

# %%
traindataset2.sum()

# %%
traindataset2.shape

# %%
features = ['userid','movieid','rating','timestamp']
testdataset2 = pd.read_csv('Dataset/ml-100k/u2.test', sep ='\t', names=features)
print(testdataset2.head())

# %%
testdataset2.isnull().sum()

# %%
testdataset2.sum()

# %%
testdataset2.shape

# %%
#spliting datasets
X2 = traindataset2.iloc[:,0:2]
y2 = traindataset2.iloc[:,2]
X_train, X_test, y_train, y_test = train_test_split(X2, y2, random_state=0, test_size = 0.2)

# %%
#Feature Scaling 
sc_X2 = StandardScaler()
X_train = sc_X2.fit_transform(X_train)
X_test = sc_X2.transform(X_test)

# %%
math.sqrt(len(y_test))

# %%
#Defining the model
classifier = KNeighborsClassifier(n_neighbors=125, p=2, metric='cosine')

# %%
#fiting the model
classifier.fit(X_train, y_train)

# %%
#predicting the test set results
y_pred = classifier.predict(X_test)

# %%
y_pred

# %%
#evaluation model
cm2 = confusion_matrix(y_test, y_pred)
print(cm2)

# %%
#f1 score
print(f1_score(y_test, y_pred,average='macro'))

# %%
#accuracy score
print(accuracy_score(y_test, y_pred))

# %%
#mean_squared_error

mean_squared_error(y_pred, y_test)

# %% [markdown]
# # U3.BASE DATASET AND U3.TEST DATASET

# %%
a=np.array([[5,1231,53,123],[1234,123,2,4],[123,455,12,86]])

# %%
a

# %%
a.argsort()

# %%
a.argsort()[:,2:]# stunları alıyor soldan sağ doğru gidyor

# %%
a.argsort()[:,2:][:,::-1] # doğru çalışıyor indexleri doğru arlıyor

# %%
features = ['userid','movieid','rating','timestamp']
traindataset3 = pd.read_csv('Dataset/ml-100k/u3.base', sep ='\t', names=features)
print(traindataset3.head())

# %%
#checkng for null values
traindataset3.isnull().sum()

# %%
traindataset3.sum()

# %%
traindataset3.shape

# %%
features = ['userid','movieid','rating','timestamp']
testdataset3 = pd.read_csv('Dataset/ml-100k/u3.test', sep ='\t', names=features)
print(testdataset3.head())

# %%
testdataset3.isnull().sum()

# %%
testdataset3.sum()

# %%
testdataset3.shape

# %%
#spliting datasets
X3 = traindataset3.iloc[:,0:2]
y3 = traindataset3.iloc[:,2]
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, random_state=0, test_size = 0.2)

# %%
#Feature Scaling 
sc_X3 = StandardScaler()
X3_train = sc_X3.fit_transform(X3_train)
X3_test = sc_X3.transform(X3_test)

# %%
math.sqrt(len(y_test))

# %%
#Defining the model
classifier = KNeighborsClassifier(n_neighbors=125, p=2, metric='cosine')

# %%
#fiting the model
classifier.fit(X3_train, y3_train)

# %%
#predicting the test set results
y_pred = classifier.predict(X3_test)

# %%
y_pred

# %%
#evaluation model
cm3 = confusion_matrix(y_test, y_pred)
print(cm3)

# %%
#f1 score
print(f1_score(y_test, y_pred,average='macro'))

# %%
#accuracy score
print(accuracy_score(y_test, y_pred))

# %%
mean_squared_error(y_pred, y_test)

# %% [markdown]
# # U4.BASE DATASET AND U4.TEST DATASET

# %%
features = ['userid','movieid','rating','timestamp']
traindataset4 = pd.read_csv('Dataset/ml-100k/u4.base', sep ='\t', names=features)
print(traindataset4.head())

# %%
traindataset4.isnull().sum()

# %%
traindataset4.sum()

# %%
traindataset4.shape

# %%
features = ['userid','movieid','rating','timestamp']
testdataset4 = pd.read_csv('Dataset/ml-100k/u4.test', sep ='\t', names=features)
print(testdataset4.head())

# %%
testdataset4.isnull().sum()

# %%
testdataset4.sum()

# %%
testdataset4.shape

# %%
#spliting datasets
X4 = traindataset4.iloc[:,0:2]
y4 = traindataset4.iloc[:,2]
X_train, X_test, y_train, y_test = train_test_split(X4, y4, random_state=0, test_size = 0.2)

# %%
#Feature Scaling 
sc_X4 = StandardScaler()
X_train = sc_X4.fit_transform(X_train)
X_test = sc_X4.transform(X_test)

# %%
math.sqrt(len(y_test))

# %%
#Defining the model
classifier = KNeighborsClassifier(n_neighbors=125, p=2, metric='cosine')

# %%
#fiting the model
classifier.fit(X_train, y_train)

# %%
#predicting the test set results
y_pred = classifier.predict(X_test)

# %%
y_pred

# %%
#evaluation model
cm4 = confusion_matrix(y_test, y_pred)
print(cm4)

# %%
#f1 score
print(f1_score(y_test, y_pred,average='macro'))

# %%
#accuracy score
print(accuracy_score(y_test, y_pred))

# %%
#mean_squared_error
mean_squared_error(y_pred, y_test)

# %% [markdown]
# # U5.BASE DATASET AND U5.TEST DATASET

# %%
features = ['userid','movieid','rating','timestamp']
traindataset5 = pd.read_csv('Dataset/ml-100k/u5.base', sep ='\t', names=features)
print(traindataset5.head())

# %%
traindataset5.isnull().sum()

# %%
traindataset5.sum()

# %%
traindataset5.shape

# %%
features = ['userid','movieid','rating','timestamp']
testdataset5 = pd.read_csv('Dataset/ml-100k/u5.test', sep ='\t', names=features)
print(testdataset5.head())

# %%
testdataset5.isnull().sum()

# %%
testdataset5.sum()

# %%
testdataset5.shape

# %%
X5 = traindataset5.iloc[:,0:2]
y5 = traindataset5.iloc[:,2]
X_train, X_test, y_train, y_test = train_test_split(X5, y5, random_state=0, test_size = 0.2)

# %%
sc_X5 = StandardScaler()
X_train = sc_X5.fit_transform(X_train)
X_test = sc_X5.transform(X_test)

# %%
math.sqrt(len(y_test))

# %%
#Defining the model
classifier = KNeighborsClassifier(n_neighbors=10, p=2, metric='cosine')

# %%
#fiting the model
classifier.fit(X_train, y_train)

# %%
#predicting the test set results
y_pred = classifier.predict(X_test)

# %%
y_pred

# %%
#evaluation model
cm5 = confusion_matrix(y_test, y_pred)
print(cm5)

# %%
#f1 score
print(f1_score(y_test, y_pred,average='macro'))

# %%
#accuracy score
print(accuracy_score(y_test, y_pred))

# %%
#mean_squared_error

mean_squared_error(y_pred, y_test)

# %% [markdown]
# # MERGING U.USER DATASET TO U1.BASE DATASET

# %%
features = ['userid','movieid','rating','timestamp']
Utraindataset = pd.read_csv('Dataset/ml-100k/u1.base', sep ='\t', names=features)
print(Utraindataset.head())

# %%
features = ['userid','age','gender','occupation','zipcode']
uUser = pd.read_csv('Dataset/Dataset/ml-100k/u.user', sep ='|', names=features)
print(uUser.head())

# %%
features = ['userid','movieid','rating','timestamp']
Utestdataset = pd.read_csv('Dataset/ml-100k/u1.test', sep ='\t', names=features)
print(Utestdataset.head())

# %%
Utraindataset = pd.merge(Utraindataset,uUser, on='userid')

# %%
print(Utraindataset.head())

# %%
Utraindataset= Utraindataset[['userid','movieid','age','gender','occupation','zipcode','timestamp','rating']]

# %%
print(Utraindataset.head())

# %%
Utraindataset = pd.get_dummies(Utraindataset, columns=['occupation', 'gender'])

# %%
print(Utraindataset.head())

# %%
Utraindataset.isnull().sum()

# %%
Utraindataset = Utraindataset.drop(['gender_F', 'occupation_none'], axis =1)
Utraindataset = Utraindataset.drop(['zipcode'], axis =1)

# %%
print(Utraindataset.head(5))

# %%
Utraindataset= Utraindataset[['userid','movieid','age','gender_M','occupation_librarian','occupation_lawyer','occupation_homemaker','occupation_healthcare','occupation_executive','occupation_entertainment','occupation_writer','occupation_technician','occupation_student','occupation_scientist','occupation_salesman','occupation_retired','occupation_programmer','occupation_other','occupation_marketing','occupation_engineer','occupation_educator','occupation_doctor','occupation_administrator','occupation_artist','timestamp','rating']]

# %%
Utraindataset.sum()

# %%
Utraindataset.shape

# %%
#spliting datasets
XU = Utraindataset.iloc[:,0:24]
yU = Utraindataset.iloc[:,25]
X_train, X_test, y_train, y_test = train_test_split(XU, yU, random_state=0, test_size = 0.2)

# %%
#Feature Scaling 
sc_XU = StandardScaler()
X_train = sc_XU.fit_transform(X_train)
X_test = sc_XU.transform(X_test)

# %%
math.sqrt(len(y_test))

# %%
#Defining the model
classifier = KNeighborsClassifier(n_neighbors=10, p=2, metric='cosine')

# %%
#fiting the model
classifier.fit(X_train, y_train)

# %%
#predicting the test set results
y_pred = classifier.predict(X_test)

# %%
y_pred

# %%
#evaluation model
cmU = confusion_matrix(y_test, y_pred)
print(cmU)

# %%
#accuracy score
print(accuracy_score(y_test, y_pred))

# %%
#mean_squared_error
mean_squared_error(y_pred, y_test)

# %% [markdown]
# # MERGING U.USER DATASET TO U2.BASE DATASET AND USING U2.TEST

# %%
#mporting u2.base dataset
features = ['userid','movieid','rating','timestamp']
Utraindataset2 = pd.read_csv('Dataset/ml-100k/u2.base', sep ='\t', names=features)
print(Utraindataset2.head())

# %%
#mporting u2.base dataset
features = ['userid','age','gender','occupation','zipcode']
uUser2 = pd.read_csv('Dataset/ml-100k/u.user', sep ='|', names=features)
print(uUser2.head())

# %%
#merging the u2.base dataset with the u.user dataset
Utraindataset2 = pd.merge(Utraindataset2,uUser, on='userid')

# %%
print(Utraindataset2.head())

# %%
Utraindataset2 = pd.get_dummies(Utraindataset2, columns=['occupation', 'gender'])

# %%
print(Utraindataset2.head())

# %%
Utraindataset2 = Utraindataset2.drop(['gender_F', 'occupation_none'], axis =1)
Utraindataset2 = Utraindataset2.drop(['zipcode'], axis =1)

# %%
print(Utraindataset2.head(5))

# %%
Utraindataset2= Utraindataset2[['userid','movieid','age','gender_M','occupation_librarian','occupation_lawyer','occupation_homemaker','occupation_healthcare','occupation_executive','occupation_entertainment','occupation_writer','occupation_technician','occupation_student','occupation_scientist','occupation_salesman','occupation_retired','occupation_programmer','occupation_other','occupation_marketing','occupation_engineer','occupation_educator','occupation_doctor','occupation_administrator','occupation_artist','timestamp','rating']]

# %%
Utraindataset2.head(5)

# %%
Utraindataset2.shape

# %%
#spliting datasets
XU2 = Utraindataset2.iloc[:,0:24]
yU2 = Utraindataset2.iloc[:,25]
X_train, X_test, y_train, y_test = train_test_split(XU2, yU2, random_state=0, test_size = 0.2)

# %%
#Feature Scaling 
sc_XU2 = StandardScaler()
X_trainu2 = sc_XU2.fit_transform(X_train)
X_testu2 = sc_XU2.transform(X_test)

# %%
math.sqrt(len(y_test))

# %%
#Defining the model
classifier = KNeighborsClassifier(n_neighbors=10, p=2, metric='cosine')

# %%
#fiting the model
classifier.fit(X_train, y_train)

# %%
#predicting the test set results
y_pred = classifier.predict(X_test)

# %%
y_pred

# %%
#evaluation model
cmU2 = confusion_matrix(y_test, y_pred)
print(cmU2)

# %%
#accuracy score
print(accuracy_score(y_test, y_pred))

# %%
#mean_squared_error
mean_squared_error(y_pred, y_test)

# %% [markdown]
# # MERGING U.USER DATASET TO U3.BASE DATASET AND USING U3.TEST

# %%
#mporting u3.base dataset
features = ['userid','movieid','rating','timestamp']
Utraindataset3 = pd.read_csv('Dataset/ml-100k/u3.base', sep ='\t', names=features)
print(Utraindataset3.head())

# %%
#mporting u3.base dataset 
features = ['userid','age','gender','occupation','zipcode']
uUser3 = pd.read_csv('Dataset/ml-100k/u.user', sep ='|', names=features)
print(uUser3.head())

# %%
#merging the u3.base dataset with the u.user dataset
Utraindataset3 = pd.merge(Utraindataset3,uUser, on='userid')

# %%
Utraindataset3 = pd.get_dummies(Utraindataset3, columns=['occupation', 'gender'])

# %%
Utraindataset3.head(5)

# %%
Utraindataset3 = Utraindataset3.drop(['gender_F', 'occupation_none'], axis =1)
Utraindataset3 = Utraindataset3.drop(['zipcode'], axis =1)

# %%
print(Utraindataset3.head(5))

# %%
Utraindataset3= Utraindataset3[['userid','movieid','age','gender_M','occupation_librarian','occupation_lawyer','occupation_homemaker','occupation_healthcare','occupation_executive','occupation_entertainment','occupation_writer','occupation_technician','occupation_student','occupation_scientist','occupation_salesman','occupation_retired','occupation_programmer','occupation_other','occupation_marketing','occupation_engineer','occupation_educator','occupation_doctor','occupation_administrator','occupation_artist','timestamp','rating']]

# %%
Utraindataset3.shape

# %%
#spliting datasets
XU3 = Utraindataset3.iloc[:,0:24]
yU3 = Utraindataset3.iloc[:,25]
X_train, X_test, y_train, y_test = train_test_split(XU3, yU3, random_state=0, test_size = 0.2)

# %%
#Feature Scaling 
sc_XU3 = StandardScaler()
X_trainu3 = sc_XU3.fit_transform(X_train)
X_testu3 = sc_XU3.transform(X_test)

# %%
import math
math.sqrt(len(y_test))

# %%
#Defining the model
classifier = KNeighborsClassifier(n_neighbors=10, p=2, metric='cosine')

# %%
#fiting the model
classifier.fit(X_train, y_train)

# %%
#predicting the test set results
y_pred = classifier.predict(X_test)

# %%
y_pred

# %%
#evaluation model
cmU3 = confusion_matrix(y_test, y_pred)
print(cmU3)

# %%
#accuracy score
print(accuracy_score(y_test, y_pred))

# %%
#mean_squared_error
mean_squared_error(y_pred, y_test)

# %% [markdown]
# # MERGING U.USER DATASET TO U4.BASE DATASET AND USING U4.TEST

# %%
#mporting u4.base dataset ml-100k/
features = ['userid','movieid','rating','timestamp']
Utraindataset4 = pd.read_csv('Dataset/ml-100k/u4.base', sep ='\t', names=features)
print(Utraindataset4.head())

# %%
#mporting u4.base dataset
features = ['userid','age','gender','occupation','zipcode']
uUser4 = pd.read_csv('Dataset/ml-100k/u.user', sep ='|', names=features)
print(uUser4.head())

# %%
#merging the u4.base dataset with the u.user dataset
Utraindataset4 = pd.merge(Utraindataset4,uUser, on='userid')

# %%
print(Utraindataset4.head())

# %%
Utraindataset4 = pd.get_dummies(Utraindataset4, columns=['occupation', 'gender'])

# %%
Utraindataset4.head(5)

# %%
Utraindataset4 = Utraindataset4.drop(['gender_F', 'occupation_none'], axis =1)
Utraindataset4 = Utraindataset4.drop(['zipcode'], axis =1)

# %%
print(Utraindataset4.head(5))

# %%
Utraindataset4= Utraindataset4[['userid','movieid','age','gender_M','occupation_librarian','occupation_lawyer','occupation_homemaker','occupation_healthcare','occupation_executive','occupation_entertainment','occupation_writer','occupation_technician','occupation_student','occupation_scientist','occupation_salesman','occupation_retired','occupation_programmer','occupation_other','occupation_marketing','occupation_engineer','occupation_educator','occupation_doctor','occupation_administrator','occupation_artist','timestamp','rating']]

# %%
Utraindataset4.shape

# %%
#spliting datasets
XU4 = Utraindataset4.iloc[:,0:24]
yU4 = Utraindataset4.iloc[:,25]
X_train, X_test, y_train, y_test = train_test_split(XU4, yU4, random_state=0, test_size = 0.2)

# %%
#Feature Scaling 
sc_XU4 = StandardScaler()
X_trainu4 = sc_XU4.fit_transform(X_train)
X_testu4 = sc_XU4.transform(X_test)

# %%
import math
math.sqrt(len(y_test))

# %%
#Defining the model
classifier = KNeighborsClassifier(n_neighbors=10, p=2, metric='cosine')

# %%
#fiting the model
classifier.fit(X_train, y_train)

# %%
#predicting the test set results
y_pred = classifier.predict(X_test)

# %%
y_pred

# %%
#evaluation model
cmU4 = confusion_matrix(y_test, y_pred)
print(cmU4)

# %%
#accuracy score
print(accuracy_score(y_test, y_pred))

# %%
#mean_squared_error
mean_squared_error(y_pred, y_test)

# %% [markdown]
# # MERGING U.USER DATASET TO U5.BASE DATASET AND USING U5.TEST

# %%
#mporting u4.base dataset
features = ['userid','movieid','rating','timestamp']
Utraindataset5 = pd.read_csv('Dataset/ml-100k/u5.base', sep ='\t', names=features)
print(Utraindataset5.head())

# %%
#mporting u4.base dataset
features = ['userid','age','gender','occupation','zipcode']
uUser5 = pd.read_csv('Dataset/ml-100k/u.user', sep ='|', names=features)
print(uUser5.head())

# %%
#merging the u4.base dataset with the u.user dataset
Utraindataset5 = pd.merge(Utraindataset5,uUser, on='userid')

# %%
print(Utraindataset5.head())

# %%
Utraindataset5 = pd.get_dummies(Utraindataset5, columns=['occupation', 'gender'])

# %%
Utraindataset5.head(5)

# %%
Utraindataset5 = Utraindataset5.drop(['gender_F', 'occupation_none'], axis =1)
Utraindataset5 = Utraindataset5.drop(['zipcode'], axis =1)

# %%
print(Utraindataset5.head(5))

# %%
Utraindataset5= Utraindataset5[['userid','movieid','age','gender_M','occupation_librarian','occupation_lawyer','occupation_homemaker','occupation_healthcare','occupation_executive','occupation_entertainment','occupation_writer','occupation_technician','occupation_student','occupation_scientist','occupation_salesman','occupation_retired','occupation_programmer','occupation_other','occupation_marketing','occupation_engineer','occupation_educator','occupation_doctor','occupation_administrator','occupation_artist','timestamp','rating']]

# %%
Utraindataset5.shape

# %%
#spliting datasets
XU5 = Utraindataset5.iloc[:,0:24]
yU5 = Utraindataset5.iloc[:,25]
X_train, X_test, y_train, y_test = train_test_split(XU5, yU5, random_state=0, test_size = 0.2)

# %%
#Feature Scaling 
sc_XU5 = StandardScaler()
X_trainu5 = sc_XU5.fit_transform(X_train)
X_testu5 = sc_XU5.transform(X_test)

# %%
import math
math.sqrt(len(y_test))

# %%
#Defining the model
classifier = KNeighborsClassifier(n_neighbors=10, p=2, metric='cosine')

# %%
#fiting the model
classifier.fit(X_train, y_train)

# %%
#predicting the test set results
y_pred = classifier.predict(X_test)

# %%
y_pred

# %%
#evaluation model
cmU5 = confusion_matrix(y_test, y_pred)
print(cmU5)

# %%
#accuracy score
print(accuracy_score(y_test, y_pred))

# %%
#mean_squared_error
mean_squared_error(y_pred, y_test)

# %% [markdown]
# # TASK2 - FINDING RECOMMENDED MOVIES BASED ON MOVIE ID

# %%
# Get ratings people give to movies
features = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('Dataset/ml-100k/u.data', sep='\t', names=features, usecols=range(3))

# %%
ratings.head(5)

# %%
ratings.shape

# %%
# Divide film ratings into total size and average
movieProperties = ratings.groupby('movie_id').agg({'rating': [np.size, np.mean]})

# %%
movieProperties.shape

# %%
movieProperties.head(5)

# %%
# Normalize rating sizes of movies
movieNumRatings = pd.DataFrame(movieProperties['rating']['size'])
movieNormalizedNumRatings = movieNumRatings.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

# %%
movieNormalizedNumRatings.head(5)

# %%
movieNormalizedNumRatings.shape

# %%
# Get film data
movieDict = {}
with open('ml-100k/u.item', mode='r', encoding='latin-1') as f:
    temp = ''
    for line in f:
        fields = line.rstrip('\n').split('|')
        movieID = int(fields[0])
        name = fields[1]
        genres = fields[5:25]
        genres = list(map(int, genres))
        movieDict[movieID] = (name, genres, movieNormalizedNumRatings.loc[movieID].get('size'),
                              movieProperties.loc[movieID].rating.get('mean'))

# %%
print(movieDict)

# %%
# Function to calculate distances between movies
def ComputeDistance(a, b):
    genresA = a[1]
    genresB = b[1]
    genreDistance = spatial.distance.cosine(genresA, genresB)
    popularityA = a[2]
    popularityB = b[2]
    popularityDistance = abs(popularityA - popularityB)
    return genreDistance + popularityDistance

# %%
# Get the neighbor K of the given film
def getNeighbors(movieID, K):
    distances = []
    for movie in movieDict:
        if (movie != movieID):
            dist = ComputeDistance(movieDict[movieID], movieDict[movie])
            distances.append((movie, dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(K):
        neighbors.append(distances[x][0])
    return neighbors

# K IS THE AMOUNT OF HOW MANY NEAREST NEIGHBORS YOU WANT 
# CHANGE THAT VALUE IF YOU WANT MORE 
K = 10 
avgRating = 0
#### THE NUMBER IN THE BRACKET IS THE MOVIE ID 
### CHANGE THE ID IF YOU WANT TO FIND NEAREST NEIGHBORS FOR ANOTHER MOVIE BASED ON THE MOVIE ID
neighbors = getNeighbors(152, K) 
print("Nearest Neighbors:")
print('')
for neighbor in neighbors:
    avgRating += movieDict[neighbor][3]
    print(movieDict[neighbor][0] + " " + str(movieDict[neighbor][3]))

# %%
##### Average rating score calculated based on film neighbors
avgRating /= float(K)  #### avgRating = avgRating / float(K)
print("\nEstimated avg. rating:")
print(avgRating)

# %%
# Real avgerage rating
print("\nReal avg. rating:")
print(movieDict[1][3])

# %% [markdown]
# # TASK2 EXTENDED - User based recommendations

# %%
ratings.head(5)

# %%
ratings
#CHANGE THE USER NUMBER TO A SPECIFIC USER YOU WANT TO FIND RECOMMENDED MOVIES
user = 45

user_film = ratings[ratings['user_id']==user]
user_film = user_film[user_film['rating']==5]

user_film_ids = user_film['movie_id'].values
rec_list = [] 


for film in user_film_ids:
    recs = getNeighbors(film, 5)
    rec_list.append(recs)
    
flat_list = []

for sublist in rec_list:
    for item in sublist:
        flat_list.append(item)
        
        
flat_list.sort()
print(flat_list)

# %%
countList = {i:flat_list.count(i) for i in flat_list}
print (countList)

# %%
films1111 = {k: v for k, v in sorted(countList.items(), key=lambda item: item[1])}

movie_recs = list(films1111.keys())[-5:]
print(movie_recs)

movie_recs.reverse()

print(movie_recs)

# %%
print("Nearest Neighbors based on user id:")
print('')  
final_recs = []
for movieid in movie_recs:
    print(titles[titles['movieid']==movieid]['movie title'].values[0])
    final_recs.append(titles[titles['movieid']==movieid]['movie title'].values[0])

print(final_recs)

# %% [markdown]
# # TASK3 - EXTEND OF TASK2 (INCLUDING U.USER DATASET)

# %%
# Get ratings people give to movies
features = ['user_id', 'movie_id', 'rating']
newRatings = pd.read_csv('Dataset/ml-100k/u.data', sep='\t', names=features, usecols=range(3))

# %%
#importing u.user dataset 
features = ['user_id', 'age', 'gender','occupations','zip code']
userData = pd.read_csv('Dataset/ml-100k/u.user',sep='|',encoding='latin-1', names= features)
print(users.columns.values)

# %%
userData.head()

# %%
userData.shape

# %%
newRatings.head(5)

# %%
newRatings.shape

# %%
newRatings2 = pd.merge(userData, newRatings, on='user_id')

# %%
newRatings2.head(5)

# %%
newRatings2.shape

# %%
#repeat the same process from task2
movieProperties2 = newRatings2.groupby('movie_id').agg({'rating': [np.size, np.mean]})

# %%
movieProperties2.head(5)

# %%
movieProperties2.shape

# %%
# Normalize rating sizes of movies
movieNumRatings2 = pd.DataFrame(movieProperties2['rating']['size'])
movieNormalizedNumRatings = movieNumRatings2.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

# %%
movieNormalizedNumRatings.head(5)

# %%
movieNormalizedNumRatings.shape

# %%
# Get film data
movieDict = {}
with open('ml-100k/u.item', mode='r', encoding='latin-1') as f:
    temp = ''
    for line in f:
        fields = line.rstrip('\n').split('|')
        movieID = int(fields[0])
        name = fields[1]
        genres = fields[5:25]
        genres = list(map(int, genres))
        movieDict[movieID] = (name, genres, movieNormalizedNumRatings.loc[movieID].get('size'),
                              movieProperties.loc[movieID].rating.get('mean'))

# %%
print(movieDict)

# %%
# Function to calculate distances between movies
def ComputeDistance(a, b):
    genresA = a[1]
    genresB = b[1]
    genreDistance = spatial.distance.cosine(genresA, genresB)
    popularityA = a[2]
    popularityB = b[2]
    popularityDistance = abs(popularityA - popularityB)
    return genreDistance + popularityDistance

# %%
# Get the neighbor K of the given film
def getNeighbors(movieID, K):
    distances = []
    for movie in movieDict:
        if (movie != movieID):
            dist = ComputeDistance(movieDict[movieID], movieDict[movie])
            distances.append((movie, dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(K):
        neighbors.append(distances[x][0])
    return neighbors


K = 5 # AMOUNT OF HOW MANY NEAREST NEIGHBORS YOU WANT #### CHANGE THAT VALUE IF YOU WANT MORE 
avgRating = 0
neighbors = getNeighbors(300, K) #### THE NUMBER IN THE BRACKET IS THE MOVIE ID ### CHANGE THE ID IF YOU WANT TO FIND NEAREST NEIGHBORS FOR ANOTHER MOVIE
print("Nearest Neighbors:")
print('')
for neighbor in neighbors:
    avgRating += movieDict[neighbor][3]
    print(movieDict[neighbor][0] + " " + str(movieDict[neighbor][3]))

# %%
##### Average rating score calculated based on film neighbors
avgRating /= float(K)  #### avgRating = avgRating / float(K)
print("\nEstimated avg. rating:")
print(avgRating)

# %%
# Real avgerage rating
print("\nReal avg. rating:")
print(movieDict[1][3])

# %% [markdown]
# # TASK3 EXTENDED - User based recommendations

# %%
Utraindataset5.head(5)

# %%
Utraindataset5
#CHANGE THE USER NUMBER TO A SPECIFIC USER YOU WANT TO FIND RECOMMENDED MOVIES
user = 45

#change the ##Utraindataset5 to another dataset that was trained above to find simalar users and recommendations
user_film = Utraindataset5[Utraindataset5['userid']==user]
user_film = user_film[user_film['rating']==5]

user_film_ids = user_film['movieid'].values
rec_list = [] 


for film in user_film_ids:
    recs = getNeighbors(film, 5)
    rec_list.append(recs)
    
flat_list = []

for sublist in rec_list:
    for item in sublist:
        flat_list.append(item)
        
        
flat_list.sort()
print(flat_list)

# %%
countList = {i:flat_list.count(i) for i in flat_list}
print (countList)

# %%
films1111 = {k: v for k, v in sorted(countList.items(), key=lambda item: item[1])}

movie_recs = list(films1111.keys())[-5:]
print(movie_recs)

movie_recs.reverse()

print(movie_recs)

# %%
print("Nearest Neighbors based on user id:")
print('')  
final_recs = []
for movieid in movie_recs:
    print(titles[titles['movieid']==movieid]['movie title'].values[0])
    final_recs.append(titles[titles['movieid']==movieid]['movie title'].values[0])

print(final_recs)

# %% [markdown]
# # LightFM

# %%
from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k

# Load the MovieLens 100k dataset. Only five
# star ratings are treated as positive.
data = fetch_movielens(min_rating=3.0)

# Instantiate and train the model
model = LightFM(loss='warp')
model.fit(data['train'], epochs=30, num_threads=2)

# Evaluate the trained model
test_precision = precision_at_k(model, data['test'], k=5).mean()

# %%
test_precision

# %%
model

# %%
data

# %%
import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#fetch data and format it
data = fetch_movielens(min_rating=4.0)

# print training and testing data

print(repr(data["train"]))
print(repr(data["test"]))

# %%
# crate model
model = LightFM(loss='warp')

# %%
model.fit(data['train'], epochs=30, num_threads=2)

# %%
def sample_recommendation(model, data, user_ids):
    # number of users and movies in training data
    n_users, n_items = data['train'].shape
    
    # generate recommendations for each user we input
    for user_id in user_ids:
        
        #mobies the already like
        known_positives=data['item_labels'][data['train'].tocsr()[user_id].indices]
        
        #movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))
        
        #rank them in order of most liked to least
        top_items=data['item_labels'][np.argsort(-scores)]
        
        #print out the results
        print("User %s" % user_id)
        print("     Known positives:")
        
        for x in known_positives[:5]:
            print("           %s"% x)
            
        print("     Recommended:")
        
        for x in top_items[:5]:
            print("           %s"% x)
        

# %%
sample_recommendation(model,data,[4,25,450])

# %%
model.get_item_representations


