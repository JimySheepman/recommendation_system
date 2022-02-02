# %%
import warnings
warnings.filterwarnings('ignore')

# %%
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# %%
# machine learning
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR 
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# %%
rs_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
df = pd.read_csv('Dataset/ml-100k/ua.base', sep='\t', names=rs_cols, encoding='latin-1')

# %%
df.head()

# %%
df.columns

# %%
df=df.drop(columns=['unix_timestamp'])

# %%
df.columns

# %%
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

# %%
df['user_id']=le.fit_transform(df['user_id'].values)

# %%
y=df['rating'].values
x=df.loc[:, df.columns != 'rating'].values

# %%

X_train, X_test, Y_trian, Y_test= train_test_split(x,y, test_size=0.2, random_state=0)

# %%
df.head

# %%
X_train, X_test, Y_trian, Y_test= train_test_split(x,y, test_size=0.2, random_state=0)

# %% [markdown]
# ### Random Forest
# 

# %%

randregrssor=RandomForestRegressor(n_estimators=50)

# %%
randregrssor.fit(X_train,Y_trian)

# %%
predictions = randregrssor.predict(X_test)

# %%
errors = abs(predictions - Y_test)

# %%
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# %%

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / Y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# %% [markdown]
# ### Decision Tree
# 

# %%
dr=DecisionTreeRegressor()

# %%
dr.fit(X_train,Y_trian)

# %%
predictions = dr.predict(X_test)

# %%
errors = abs(predictions - Y_test)

# %%
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# %%

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / Y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# %% [markdown]
# ### Support Vector Machine

# %%
svm=SVR()

# %%
svm.fit(X_train,Y_trian)

# %%
predictions = svm.predict(X_test)

# %%
errors = abs(predictions - Y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / Y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# %% [markdown]
# ### XG Boost

# %%
from xgboost import XGBRegressor

# %%
xgboost=XGBRegressor(n_estimators=50)

# %%
xgboost.fit(X_train,Y_trian)

# %%
predictions=xgboost.predict(X_test)

# %%
errors = abs(predictions - Y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / Y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


