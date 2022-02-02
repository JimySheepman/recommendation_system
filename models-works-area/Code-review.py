# %%
import keras
import warnings
import numpy as np
import pandas as pd
from math import sqrt
from sklearn import metrics
from keras.optimizers import Adam
from sklearn.cluster import KMeans
from keras.models import load_model
from scipy.sparse.linalg import svds
from models import matrix_factorization_utilities
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Activation, Flatten ,Embedding, Reshape, concatenate
warnings.filterwarnings('ignore')

# %% [markdown]
# ###  1. Data Load

# %%
# Reading the ratings data
ratings = pd.read_csv('Dataset/ratings.csv')

# %%
ratings

# %%
ratings = ratings[['userId', 'movieId','rating']]

# %%
ratings

# %%
    #reading the movies dataset
    movies = pd.read_csv('Dataset/movies.csv')


# %%
movies

# %%
#reading the links dataset
links = pd.read_csv('Dataset/movies.dat',sep='\t', encoding='latin-1')

# %%
links

# %%
#drop link dataframe
for i in links.columns:
    if i=='id'or i == 'title'or i=='year' or i == 'rtPictureURL':
        continue
    else:
        links.drop(i, inplace=True, axis=1)

# %%
links

# %%
links.columns=['movieId','title_old', 'year', 'rtPictureURL'] # columns rename

# %%
links

# %%
# title_old and year formating merge and add new columns
links['title']=links.apply(lambda x:'%s (%s)' % (x['title_old'],x['year']),axis=1)

# %%


# %%
links

# %%
# drop old columns
links.drop('title_old', inplace=True, axis=1)
links.drop('year', inplace=True, axis=1)

# %%
links

# %%
#replace columns
links=links[['movieId','title','rtPictureURL']]

# %%
links

# %%
# merge dataframe 
movie_list=pd.merge(movies,links, on='movieId')

# %%
# drop column
movie_list.drop('title_y', inplace=True, axis=1)

# %%
# rename columns
movie_list.columns=['movieId','title','genres','rtPictureURL']

# %%
ratings.userId

# %%
# convert type category
ratings.userId = ratings.userId.astype('category').cat.codes.values

# %%
ratings.userId

# %%
ratings.userId.max()

# %%
ratings.movieId = ratings.movieId.astype('category').cat.codes.values

# %%
ratings.movieId

# %%
ratings.movieId.max()

# %%
#take 80% as the training set and 20% as the test set
train, test= train_test_split(ratings,test_size=0.2)

# %%
train

# %%
test

# %% [markdown]
# ### 2.Matrix Factorisation Model

# %% [markdown]
# #### 2.1. Matrix Factorisation Model Calculate Best n_latent

# %% [markdown]
# **Latent features:** At the expense of over-simplication, latent features are 'hidden' features to distinguish them from observed features. Latent features are computed from observed features using matrix factorization. An example would be text document analysis. 'words' extracted from the documents are features.

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

# %%
#Calculate the rmse sscore of SVD using different values of k (latent features)
rmse_list = []
mae_list = []
accuracy=[]
pred_list=[]
a=[1,2,5,10,20]
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

# %% [markdown]
# ```
# keras.Input(
#     shape=None,
#     batch_size=None,
#     name=None,
#     dtype=None,
#     sparse=None,
#     tensor=None,
#     ragged=None,
#     type_spec=None,
#     **kwargs
# )
# ```
# ##### Arguments
# 
# * shape: A shape tuple (integers), not including the batch size. For instance, shape=(32,) indicates that the expected input will be batches of 32-dimensional vectors. Elements of this tuple can be None; 'None' elements represent dimensions where the shape is not known.
# 
# * batch_size: optional static batch size (integer).
# 
# * name: An optional name string for the layer. Should be unique in a model (do not reuse the same name twice). It will be autogenerated if it isn't provided.
#  
# * dtype: The data type expected by the input, as a string (float32, float64, int32...)
# 
# * sparse: A boolean specifying whether the placeholder to be created is sparse. Only one of 'ragged' and 'sparse' can be True. Note that, if sparse is False, sparse tensors can still be passed into the input - they will be densified with a default value of 0.
# 
# * tensor: Optional existing tensor to wrap into the Input layer. If set, the layer will use the tf.TypeSpec of this tensor rather than creating a new placeholder tensor.
# 
# * ragged: A boolean specifying whether the placeholder to be created is ragged. Only one of 'ragged' and 'sparse' can be True. In this case, values of 'None' in the 'shape' argument represent ragged dimensions. For more information about RaggedTensors, see this guide.
# 
# * type_spec: A tf.TypeSpec object to create the input placeholder from. When provided, all other args except name must be None.
# 
# * **kwargs: deprecated arguments support. Supports batch_shape and batch_input_shape.
# 
# 

# %%
# girdi boyutunu belirler movies
movie_input = keras.layers.Input(shape=[1],name='Item')

# %% [markdown]
# ```
# tf.keras.layers.Embedding(
#     input_dim,
#     output_dim,
#     embeddings_initializer="uniform",
#     embeddings_regularizer=None,
#     activity_regularizer=None,
#     embeddings_constraint=None,
#     mask_zero=False,
#     input_length=None,
#     **kwargs
# )
# ```
# 
# #### Arguments
# 
# * input_dim: Integer. Size of the vocabulary, i.e. maximum integer index + 1.
# * output_dim: Integer. Dimension of the dense embedding.
# * embeddings_initializer: Initializer for the embeddings matrix (see keras.initializers).
# * embeddings_regularizer: Regularizer function applied to the embeddings matrix (see keras.regularizers).
# * embeddings_constraint: Constraint function applied to the embeddings matrix (see keras.constraints).
# * mask_zero: Boolean, whether or not the input value 0 is a special "padding" value that should be masked out. This is useful when using recurrent layers which may take variable length input. If this is True, then all subsequent layers in the model need to support masking or an exception will be raised. If mask_zero is set to True, as a consequence, index 0 cannot be used in the vocabulary (input_dim should equal size of vocabulary + 1).
# * input_length: Length of input sequences, when it is constant. This argument is required if you are going to connect Flatten then Dense layers upstream (without it, the shape of the dense outputs cannot be computed).

# %%
#input_dim = movie index +1
# output_dim = Dimension of the dense embedding.
movie_embedding = keras.layers.Embedding(14026 + 1, 1, name='Movie-Embedding')(movie_input)

# %% [markdown]
# ```
# tf.keras.layers.Flatten(data_format=None, **kwargs)
# ```
# 
# Flattens the input. Does not affect the batch size.
# 
# * Note: If inputs are shaped (batch,) without a feature axis, then flattening adds an extra channel dimension and output shape is (batch, 1).
# 
# #### Arguments
# 
# * data_format: A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, ..., channels) while channels_first corresponds to inputs with shape (batch, channels, ...). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".

# %%
# Reshaping layers
movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)

# %%
# girdi boyutunu belirler users
user_input = keras.layers.Input(shape=[1],name='User')


# %% [markdown]
# Girdi verilerinin tamsayı olarak kodlanmasını gerektirir, böylece her kelime benzersiz bir tamsayı ile temsil edilir. Bu veri hazırlama adımı, Keras ile birlikte sağlanan Tokenizer API kullanılarak gerçekleştirilebilir.
# 
# Gömme katmanı rastgele ağırlıklarla başlatılır ve eğitim veri setindeki tüm kelimeler için bir yerleştirme öğrenir.

# %%

user_vec = keras.layers.Flatten(name='FlattenUsers')(keras.layers.Embedding(7120 + 1, 1,name='User-Embedding')(user_input))


# %%
# merge e embeedding layer input

# %%
prod = keras.layers.concatenate([movie_vec, user_vec],name='DotProduct')


# %% [markdown]
# ```
# tf.keras.layers.Dense(
#     units,
#     activation=None,
#     use_bias=True,
#     kernel_initializer="glorot_uniform",
#     bias_initializer="zeros",
#     kernel_regularizer=None,
#     bias_regularizer=None,
#     activity_regularizer=None,
#     kernel_constraint=None,
#     bias_constraint=None,
#     **kwargs
# )
# ```
# #### Arguments
# 
#  * units: Positive integer, dimensionality of the output space.
#  * activation: Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
#  * use_bias: Boolean, whether the layer uses a bias vector.
#  * kernel_initializer: Initializer for the kernel weights matrix.
#  * bias_initializer: Initializer for the bias vector.
#  * kernel_regularizer: Regularizer function applied to the kernel weights matrix.
#  * bias_regularizer: Regularizer function applied to the bias vector.
#  * activity_regularizer: Regularizer function applied to the output of the layer (its "activation").
#  * kernel_constraint: Constraint function applied to the kernel weights matrix.
#  * bias_constraint: Constraint function applied to the bias vector.

# %% [markdown]
#  etkinleştirme argümanı olarak iletilen öğe bazlı etkinleştirme işlevidir, çekirdek, katman tarafından oluşturulan bir ağırlık matrisidir ve önyargı, oluşturulan bir önyargı vektörüdür katmana göre (yalnızca use_bias True ise geçerlidir).

# %%
result = keras.layers.Dense(1, activation='relu',name='Activation')(prod)
# Eğittiğimiz yapay sinir ağı modeline gerçek dünya verilerini,
# yani karmaşık verileri öğretebilmek için kullandığımız fonksiyonlardır.
# relu =[0,+∞)
# ReLU fonksiyonunun ana avantajı aynı anda tüm nöronları aktive etmemesidir.

# %% [markdown]
# * Derin öğrenme uygulamalarında öğrenme işleminin temelde bir optimizasyon problemi olduğu daha önce vurgulanmıştı.
# * Doğrusal olmayan problemlerin çözümünde optimum değeri bulmak için optimizasyon yöntemleri kullanılmaktadır. 

# %%
# lr =learning rate

# %%
adam = Adam(lr=0.05)


# %%
# creating model
model = keras.Model([user_input, movie_input], result)

# %% [markdown]
# ```
# Model.compile(
#     optimizer="rmsprop",
#     loss=None,
#     metrics=None,
#     loss_weights=None,
#     weighted_metrics=None,
#     run_eagerly=None,
#     steps_per_execution=None,
#     **kwargs
# )
# ```
# #### Arguments
# 
# * optimizer: String (name of optimizer) or optimizer instance. See tf.keras.optimizers.
# * loss: String (name of objective function), objective function or tf.keras.losses.Loss instance. See tf.keras.losses. An objective function is any callable with the signature loss = fn(y_true, y_pred), where y_true = ground truth values with shape = [batch_size, d0, .. dN], except sparse loss functions such as sparse categorical crossentropy where shape = [batch_size, d0, .. dN-1]. y_pred = predicted values with shape = [batch_size, d0, .. dN]. It returns a weighted loss float tensor. If a custom Loss instance is used and reduction is set to NONE, return value has the shape [batch_size, d0, .. dN-1] ie. per-sample or per-timestep loss values; otherwise, it is a scalar. If the model has multiple outputs, you can use a different loss on each output by passing a dictionary or a list of losses. The loss value that will be minimized by the model will then be the sum of all individual losses.
# * metrics: List of metrics to be evaluated by the model during training and testing. Each of this can be a string (name of a built-in function), function or a tf.keras.metrics.Metric instance. See tf.keras.metrics. Typically you will use metrics=['accuracy']. A function is any callable with the signature result = fn(y_true, y_pred). To specify different metrics for different outputs of a multi-output model, you could also pass a dictionary, such as metrics={'output_a': 'accuracy', 'output_b': ['accuracy', 'mse']}. You can also pass a list (len = len(outputs)) of lists of metrics such as metrics=[['accuracy'], ['accuracy', 'mse']] or metrics=['accuracy', ['accuracy', 'mse']]. When you pass the strings 'accuracy' or 'acc', we convert this to one of tf.keras.metrics.BinaryAccuracy, tf.keras.metrics.CategoricalAccuracy, tf.keras.metrics.SparseCategoricalAccuracy based on the loss function used and the model output shape. We do a similar conversion for the strings 'crossentropy' and 'ce' as well.
# * loss_weights: Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss contributions of different model outputs. The loss value that will be minimized by the model will then be the weighted sum of all individual losses, weighted by the loss_weights coefficients. If a list, it is expected to have a 1:1 mapping to the model's outputs. If a dict, it is expected to map output names (strings) to scalar coefficients.
# * weighted_metrics: List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing.
# * run_eagerly: Bool. Defaults to False. If True, this Model's logic will not be wrapped in a tf.function. Recommended to leave this as None unless your Model cannot be run inside a tf.function. run_eagerly=True is not supported when using tf.distribute.experimental.ParameterServerStrategy.
# * steps_per_execution: Int. Defaults to 1. The number of batches to run during each tf.function call. Running multiple batches inside a single tf.function call can greatly improve performance on TPUs or small models with a large Python overhead. At most, one full epoch will be run each execution. If a number larger than the size of the epoch is passed, the execution will be truncated to the size of the epoch. Note that if steps_per_execution is set to N, Callback.on_batch_begin and Callback.on_batch_end methods will only be called every N batches (i.e. before/after each tf.function execution).
# * **kwargs: Arguments supported for backwards compatibility only.

# %%
model.compile(optimizer=adam,loss= 'mean_absolute_error', metrics=['accuracy'])

# %%
# model details
model.summary()

# %% [markdown]
# ```
# Model.fit(
#     x=None,
#     y=None,
#     batch_size=None,
#     epochs=1,
#     verbose="auto",
#     callbacks=None,
#     validation_split=0.0,
#     validation_data=None,
#     shuffle=True,
#     class_weight=None,
#     sample_weight=None,
#     initial_epoch=0,
#     steps_per_epoch=None,
#     validation_steps=None,
#     validation_batch_size=None,
#     validation_freq=1,
#     max_queue_size=10,
#     workers=1,
#     use_multiprocessing=False,
# )
# ```
# 
# #### Arguments
# 
# * x: Input data. It could be:
#   * A Numpy array (or array-like), or a list of arrays (in case the model has multiple inputs).
#   * A TensorFlow tensor, or a list of tensors (in case the model has multiple inputs).
#   * A dict mapping input names to the corresponding array/tensors, if the model has named inputs.
#   * A tf.data dataset. Should return a tuple of either (inputs, targets) or (inputs, targets, sample_weights).
#   * A generator or keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample_weights).
#   * A tf.keras.utils.experimental.DatasetCreator, which wraps a callable that takes a single argument of type tf.distribute.InputContext, and returns a tf.data.Dataset. DatasetCreator should be used when users prefer to specify the per-replica batching and sharding logic for the Dataset. See tf.keras.utils.experimental.DatasetCreator doc for more information. A more detailed description of unpacking behavior for iterator types (Dataset, generator, Sequence) is given below. If using tf.distribute.experimental.ParameterServerStrategy, only DatasetCreator type is supported for x.
# * y: Target data. Like the input data x, it could be either Numpy array(s) or TensorFlow tensor(s). It should be consistent with x (you cannot have Numpy inputs and tensor targets, or inversely). If x is a dataset, generator, or keras.utils.Sequence instance, y should not be specified (since targets will be obtained from x).
# * batch_size: Integer or None. Number of samples per gradient update. If unspecified, batch_size will default to 32. Do not specify the batch_size if your data is in the form of datasets, generators, or keras.utils.Sequence instances (since they generate batches).
# * epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided. Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached.
# * verbose: 'auto', 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 'auto' defaults to 1 for most cases, but 2 when used with ParameterServerStrategy. Note that the progress bar is not particularly useful when logged to a file, so verbose=2 is recommended when not running interactively (eg, in a production environment).
# * callbacks: List of keras.callbacks.Callback instances. List of callbacks to apply during training. See tf.keras.callbacks. Note tf.keras.callbacks.ProgbarLogger and tf.keras.callbacks.History callbacks are created automatically and need not be passed into model.fit. tf.keras.callbacks.ProgbarLogger is created or not based on verbose argument to model.fit. Callbacks with batch-level calls are currently unsupported with tf.distribute.experimental.ParameterServerStrategy, and users are advised to implement epoch-level calls instead with an appropriate steps_per_epoch value.
# * validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the x and y data provided, before shuffling. This argument is not supported when x is a dataset, generator or keras.utils.Sequence instance. validation_split is not yet supported with tf.distribute.experimental.ParameterServerStrategy.
# * validation_data: Data on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. Thus, note the fact that the validation loss of data provided using validation_split or validation_data is not affected by regularization layers like noise and dropout. validation_data will override validation_split. validation_data could be: - A tuple (x_val, y_val) of Numpy arrays or tensors. - A tuple (x_val, y_val, val_sample_weights) of NumPy arrays. - A tf.data.Dataset. - A Python generator or keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample_weights). validation_data is not yet supported with tf.distribute.experimental.ParameterServerStrategy.
# * shuffle: Boolean (whether to shuffle the training data before each epoch) or str (for 'batch'). This argument is ignored when x is a generator or an object of tf.data.Dataset. 'batch' is a special option for dealing with the limitations of HDF5 data; it shuffles in batch-sized chunks. Has no effect when steps_per_epoch is not None.
# * class_weight: Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
# * sample_weight: Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only). You can either pass a flat (1D) Numpy array with the same length as the input samples (1:1 mapping between weights and samples), or in the case of temporal data, you can pass a 2D array with shape (samples, sequence_length), to apply a different weight to every timestep of every sample. This argument is not supported when x is a dataset, generator, or keras.utils.Sequence instance, instead provide the sample_weights as the third element of x.
# * initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
# * steps_per_epoch: Integer or None. Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined. If x is a tf.data dataset, and 'steps_per_epoch' is None, the epoch will run until the input dataset is exhausted. When passing an infinitely repeating dataset, you must specify the steps_per_epoch argument. This argument is not supported with array inputs. steps_per_epoch=None is not supported when using tf.distribute.experimental.ParameterServerStrategy.
# * validation_steps: Only relevant if validation_data is provided and is a tf.data dataset. Total number of steps (batches of samples) to draw before stopping when performing validation at the end of every epoch. If 'validation_steps' is None, validation will run until the validation_data dataset is exhausted. In the case of an infinitely repeated dataset, it will run into an infinite loop. If 'validation_steps' is specified and only part of the dataset will be consumed, the evaluation will start from the beginning of the dataset at each epoch. This ensures that the same validation samples are used every time.
# * validation_batch_size: Integer or None. Number of samples per validation batch. If unspecified, will default to batch_size. Do not specify the validation_batch_size if your data is in the form of datasets, generators, or keras.utils.Sequence instances (since they generate batches).
# * validation_freq: Only relevant if validation data is provided. Integer or collections.abc.Container instance (e.g. list, tuple, etc.). If an integer, specifies how many training epochs to run before a new validation run is performed, e.g. validation_freq=2 runs validation every 2 epochs. If a Container, specifies the epochs on which to run validation, e.g. validation_freq=[1, 2, 10] runs validation at the end of the 1st, 2nd, and 10th epochs.
# * max_queue_size: Integer. Used for generator or keras.utils.Sequence input only. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
# * workers: Integer. Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1.
# * use_multiprocessing: Boolean. Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False. Note that because this implementation relies on multiprocessing, you should not pass non-picklable arguments to the generator as they can't be passed easily to children processes.

# %%
# x: input
# y: target
# epochs number of epochs to train the model
# verbose: for 0: don't show result ,for 1: step to step show result , for 2:every complete epoch then show result
# val_split: Model, eğitim verilerinin bu bölümünü ayıracak, 
# bunun üzerinde çalışmayacak ve her dönemin sonunda bu verilerdeki kaybı ve model ölçütlerini değerlendirecektir. 
history = model.fit([train.userId, train.movieId], train.rating,  epochs=10,batch_size=64,verbose=1, validation_split=0.1)

# %%
a=model.evaluate([train.userId, train.movieId], train.rating , batch_size=128)

# %%
a

# %%
y_hat = np.round(model.predict([test.userId, test.movieId]),0)
y_true = test.rating

# %%
mean_absolute_error(y_true, y_hat)

# %%
errors=mean_absolute_error(y_true, y_hat)
mape = 100 * (errors / y_true)
accuracy = 100 - np.mean(mape)

# %%
print("(MAE)Mean Absolute Error:",round(mean_absolute_error(y_true, y_hat),4))
print("(RMSE)Root Mean Square Error:",round(np.sqrt(mean_squared_error(y_true,y_hat)) ,4))
print(' Accuracy:', round(accuracy, 2), '%.')

# %%
mlp_user_embedding_weights = (next(iter(filter(lambda x: x.name == 'User-Embedding', model.layers))).get_weights())

# %%
print(mlp_user_embedding_weights)
len(mlp_user_embedding_weights[0])

# %%
user_latent_matrix = mlp_user_embedding_weights[0]

# %%
user_latent_matrix

# %%
desired_user_id=38

# %%
one_user_vector = user_latent_matrix[desired_user_id,:]

# %%
one_user_vector

# %%
one_user_vector = np.reshape(one_user_vector, (1,1))

# %%
one_user_vector

# %%
print('Performing kmeans to find the nearest users...')

# %%
kmeans = KMeans(n_clusters=20, random_state=0, verbose=0).fit(user_latent_matrix)

# %%
kmeans

# %%
desired_user_label = kmeans.predict(one_user_vector)

# %%
desired_user_label

# %%
user_label = kmeans.labels_

# %%
user_label

# %%
neighbors = []
for user_id, user_label in enumerate(user_label):
    if user_label == desired_user_label:
        neighbors.append(user_id)

# %%
neighbors

# %%
print('Found {0} neighbor users.'.format(len(neighbors)))

# %%
movies = []
for user_id in neighbors:
    movies += list(ratings[ratings['userId'] == int(user_id)]['movieId'])
    

# %%
movies = list(set(movies))

# %%
movies

# %%
print('Found {0} neighbor movies from these users.'.format(len(movies)))

# %%
users = np.full(len(movies), desired_user_id, dtype='int32')


# %%
users

# %%
items = np.array(movies, dtype='int32')

# %%
items

# %%
print('Ranking most likely tracks using the NeuMF model...')

# %%
results = model.predict([users,items], verbose=1)

# %%
results = results.tolist()

# %%
results

# %%
print('Ranked the movies!')

# %%
results = pd.DataFrame(results, columns=['pre_rating']).astype("float")

# %%
results

# %%
items = pd.DataFrame(items, columns=['movieId'])

# %%
items

# %%
results = pd.concat([items, results], ignore_index=True, sort=False, axis=1)

# %%
results

# %%
results.columns =['movieId', 'pre_rating']

# %%
results

# %%
results_df = pd.DataFrame(np.nan, index=range(len(results)), columns=['pre_rating','movieId'])

# %%
results_df

# %%
for index, row in results.iterrows():
        results_df.loc[index] = [row['pre_rating'], ratings[ratings['movieId'] == row['movieId']].iloc[0]['movieId']]

# %%
results_df

# %%
results_df= results_df.sort_values(by=['pre_rating'], ascending=False)

# %%
results_df

# %%
results_df["movieId"]=results_df["movieId"].astype(int)

# %%
results_df

# %%
results_df=pd.merge(results_df,movie_list,on="movieId")[:10]

# %%
results_df

# %%



