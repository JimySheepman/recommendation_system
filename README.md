# Movie Recommendation Systems 



We did this project with my teammates as a university graduation project.The main purpose of the project was to understand the movie proposal systems and realize a project by putting them into practice. In this project, I have experimented with multiple data sets and multiple machine learning and deep learning models. We followed a path from simple to difficult. I used flask for interface and model integration. 

#### Project Role Description:
  My team consisted of 4 people. My job as a team leader was managing the entire process, creating models, programming frontend and backend. **Thank you to my teammates in the project.** 

# Data Sets
Data Sets Links:

* [grouplens-hetrec](https://grouplens.org/datasets/hetrec-2011/)
* [grouplens-movielens](https://grouplens.org/datasets/movielens/)

Dataset Definitions:
| Datasets name| users | movies | ratings |
| --- | --- | --- | --- |
| movielens 100k | 943| 1682 | 100,000|
| movielen 1m | 7206 | 1426 | 855598 |
| hetrec2011-movielens-2k | 2113 | 10197 | 855598 |


# Used Models 



* ##### Basic Recommendation Systems:
  * Firstly; I tried to get results by applying user based CF manually. I used pearson and cosine similarity for similarity calculations.
  * Then I calculated the RMSE on the 10% test and 90% training matrix that I randomly generated in 100 thousand data sets and performed the same operations on 1 million data sets. 

  * Model Rusults:

 Dataset name| RMSE
 ------- | ------
 movielens 100k | 2.5719 
 movielen 1m | 2.8785 

* See this file for details:

```bash
basic_recommendations.ipynb
```

* ##### KNN and lightfm:
  * I tried to get results from the KNN model by applying user-based CF. I worked with movielens-100k dataset. I found the top 5 recommendations. 
  * I tried lightfm and just got advice. 

  * Model Rusults:

| Merge Dataset| f1 score| accuracy score | mean squared error |
| --- | --- | --- | --- |
| U.DATA  & U.USER  | 0.1715 | 0.3379 | 1.4461 |
| U2.BASE & U2.TEST | 0.1744 | 0.3382 | 1.4313 |
| U3.BASE & U3.TEST | 0.1589 | 0.3206 | 1.5639 |
| U4.BASE & U4.TEST | 0.1779 | 0.3374 | 1.4776 |
| U5.BASE & U5.TEST | 0.2195 | 0.2999 | 1.8332 |
| U5.BASE & U5.TEST | 0.2195 | 0.2999 | 1.8332 |
| U.USER & U1.BASE | - | 0.3638 | 1.577 |
| U.USER & U2.BASE & U2.TEST | - | 0.3047 | 1.794 |
| U.USER & U3.BASE & U3.TEST | - | 0.3106 | 1.8131 |
| U.USER & U4.BASE & U4.TEST | - | 0.3042 | 1.7876 |
| U.USER & U5.BASE & U5.TEST | - | 0.3106 | 1.7701 |



  * See this file for details:
```bash
KNN_and_lightfm.ipynb
```
* ##### Machine Learning Models

  * I tried 4 different models on the movielens-100k dataset. I used  %20 test  and  %80 train.
     * RandomForestRegressor(n_estimators=50)
      * XGBRegressor(n_estimators=50)

  * Model Rusults:

| model name| MEA | ACC |
| --- | --- | --- |
| Random Forest | 0.88 | %65.2 |
| Decision Tree | 1.05 | %59.93 |
| Support Vector Machine | 0.87 | %61.22 |
| XG Boost | 0.79 | %67.39 |


  * See this file for details:
```bash
try_ML_models.ipynb
```

* ##### Deep Learning Models

  * ###### Matrix Factorization Model(MFM): 
    * Calculate svd to test data and rmse score of matrix factorization to find the best n_latent and send it to the best n_latent model. I observed the result of long observations as the best n_latent = 1 for movielens-1m dataset. 

  * ###### Neural Network Model(NNM):
    * I set up a neural network and worked on movielens-1.

  * ###### Neural Collaborative Filtering Model:
    * Set up a CF-based neural network and worked on movielens-1. 

  * Finally, I examined them all on the same data set. 
  * Model Rusults:

| model name | MEA | RMSE | ACC |
| --- | --- | --- | --- |
| MFM | 0.643 | 0.8593 | %78.19 |
| NNM | 0.6197 | 0.8411 | %78.98 |
| NCF | 0.609 | 0.8527 | %79.34 |



  * See this file for details:
```bash
matrix_factorisation_model.ipynb
neural_network_model.ipynb
neural_collaborative_filtering_model.ipynb
combine_models.ipynb
```


# Rest API Application

* In this application, a user id and model are selected by the user and the best 10 recommendations are printed on the screen. The calculation process takes time.
* In addition, model information, rating prediction, movie, title and genres are printed on the screen. 
* Finally, it shows the brochures of the movies. 

###### Subdirectories:
```
├── Dataset
    ├── ml-1m
        └── # dataset 
    └── ml-100k
        └── # dataset 
├── models
    └── # .h5 models & matrix_factorization_utilities.py
├── statics
    ├── css
        └── # css files
    ├── img
        └── # image & icon files
    ├── js
        └── # js files
    └── scr
        ├── js
            └── # js files
        └── scss
            └── # scss files
├── templates
    └── #html files
└── # code files
```

# Tools List
 * Jupyter Notebook
 * Visual Studio Code
   * ###### Helper Extension List:
     * Auto Close Tag
     * Bracket Pair Colorizer 2
     * ESLint
     * flask-snippets
     * HTML CSS Supports
     * HTML Snippets
     * JavaScript (E6)
     * Jinja2 Snippets Kit
     * Prettier - Code formater
     * Python
     * vscode-icons

# Quick Start

### Installation

#### Install virtualenv


```
# Windows & Mac OS:

pip install virtualenv

# Linux
sudo pip3 install virtualenv 
```

#### Create an environment

```
$ mkdir myproject
$ cd myproject
$ python3 -m venv venv
```

#### Activate the environment
Before you work on your project, activate the corresponding environment:

```
$ . venv/bin/activate
```
On Windows:

```
venv\Scripts\activate
```


#### Install Requirements

```
pip install -r requirements.txt
```
#### Start project

```
$ export FLASK_APP=main.py
$ export FLASK_ENV=development
$ flask run
```
On Windows:

```
set FLASK_APP=main.py
set FLASK_APP=development
flask run
```
# Docker 
Building Dockerfile

```
$ docker build -t recomendation:1.0 .
```
Running Docker image
```
$ docker run -p 5000:5000 recomendation:1.0
```
Now, the application should be accessible http://0.0.0.0:5000/

# Development

###### Useful Resources: 
* Models:
  * [TensorFlow](https://www.tensorflow.org/guide)
  * [Keras](https://keras.io/api/)
  * [pandas](https://pandas.pydata.org/docs/)
  * [NumPy](https://numpy.org/doc/1.20/)
  * [scikit-learn](https://scikit-learn.org/0.21/documentation.html)
* Design 
  * [mdbootstrap](https://mdbootstrap.com/docs/standard/getting-started/installation/)
  * [getbootstrap](https://getbootstrap.com/docs/5.0/getting-started/introduction/)
  * [HTML,CSS,JavaScript](https://www.w3schools.com)
* Backend
  * [Flask](https://flask.palletsprojects.com/en/1.1.x/)

## License
[MIT](https://choosealicense.com/licenses/mit/)