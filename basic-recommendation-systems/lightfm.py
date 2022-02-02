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
# train model

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


