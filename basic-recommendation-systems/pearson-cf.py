# %% [markdown]
# ## pearson

# %%
import numpy as np
from scipy import stats, io
import warnings
warnings.filterwarnings("ignore")

mat = io.loadmat('ss.mat')
matrix=mat["ss"]
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

# %%
def pred(average_list,similarity_row,rating_row_matrix,pred_id):
    pred=0   


