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