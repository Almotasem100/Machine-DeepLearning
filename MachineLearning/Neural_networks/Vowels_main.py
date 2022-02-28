import numpy as np
import pandas as pd
from neural import *
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


##################################################################################
# HANDLING THE VOWELS DATASET
##########################################
# store mapping of targets and target names
vowel = pd.read_csv('data_vowel.csv')
vowel.drop('Train_or_Test',axis='columns', inplace=True)
spk_no = {'Andrew':0, 'Bill':1, 'David':2, 'Mark':3, 'Jo':4, 'Kate':5, 'Penny':6, 
            'Rose':7, 'Mike':8, 'Nick':9, 'Rich':10, 'Tim':11,
                'Sarah':12, 'Sue':13, 'Wendy':14}
Sex = {'Male': 1, 'Female':0}
vowel["Sex"] = vowel.Sex.map(Sex)
vowel["Speaker_Number"] = vowel.Speaker_Number.map(spk_no)
# add the target labels and the feature names
test_data = vowel["Class"]
Y = []
Cls = ['hid', 'hId', 'hed', 'hEd', 'had', 'hAd', 'hod', 'hOd', 'hud', 'hUd', 'hYd']
for i in Cls:
    ls = [1 if row == i else 0 for row in test_data]
    Y.append(ls)

# setting X and y 
y_ = np.array(Y).T
X = vowel.iloc[:, :-1]
X_ =np.array(X)
x_train, x_test, y_train, y_test = train_test_split(X_, y_, test_size=0.25, random_state=12)
##################################################################################

m = x_train.shape[0]        # no. of training examples
n = x_train.shape[1]        # no. of features
c = y_train.shape[1]       # no. of classes
# print(x_train.shape, y_train.shape)
learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
# figure, axis = plt.subplots(4, 2)
total_cost = []
layers_dims = [n,7,7,c]
parameters, costs = L_layer_model(x_train.T, y_train.T, layers_dims, learning_rate=0.1, num_iterations=2500, print_cost=True, momentum=True)
pred_train, acc = predict(x_test.T, y_test.T, parameters)
# figure.suptitle('Cost against time for: different learning rates')
####### for loop to determine best laering rate
# layers_dims = [n,6,4,c]
# for i in learning_rates: 
#     parameters, costs = L_layer_model(x_train.T, y_train.T, layers_dims, learning_rate=i, num_iterations=2500, print_cost=False)
#     pred_train, acc = predict(x_test.T, y_test.T, parameters)
#     total_cost.append(costs)

# count = 0
# for i in range(4):
#     for j in range(2):
#         axis[i, j].plot(total_cost[count])
#         axis[i, j].set(ylabel='cost')
#         axis[i, j].set_title("Learning rate =" + str(learning_rates[count]))
#         count += 1
#         if count == 7:
#             break
# plt.show()
####### for loop to determine best no of nodes
# for i in range(1,9):
#     layers_dims = [n,i,i,c]
#     parameters, costs = L_layer_model(x_train.T, y_train.T, layers_dims, learning_rate=0.1, num_iterations=2500, print_cost=False)
#     pred_train, acc = predict(x_test.T, y_test.T, parameters)
#     total_cost.append(costs)

# count = 0
# for i in range(4):
#     for j in range(2):
#         axis[i, j].plot(total_cost[count])
#         axis[i, j].set(ylabel='cost')
#         axis[i, j].set_title("No of nodes =" + str(count+1))
#         count += 1
#         if count == 8:
#             break
# plt.show()


