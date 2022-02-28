import numpy as np
import pandas as pd
from neural import *
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


##################################################################################
# HANDLING THE IRIS DATASET
##########################################
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns= iris.feature_names)
# store mapping of targets and target names
target_dict = dict(zip(set(iris.target), iris.target_names))
target1 = {'setosa':1, 'versicolor':0, 'virginica':0}
target2 = {'setosa':0, 'versicolor':1, 'virginica':0}
target3 = {'setosa':0, 'versicolor':0, 'virginica':1}
# add the target labels and the feature names
df["target"] = iris.target
df["target_names"] = df.target.map(target_dict)
df["target1"] = df.target_names.map(target1)
df["target2"] = df.target_names.map(target2)
df["target3"] = df.target_names.map(target3)
df.drop('target_names',axis='columns', inplace=True)
df.drop('target',axis='columns', inplace=True)

# setting X and y 
X = df.iloc[:, :-3] 
y = df.iloc[:, -3:] 
X_ = pd.DataFrame.to_numpy(X)
y_ = pd.DataFrame.to_numpy(y)
x_train, x_test, y_train, y_test = train_test_split(X_, y_, test_size=0.3, random_state=10)
##################################################################################

m = x_train.shape[0]        # no. of training examples
n = x_train.shape[1]        # no. of features
c = y_train.shape[1]       # no. of classes
# print(x_train.shape, y_train.shape)
learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
total_cost = []
layers_dims = [n,5,c]
parameters, costs = L_layer_model(x_train.T, y_train.T, layers_dims, learning_rate=0.05, num_iterations=2500, print_cost=True, momentum=True)
pred_train, acc = predict(x_test.T, y_test.T, parameters)
# figure, axis = plt.subplots(4, 2)
####### for loop to determine best laering rate
# layers_dims = [n,4,c]
# figure.suptitle('Cost against time for: different learning rates')
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
#     layers_dims = [n,i,c]
#     parameters, costs = L_layer_model(x_train.T, y_train.T, layers_dims, learning_rate=0.05, num_iterations=2500, print_cost=False)
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