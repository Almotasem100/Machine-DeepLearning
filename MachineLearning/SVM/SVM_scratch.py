import pandas as pd 
import numpy as np 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from statistics import mean
# store the data 
iris = load_iris()
# convert to DataFrame
df = pd.DataFrame(data=iris.data,
                  columns= iris.feature_names)
# store mapping of targets and target names
#   and selecting two classes to build the binary classifier
target_dict = dict(zip(set(iris.target), iris.target_names))
target_dict2 = {'setosa':-1, 'versicolor':1, 'virginica':2}
# add the target labels and the feature names
df["target"] = iris.target
df["target_names"] = df.target.map(target_dict)
df["target"] = df.target_names.map(target_dict2)

# setting X and y 
X = df.query("target_names == 'setosa' or target_names == 'versicolor'").loc[:, "petal length (cm)":"petal width (cm)"] 
y = df.query("target_names == 'setosa' or target_names == 'versicolor'").loc[:, "target"] 
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
x_train = pd.DataFrame.to_numpy(x_train)
x_test = pd.DataFrame.to_numpy(x_test)
y_train = pd.DataFrame.to_numpy(y_train)
y_test = pd.DataFrame.to_numpy(y_test)
#####################################################################

def init_parameters(X): 
    m, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    return w, b

def fit(X, y, learning_rate=0.001, lambd=0.01, n_iters=1000):
    w, b = init_parameters(X)
    for _ in range(n_iters): 
        for idx, x_i in enumerate(X): 
            condition = y[idx] * (np.dot(x_i, w) + b) >= 1
            if condition: 
                w -= learning_rate * (2 * lambd * w)
            else: 
                w -= learning_rate * (2 * lambd *  w - np.dot(x_i, y[idx]))
                b -= learning_rate * y[idx]
    return w, b

def predict(X, w, b):
    decision = np.dot(X, w) + b
    return np.sign(decision)
def accuaracy(y_true, y_pred):
    true_pred = np.zeros(y_true.shape)
    for i in range(len(y_true)):
        true_pred[i] = 1 if y_true[i] == y_pred[i] else 0
    return sum(true_pred)/len(true_pred)

weights, bias = fit(x_train, y_train)
y_pred = predict(x_test, weights, bias)
print('Model accuarcy: ', accuaracy(y_test, y_pred))
