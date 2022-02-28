import numpy as np


def computeCost(X, y, theta):
    m = X.shape[0]
    y_pred = X.dot(theta)
    J = (1/(2*m)) * np.sum((y_pred - y)**2)
    return J
# computeCost(X,y,theta)

def predict(X, theta):
    try:
        y_pred = X.dot(theta)
    except:
        X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        y_pred = X.dot(theta)
    return y_pred

def gradientDescent(X, y, theta, alpha=0.01, num_iter=100):
    j_his = [0 for i in range(num_iter)]
    m = X.shape[0]
    for i in range(num_iter):
        h = X.dot(theta)-y
        grad = (1/m)* np.matmul(X.T, h)
        theta -= alpha*grad
        j_his[i] = computeCost(X, y, theta)
    return theta, j_his

def EvaluatePerformance(X, y, theta):
    X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
    y_pred = predict(X, theta)
    accuracy = np.abs(y-y_pred)*100/y
    return (100-np.mean(accuracy))

def fit(X, y, alpha=0.01, num_iter=100):
    m, n = X.shape
    X = np.insert(X, 0, np.ones(m), axis=1)
    theta = np.zeros((n+1, 1))
    theta , j_his = gradientDescent(X, y, theta, alpha, num_iter)
    hypothesis = predict(X, theta)
    return hypothesis, theta




# data = np.loadtxt('univariateData.dat', delimiter=',')
# data = np.loadtxt('multivariateData.dat', delimiter=',') #For multivariate
# X = data[:, :-1]
# X = (X -np.mean(X))/np.std(X)
# y = data[:, -1].reshape(-1,1)
X = np.array([i for i in range(50)]).reshape(-1,1)
y = np.array([3*i-5 for i in range(50)]).reshape(-1,1)
# print(X[0:5], y[0:5])
h, theta = fit(X, y, 0.001, 100000)
accuracy = EvaluatePerformance(X[11:13], y[11:13], theta)
print("results of Multivariate data using random index:")
print('theta', theta)
print('y example:\n', y[11:13])
print('y prdection:\n', h[11:13])
print("accuarcy:", accuracy, '%')

