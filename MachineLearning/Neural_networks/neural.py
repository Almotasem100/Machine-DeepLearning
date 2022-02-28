import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from help_func import *



# function to random initialize the parameters
# by looping on every layer dimensions
# and return a dictionary which holds them 
def init_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims)          

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
   
    return parameters

# function to create the neural network and it's layers
# and to apply the forwrd propagation upon them
def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation='relu')
        caches.append(cache)

    Al, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation='sigmoid')
    caches.append(cache)

    return Al, caches

# function to compute the cost with each iteration
# by applying the gradient descent algorithm and momentum optimization
def compute_cost(Al, Y):
    m = Y.shape[1]
    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(Al)) + np.multiply(1 - Y, np.log(1 - Al)))
    cost = np.squeeze(cost) 
    return cost

# function to update the neural network and it's layers
# by applying the backward propagation upon them
def L_model_backward(Al, Y, caches):
    grads = {}
    L = len(caches)
    # m = Al.shape[1]
    Y = Y.reshape(Al.shape)
    dAl = - (np.divide(Y, Al) - np.divide(1 - Y, 1 - Al))
    current_cache = caches[-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAl, current_cache, activation = "sigmoid")
    # print(grads)
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+2)], current_cache, activation = "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

# function to update the parameters according to the 
# gradient values calculated 
def update_parameters(parameters, grads, learning_rate, momen_paramet, momentum):
    L = len(parameters) // 2
    if momentum:
        gamma = 0.9
        for l in range(L):
            momen_paramet["W" + str(l + 1)] = gamma*momen_paramet["W" + str(l + 1)] + (1-gamma)*grads["dW" + str(l + 1)]
            momen_paramet["b" + str(l + 1)] = gamma*momen_paramet["b" + str(l + 1)] + (1-gamma)*grads["db" + str(l + 1)]
            parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * momen_paramet["W" + str(l + 1)]
            parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * momen_paramet["b" + str(l + 1)]
    else:
        for l in range(L):
            parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
            parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters, momen_paramet

def L_layer_model(x_train, y_train, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False, momentum=False): #lr was 0.009
    # np.random.seed(1)
    costs = []
    parameters = init_parameters(layers_dims)
    momen_paramet = init_momentum(layers_dims)
    for i in range(0, num_iterations):
        X, _, Y, _ = train_test_split(x_train.T, y_train.T, test_size=0.1)
        X = X.T
        Y = Y.T
        Al, caches = L_model_forward(X, parameters)
        cost = compute_cost(Al, Y)
        grads = L_model_backward(Al, Y, caches)
        parameters, momen_paramet = update_parameters(parameters, grads, learning_rate, momen_paramet, momentum)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
        if i % 100 == 0:
            costs.append(cost)
    if print_cost:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    return parameters, costs

def predict(X, y, parameters):
    p = np.zeros(y.shape, dtype = np.int)
    a_l, _ = L_model_forward(X, parameters)
    for i in range(a_l.shape[1]):
        for j in range(a_l.shape[0]):
            if a_l[j,i] > 0.5:
                p[j,i] = 1
            else:
                p[j,i] = 0
    accuarcy = np.mean((p == y))
    print("Accuracy: "  + str(accuarcy))
    return p, accuarcy

