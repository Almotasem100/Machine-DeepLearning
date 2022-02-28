import numpy as np

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache
def relu(Z):
    A = np.maximum(0,Z)  
    cache = Z 
    return A, cache
def sigmoid_backward(dA, cache):
    Z = cache 
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ
def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

# function to calculate the forward propagation
# and apply the sigmoid or the relu computation
#on the activation layer elemnets
def linear_activation_forward(A_prev, W, b, activation):
    Z = np.dot(W, A_prev) + b
    linear_cache = (A_prev, W, b)
    
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        A, activation_cache = relu(Z)
    
    cache = (linear_cache, activation_cache)

    return A, cache

# function to calculate the bacjward propagation
# and apply the inverse of sigmoid or relu computation
#on the activation layer elemnets to calculate thier derivation
def linear_activation_backward(dA, cache, activation, momentum=False):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]
    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db

def init_momentum(layer_dims):
    parameters = {}
    L = len(layer_dims)          

    for l in range(1, L):
        parameters['W' + str(l)] = np.zeros((layer_dims[l], layer_dims[l - 1]))
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
   
    return parameters