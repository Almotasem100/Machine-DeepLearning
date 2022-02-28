#import the used libraries
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from neural import *

#import mnist dataset, reshape it and slicing it to improve computation speed
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
n1 = len(x_train)
n2 = len(x_test)
X_train = x_train.reshape((n1, -1)) # reshaping data into a vector
X_test = x_test.reshape((n2, -1))
X_train = X_train[:10000, :]
Y_train = y_train[:10000]
X_test = X_test[:1000, :]
Y_test = y_test[:1000]


# Binary classifier for a chosen digit
class_digit = 6             # Choosing the desired digit upon which the model will classify
Y_binary_tr = [1 if i==class_digit else 0 for i in Y_train]     #updating data output according to the digit
Y_binary_ts = [1 if i==class_digit else 0 for i in Y_test]
Y_binary_tr = np.array(Y_binary_tr).reshape(1, -1)
Y_binary_ts = np.array(Y_binary_ts).reshape(1, -1)
# Bin_classifier = KNeighborsClassifier()         #Building the model
# Bin_classifier.fit(X_train,Y_binary_tr)         #Training it....
# B_predicted = Bin_classifier.predict(X_test)          #Testing it....
layers_dims = [784, 300, 150, 60, 10, 1]
parameters = L_layer_model(X_train.T, Y_binary_tr, layers_dims, num_iterations=2500, print_cost=True)
pred_train = predict(X_test.T, Y_binary_ts, parameters)
# print("Binary model Accuarcy", accuracy_score(B_predicted, Y_binary_ts))
# print("The Confusion matrix\n", confusion_matrix(B_predicted, Y_binary_ts))

#Manual test for the model
B_index = 514     #select a random example.
print("The model predicts the example to be ", str(class_digit), "is: ", pred_train[:, 50:100]) # print the model prediction
print("The example real value: ", Y_test[50:100])                # print the real output
# plt.imshow(x_test[index,:], cmap='Greys')      #plot the example image
# plt.show()



# Multi-classifier model
# Mul_classifier = KNeighborsClassifier()         #Building the model
# Mul_classifier.fit(X_train,Y_train)         #Training it....
# M_predicted = Mul_classifier.predict(X_test)          #Testing it....
# print("Multi_class. model Accuarcy", accuracy_score(M_predicted, Y_test))
# # print("The Confusion matrix\n", confusion_matrix(M_predicted, Y_test))
# print("\n")
# #Manual test for the model
# B_index = 713     #select a random example.
# print("The model predicts the example to be: ", B_predicted[B_index]) # print the model prediction
# print("The example real value: ", Y_binary_ts[B_index])                # print the real output
# # plt.imshow(x_test[index,:], cmap='Greys')      #plot the example image
# # plt.show()

