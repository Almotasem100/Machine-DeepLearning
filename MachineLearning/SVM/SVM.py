#importing needed libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from statistics import mean

#reading the data from the txt file and creating the model
svclassifier = SVC(kernel='linear')
df = pd.read_csv('data.txt', sep=" ", header=None)
x = df.loc[:, 0:23].copy()
y = df.loc[:, 24].copy()


# 1) PART ONE (WITH RAW DATA)
# looping 10 times with different split 
#     and calculating the accuaracy:
model1_accuaracy = []
for i in range(10):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=i)
    svclassifier.fit(x_train, y_train)
    y_pred1 = svclassifier.predict(x_test)
    model1_accuaracy.append(accuracy_score(y_test, y_pred1))
Avg_acc1 = mean(model1_accuaracy)
print('Accuarcy array of model:', model1_accuaracy)
print('Average accuarcy of model using Raw Data:', Avg_acc1)

# 2) PART TWO (WITH PROCESSED DATA)
# Processing data as instructed
for col in x:
    col_mean = mean(x.loc[:, col])
    i=0
    for row in x.loc[:, col]:
        x.loc[i, col] = row-col_mean
        i += 1
for col in x:
    col_max = x.loc[:, col].max()
    i=0
    for row in x.loc[:, col]:
        x.loc[i, col] = row/col_max
        i += 1
# looping 10 times with different split 
#     and calculating the accuaracy:
model2_accuaracy = []
for i in range(10):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=i)
    svclassifier.fit(x_train, y_train)
    y_pred2 = svclassifier.predict(x_test)
    model2_accuaracy.append(accuracy_score(y_test, y_pred2))
Avg_acc2 = mean(model2_accuaracy)
print('Accuarcy array of model:', model2_accuaracy)
print('Average accuarcy of the model using Processed Data:', Avg_acc2)
