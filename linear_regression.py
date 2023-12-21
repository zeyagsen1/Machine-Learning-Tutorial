import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import os
import tensorflow
import keras

data=pd.read_csv('student-mat.csv',sep=';')#we are using panda to read the csv dataset.
data=data[["G1","G2","G3","studytime","absences","failures"]]

predict="G3"
x=np.array(data.drop([predict],axis=1)) #  drops the predict value from the data.(axis=1 says column axis=0 says rows)
##we use numpy array
print(x)

y=np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection(x, y, test_size=0.1)
    linear = LinearRegression()  # Corrected instantiation heretrain_test_split
    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)
    print(accuracy)

    if accuracy > best:
        best = accuracy
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)


    # Load the model
with open("studentmodel.pickle", "rb") as pickle_in:
    linear = pickle.load(pickle_in)



print('Coefficient ',linear.coef_)
print('Intercept ',linear.intercept_)
predictions=linear.predict(x_test)

for x in range (len(predictions)):
    print(predictions[x],x_test[x],y_test[x])
p="G1"
style.use("ggplot")
plt.scatter(data[p],data["G3"])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()