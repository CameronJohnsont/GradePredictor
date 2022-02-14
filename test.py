import tensorflow
import keras

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

'''
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)'''

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print('Accuracy: \n', linear.score(x_test, y_test))
print('Coefficients: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
print()

predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print("Prediction: ", round(predictions[x]))
    print("Attributes: ", x_test[x])
    print("Actual Grade: ", y_test[x])
    print()

p = 'G1'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel("1st Period Grade")
pyplot.ylabel("Final Grade")
pyplot.show()

p2 = 'G2'
style.use("ggplot")
pyplot.scatter(data[p2], data["G3"])
pyplot.xlabel("2nd Period Grade")
pyplot.ylabel("Final Grade")
pyplot.show()

p3 = 'studytime'
style.use("ggplot")
pyplot.scatter(data[p3], data["G3"])
pyplot.xlabel("Study Time")
pyplot.ylabel("Final Grade")
pyplot.show()

p4 = 'failures'
style.use("ggplot")
pyplot.scatter(data[p4], data["G3"])
pyplot.xlabel("Failures")
pyplot.ylabel("Final Grade")
pyplot.show()

p5 = 'absences'
style.use("ggplot")
pyplot.scatter(data[p5], data["G3"])
pyplot.xlabel("Absences")
pyplot.ylabel("Final Grade")
pyplot.show()

