# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 04:58:13 2019

@author: Shiv
"""

#Google_1

from sklearn import tree
textures=[[140, 0], [130, 0], [150, 1], [170, 1]]
label=["apple", "apple", "orange", "orange"]
clf=tree.DecisionTreeClassifier()   
clf=clf.fit(textures, label)
predict=clf.predict([[180, 1]])
print(predict)

#Google_2

import numpy as np
from sklearn import datasets
from sklearn import tree

iris=datasets.load_iris()
"""print(iris.feature_names)
print(iris.target_names)
for i in range(len(iris.data)):
    print(iris.data[i])
    print(iris.target[i])"""
test_idx=[0,50,100]

#training data
train_target=np.delete(iris.target, test_idx)
train_data=np.delete(iris.data, test_idx, axis = 0)

#test data
test_target=iris.target[test_idx]
test_data=iris.data[test_idx]

clf=tree.DecisionTreeClassifier()
clf=clf.fit(train_data, train_target)

predict=clf.predict(test_data)
print(test_target)
print(predict)

#Google_3

import numpy as np
import matplotlib.pyplot as plt

greyhound=500
labrador=500

grey_height=28+4*np.random.randn(greyhound)
lab_height=24+4*np.random.randn(labrador)

plt.hist([grey_height, lab_height], stacked=True, color=["r","b"])
plt.show()


#Google_4

from sklearn import datasets
iris=datasets.load_iris()

x=iris.data
y=iris.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

#Creating Classifier

"""from sklearn import tree
my_classifier=tree.DecisionTreeClassifier()"""

#KNNeighbour Classifier

from sklearn.neighbors import KNeighborsClassifier
my_classifier=KNeighborsClassifier()

my_classifier.fit(x_train, y_train)
predictions=my_classifier.predict(x_test)
print(predictions)


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))

#Google_5

import random
class ScrappyKNN():
    def fit(self, x_train, y_train):
        self.x_train=x_train
        self.y_train=y_train
    
    def predict(self, x_test):
        predictions=[]
        for row in x_test:
            label=random.choice(self.y_train)
            predictions.append(label)
        return predictions
    
from sklearn import datasets
iris=datasets.load_iris()

x=iris.data
y=iris.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

#Creating Classifier

"""from sklearn import tree
my_classifier=tree.DecisionTreeClassifier()"""

#KNNeighbour Classifier

#from sklearn.neighbors import KNeighborsClassifier
#my_classifier=KNeighborsClassifier()
my_classifier=ScrappyKNN()

my_classifier.fit(x_train, y_train)
predictions=my_classifier.predict(x_test)
print(predictions)


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))
