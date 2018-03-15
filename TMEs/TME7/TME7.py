# -*- coding: utf-8 -*-

import numpy as np
import sklearn.naive_bayes as nb
from sklearn import svm
from sklearn import linear_model as lin


# données ultra basiques, à remplacer par vos corpus vectorisés
X = np.array([[0., 0.], [1., 1.]])
y = np.array([0, 1])

# SVM
clf = svm.LinearSVC()
# Naive Bayes
clf = nb.MultinomialNB()
# regression logistique
clf = lin.LogisticRegression()

# apprentissage
clf.fit(X, y)  
clf.predict([[2., 2.]]) # usage sur une nouvelle donnée
