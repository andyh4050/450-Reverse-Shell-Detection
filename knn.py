import pandas as pd
import numpy as np
import collections
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from one_hot_encoder import Preprocsss

classifier = KNeighborsClassifier(n_neighbors=5)

def reduce_benign(x, y):
    ratio = 10
    y_neg = y[y == 0]
    y_pos = y[y == 1]
    x_neg = x[y == 0]
    x_pos = x[y == 1]
    a = np.random.permutation(y_neg.shape[0])
    test_len = int(y_neg.shape[0] / ratio)
    a = a[:test_len]

    x_neg = x_neg[a]
    y_neg = y_neg[a]

    x_res = np.concatenate((x_neg, x_pos), axis=0)
    y_res = np.concatenate((y_neg, y_pos))
    
    x_res = x_res[np.random.permutation(x_res.shape[0])]
    y_res = y_res[np.random.permutation(y_res.shape[0])]
    return x_res, y_res




preprocsss = Preprocsss()
X_train, X_val,X_test, y_train, y_val, y_test = preprocsss.split_train_val_test()

X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)

x_train_balance, y_train_balance = reduce_benign(X_train, y_train)
x_test_balance, y_test_balance = reduce_benign(X_test, y_test)



classifier.fit(X_train, y_train)
y = classifier.predict(X_test)


tn, fp, fn, tp = confusion_matrix(y_test,y).ravel()
precision = precision_score(y_test,y)
recall = recall_score(y_test,y)
f1 = (2 * precision * recall)/(recall + precision)
tpr = tp / (tp+fn)
fpr = fp / (fp+tn)
accuracy = accuracy_score(y_test,y)
print("tp: %.4f" % tp)
print("fn: %.4f" % fn)
print("tpr: %.4f" % tpr)
print('precision:%.4f'% precision)
print("recall: %.4f" % recall)
print("fpr: %.4f" %  fpr)
print('accuracy: %.4f'% accuracy)
print('f1 score: %.4f'% f1)
