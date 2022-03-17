import numpy as np
from collections import Counter
from scipy import stats
from math import log2, sqrt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

def entropy(class_y):
    """ 
    Input: 
        - class_y: list of class labels (0's and 1's)
    

    Example: entropy([0,0,0,1,1,1,1,1]) = 0.9544
    """
    
    p1 = np.sum(np.array(class_y)==0)/len(class_y)
    p2 = np.sum(np.array(class_y)==1)/len(class_y)
    result = (-1 * p1*np.log2(p1) - p2 * np.log2(p2))
    return result

def information_gain(previous_y, current_y):
    """
    Inputs:
        - previous_y : the distribution of original labels (0's and 1's)
        - current_y  : the distribution of labels after splitting based on a particular
                     split attribute and split value
    
    Reference: http://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15381-s06/www/DTs.pdf 

    Example: previous_y = [0,0,0,1,1,1], current_y = [[0,0], [1,1,1,0]], info_gain = 0.4591
    """ 
    s = entropy(previous_y)
    i = current_y[0]
    if np.sum(np.array(i)==0)/len(i) != 1 and np.sum(np.array(i)==1)/len(i) != 1 and len(i) != 0:
        s -= len(i)/len(previous_y) * entropy(i)
    else:
        s-= 0 
    i = current_y[1]
    if np.sum(np.array(i)==0)/len(i) != 1 and np.sum(np.array(i)==1)/len(i) != 1 and len(i) != 0:
        s -= len(i)/len(previous_y) * entropy(i)
    else:
        s-= 0 
    return s
    
def partition_classes(X, y, split_attribute, split_val):
    """
    Inputs:
    - X               : (N,D) list containing all data attributes
    - y               : a list of labels
    - split_attribute : column index of the attribute to split on
    - split_val       : either a numerical or categorical value to divide the split_attribute
    
               
    Return in this order: (X_left, X_right, y_left, y_right)           
    """
    X_data = np.array(X, dtype= 'object')
    y_data = np.array(y)
    if str(X_data[0,split_attribute]).isdigit():
        s = X_data[:,split_attribute].astype('float64')
        X_left = X_data[s <= split_val ]
        X_right = X_data[s > split_val ]
        y_left = y_data[s <= split_val ]
        y_right = y_data[s > split_val ]
    else:
        X_left = X_data[X_data[:,split_attribute] == split_val ]
        X_right = X_data[X_data[:,split_attribute] != split_val ]
        y_left = y_data[X_data[:,split_attribute] == split_val ]
        y_right = y_data[X_data[:,split_attribute] != split_val ]
    # Delete this line when you implement the function
    return X_left.tolist(), X_right.tolist(), y_left.tolist(), y_right.tolist()

def find_best_split(X, y, split_attribute):
    """Inputs:
        - X               : (N,D) list containing all data attributes
        - y               : a list array of labels
        - split_attribute : Column of X on which to split
    
    """
    x = np.array(X, dtype = 'object')
    value = np.unique(x[:,split_attribute])
    info_gain = 0
    best_split_val = 0
    for j in value:
        X_left, X_right, y_left, y_right = partition_classes(X, y, split_attribute, j)
        current_y = [y_left,y_right]
        s = information_gain(y, current_y)
        if s > info_gain:
            best_split_val = j
            info_gain = s
        #print('split_val =', j,  '-->  info_gain =', s)
    return best_split_val, info_gain

def find_best_feature(X, y):
    """
    Inputs:
        - X: (N,D) list containing all data attributes
        - y : a list of labels
    
    """
    x = np.array(X)
    best_split_feature = 0
    info_gain = 0
    best_split_val = 0
    #print(x.shape)
    for i in range(x.shape[1]):
        split_val, info_gains = find_best_split(X, y, i)
        if info_gains > info_gain:
            best_split_feature = i
            best_split_val = split_val
    return best_split_feature, best_split_val

'''
def pruning(dt, X, y):
    """
    """
    dt.fit(X,y, 0)
    if dt.tree['isleaf'] == False:
        defind_tree(dt)
    return dt

def defind_tree(dt):
    if dt.tree['isleaf'] == True:
        return
    else:
        dt_new = dt
        if dt_new.tree['left'].tree['isleaf'] == True and dt_new.tree['right'].tree['isleaf'] == True: 
            class_labels = dt_new.tree['left'].tree['y']
            class_labels.extend(dt_new.tree['right'].tree['y'])
            x = dt_new.tree['left'].tree['x']
            x.extend(dt_new.tree['right'].tree['x'])
            loss_old = DecisionTreeEvalution(dt_new, x, class_labels, False)
            loss_new = DecisionTreeError(class_labels)
            if loss_new <= loss_old:
                dt_new.tree['class_label'] = np.argmax(np.bincount(np.array(class_labels)))
                dt_new.tree['isleaf'] = True
                defind_tree(dt_new)
            else:
                defind_tree(dt)
        else:
            defind_tree(dt_new.tree['left'])
            defind_tree(dt_new.tree['right'])
'''

                  
def DecisionTreeError(y):
    # helper function for calculating the error of the entire subtree if converted to a leaf with majority class label.
    # You don't have to modify it 
    num_ones = np.sum(y)
    num_zeros = len(y) - num_ones
    return 1.0 - max(num_ones, num_zeros) / float(len(y))



class MyDecisionTree(object):
    def __init__(self, max_depth=None):
        """
        """
        self.tree = {}
        if max_depth == None:
            max_depth = np.inf
        self.max_depth = max_depth
        self.old = None
        
    def find_Best_feature(self, X, y):
        x = np.array(X)
        best_split_feature = 0
        info_gain = 0
        best_split_val = 0
        #print(x.shape)
        for i in range(x.shape[1]):
            if str(x[0,i]).isdigit():
                s = x[:,i].astype('float64')
                split_val = np.mean(s)
                X_left, X_right, y_left, y_right = partition_classes(X, y, i, split_val)
                current_y = [y_left,y_right]
                info_gains = information_gain(y, current_y)
                split_val1 = np.median(s)
                X_left, X_right, y_left, y_right = partition_classes(X, y, i, split_val1)
                current_y = [y_left,y_right]
                if information_gain(y, current_y) > info_gains:
                    split_val = split_val1
            else:
                unique,pos = np.unique(x[:,i],return_inverse=True)
                counts = np.bincount(pos)
                maxpos = counts.argmax() 
                split_val = unique[maxpos]
            X_left, X_right, y_left, y_right = partition_classes(X, y, i, split_val)
            current_y = [y_left,y_right]
            info_gains = information_gain(y, current_y)
            if info_gains > info_gain:
                best_split_feature = i
                best_split_val = split_val
        return best_split_feature, best_split_val
        

    def fit(self, X, y, depth):
        """
        """
        if len(X)!= 0:
            #print(info_gain)
            if ( depth == self.max_depth or entropy(y) < 0.01):
                self.tree['isleaf'] = True
                self.tree['class_label'] = np.argmax(np.bincount(np.array(y)))
                self.old = np.argmax(np.bincount(np.array(y)))
                self.tree['x'] = X
                self.tree['y'] = y
                #self.tree['class_label'] = y
            else:
                best_split_feature, best_split_val = self.find_Best_feature(X, y)
                X_left, X_right, y_left, y_right = partition_classes(X, y, best_split_feature, best_split_val)
                #for i in self.tree:
                self.tree['isleaf'] = False
                self.tree['split_feature'] = best_split_feature
                self.tree['split_value'] = best_split_val
                dl = MyDecisionTree(self.max_depth)
                dr = MyDecisionTree(self.max_depth)
                dl.fit(X_left, y_left, depth + 1)
                self.tree['left'] = dl # another object of MyDecisionTree class
                #print('l')
                dr.fit(X_right, y_right, depth + 1)
                self.tree['right'] = dr  # another object of MyDecisionTree class
                #print('r')
        else:
            if self.old == 1:
                self.tree['class_label'] = 0
            else:
                self.tree['class_label'] = 1
            self.tree['isleaf'] = True
            self.tree['x'] = list()
            self.tree['y'] = list()


    def predict(self, record):
        """
        """
        if self.tree['isleaf'] == True:
            return self.tree['class_label']
        else:
            if str(record[self.tree['split_feature']]).isdigit():
                if (record[self.tree['split_feature']] <= self.tree['split_value']):
                    return (self.tree['left']).predict(record)
                else:
                    return (self.tree['right']).predict(record)
            else:
                if (record[self.tree['split_feature']] == self.tree['split_value']):
                    return (self.tree['left']).predict(record)
                else:
                    return (self.tree['right']).predict(record)
                    

"""
Note: 
For undergraduate students, you can import the DecisionTreeClassifier from sklearn so that even if you can not build a decision tree sucessfully, 
you could still finish the random forest classifer. 10 bonus points will be given for undergraduate students using your own decision tree MyDecisionTree()
to finish random forest.

"""
class RandomForest(object):
    def __init__(self, n_estimators=50, max_depth=None, max_features=0.7):
        # helper function. You don't have to modify it
        # Initialization done here
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstraps_row_indices = []
        self.feature_indices = []
        self.out_of_bag = []    
        #self.decision_trees = [MyDecisionTree(max_depth=max_depth) for i in range(n_estimators)]
        self.decision_trees = [DecisionTreeClassifier(max_depth=max_depth) for i in range(n_estimators)]
        
    def _bootstrapping(self, num_training, num_features):
        """
        """
        row_idx = np.random.choice(num_training, num_training, replace=True)
        col_idx = np.random.choice(num_features, int(self.max_features * num_features), replace= False)
        return row_idx, col_idx
            
    def bootstrapping(self, num_training,num_features):
        # helper function. You don't have to modify it
        # Initializing the bootstap datasets for each tree
        for i in range(self.n_estimators):
            total = set(list(range(num_training)))
            row_idx, col_idx = self._bootstrapping(num_training, num_features)
            total = total - set(row_idx)
            self.bootstraps_row_indices.append(row_idx)
            self.feature_indices.append(col_idx)
            self.out_of_bag.append(total)

    def fit(self, X, y):
        """
        """
        # Delete this line when you implement the function
        self.bootstrapping(X.shape[0],X.shape[1])
        for i in range(self.n_estimators):
            X_train = X[self.bootstraps_row_indices[i],:]
            X_train = X_train[:,self.feature_indices[i]]
            y_train = y[self.bootstraps_row_indices[i]]
            #self.decision_trees[i].fit(X_train, y_train, 0)
            self.decision_trees[i].fit(X_train, y_train)
        #raise NotImplementedError

    def predict(self,X):
        prediction = []
        for i in range(len(X)):
            predictions = []
            for t in range(self.n_estimators):
                if i in self.out_of_bag[t]:
                    #predictions.append(self.decision_trees[t].predict((X[i][self.feature_indices[t]].reshape(1, -1)[0])))
                    predictions.append(self.decision_trees[t].predict(X[i][self.feature_indices[t]].reshape(1, -1)))
            if predictions.count(1) >= predictions.count(0):
                prediction.append(1)
            else:
                prediction.append(0)
        return prediction
    
    def OOB_score(self, X, y):
        # helper function. You don't have to modify it
        print(y)
        accuracy = []
        tp = []
        fp = []
        post_ind = (np.array(y) == 1)
        fal_ind = (np.array(y) == 0)
        for i in range(len(X)):
            predictions = []
            for t in range(self.n_estimators):
                if i in self.out_of_bag[t]:
                    #predictions.append(self.decision_trees[t].predict((X[i][self.feature_indices[t]].reshape(1, -1)[0])))
                    predictions.append(self.decision_trees[t].predict(X[i][self.feature_indices[t]].reshape(1, -1)))
            if len(predictions) > 0:
                accuracy.append(np.sum(predictions == y[i]) / float(len(predictions)))
                #print(predictions)
                #temp = predictions == y[i]
                if y[i] == 1:
                    tp.append(np.sum(predictions == y[i]) / float(len(np.array(y)[post_ind])))
                if y[i] == 0:
                    fp.append(np.sum(predictions != y[i]) / float(len(np.array(y)[fal_ind])))
        return np.mean(accuracy), np.mean(tp), np.mean(fp)


