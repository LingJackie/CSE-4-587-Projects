# -*- coding: utf-8 -*-
"""
Predicitve_Analytics.py
"""
import numpy as np



""" HELPERS """

""" for K Nearest Neigbor """
def EuclideanDistance(vect1, vect2):
    return np.sqrt(  np.sum( np.power(vect1 - vect2, 2) )  )










def Accuracy(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    
    """
    cm = ConfusionMatrix(y_true,y_pred)
    
    TN = []
    for i in range(0,cm[0].size):
        stuff = np.delete(cm, i, 0)
        stuff2 = np.delete(stuff, i, 1)
        TN.append(np.sum(stuff2))
    TN = np.asarray(TN)   
    
    
    TP = np.diag(cm)
    FN = np.sum(cm, axis=1) - TP
    FP = np.sum(cm, axis=0) - TP
    tmp = (TP + TN + FN + FP)
    return np.sum( (TP + TN)/ tmp) / cm[0].size
    
def Recall(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
    cm = ConfusionMatrix(y_true,y_pred)
    TP = np.diag(cm)
    FN = np.sum(cm, axis=1) - TP
    tmp = (TP + FN)
    return np.sum(TP / tmp) / cm[0].size

def Precision(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    
    PRECISION = TRUE POSITIVE/ (FALSE POSITIVE + TRUE POSITIVE)
    """
    
    
    cm = ConfusionMatrix(y_true,y_pred)
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    tmp = (TP + FP)
    return np.sum(TP / tmp) / cm[0].size

    """
    bools = np.equal(y_true, y_pred)
    count = bools.count(True)
    return float(count) / len(y_true)
    """
def WCSS(Clusters):
    """
    :Clusters List[numpy.ndarray]
    :rtype: float
    """
def ConfusionMatrix(y_true,y_pred):
    
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """  
    occurences = np.bincount(np.concatenate((y_true, y_pred), axis=0))

    uniqueClasses = np.nonzero(occurences)[0]

    numOfClasses = uniqueClasses.size
    
    tmp = y_true*numOfClasses + y_pred

    stuff = np.histogram(tmp, bins = np.arange(0, numOfClasses**2+1))
    
    return stuff[0].reshape(numOfClasses,numOfClasses)
def KNN(X_train,X_test,Y_train):
     """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: numpy.ndarray
    
    train with x_train and y_train
    """
    k = 5 
    distanceMetric = EuclideanDistance(X_train, Y_train)    
    for i in range(X_train.shape[0]):'''shape[0] is length of column '''
        
    
    
    
def RandomForest(X_train,Y_train,X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: numpy.ndarray
    """
    
def PCA(X_train,N):
    """
    :type X_train: numpy.ndarray                   
    :type N: int
    :rtype: numpy.ndarray
    """
    
    
    """ create an array of means of each column of x"""
    meanOfEachColumn = np.mean(X_train, axis=0)
    
    
    """ subtract each element of a column by that columns mean
    """
    zeroMeanMatrix = np.subtract(X_train, meanOfEachColumn)
     
    covariance = np.cov(zeroStandMatrix)
    
    
    """ finds eigen values and eigen vectors and sorts them largest to 
        smallest based on the eigen values
    """
    eigenVals, eigenVects = np.linalg.eig(covariance)
    idx = eigenVals.argsort()[::-1]   
    eigenVals = eigenVals[idx]
    """eigenVect is a matrix with each column being an eigen vector """
    eigenVects = eigenVects[:,idx]
    
    """grabs first N columns of the sorted eigen Vectors"""
    pComponents = eigenVects[:,:N]
    return np.dot(pComponents.T, X_train)
    
def Kmeans(X_train,N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: List[numpy.ndarray]
    """

def SklearnSupervisedLearning(X_train,Y_train,X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: List[numpy.ndarray] 
    """

def SklearnVotingClassifier(X_train,Y_train,X_test):
    
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: List[numpy.ndarray] 
    """


"""
Create your own custom functions for Matplotlib visualization of hyperparameter search. 
Make sure that plots are labeled and proper legends are used
"""



    
