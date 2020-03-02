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
    


def Recall(y_true,y_pred):
     """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """

def Precision(y_true,y_pred):""" probably right didnt test """
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
    
    bools = np.equal(y_true, y_pred)
    count = bools.count(True)
    return float(count) / len(y_true)
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
    
    meanOfEachColumn = np.mean(X_train, axis=0)""" create an array of means of each column of x)"""
    zeroMeanMatrix = np.subtract(X_train, meanOfEachColumn)""" for each column: subtracts the mean of the column from each entry
                                                               This ensures that each column has a mean of zero"""
    standardOfEachColumn = np.std(zeroMeanMatrix, axis=0) """ create an array of standard deviation of each column of x)"""
    zeroStandMatrix = np.divide(zeroMeanMatrix, standardOfEachColumn) """each column has mean of zero and standard deviation of 1""" 
    covariance = np.multiply(zeroStandMatrix.T , zeroStandMatrix)                                                
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



    
