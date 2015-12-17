"""
Methods for loading the data sets as numpy matrices.
"""

import numpy as np
import random


def load_iris(training_ratio):
    """Load the iris data.

    Args:
        training_ratio: the ratio of examples that go into the training set
    Returns:
        a tuple of numpy matrices, the first in the tuple is the training 
        data, second is test data. Each matrix row represents a data point 
        as a row vector: the first element of the row vector corresponds to 
        the label and the following elements correspond to attributes.
    """
    random.seed(1) # get same data every time
    iris_labels = {'Iris-setosa': 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2}
    f = open('data/iris.data', 'r')

    training = None
    test = None
    lines = f.readlines()
    train_index = int(len(lines)*training_ratio)
    random.shuffle(lines)
    for k, line in enumerate(lines):
        data = line.split(',')
        vector = [int(iris_labels[data[-1].rstrip('\n')])]+ [float(i) for i in data[0:-1]]
        vector = np.array(vector)
        if k < train_index:
            if training is None:
                training = vector
            else:
                training = np.vstack((training, vector))
        else:
            if test is None:
                test = vector
            else:
               test = np.vstack((test, vector))
    return (training, test)

def load_congress_data(training_ratio):
    """Load the congress data.

    Note that missing values (denoted '?', where a voter abstained) are 
    instead treated as a third type of attribute. Therefore every feature
    has 3 possible attributes.

    Args:
        training_ratio: the ratio of examples that go into the training set
    Returns:
        a tuple of numpy matrices, the first in the tuple is the training 
        data, second is test data. Each matrix row represents a data point 
        as a row vector: the first element of the row vector corresponds to 
        the label and the following elements correspond to attributes.
    """
    random.seed(1) # get same data every time
    label_conversions = {'republican' : 0, 'democrat' : 1, 
                         'n' : 0, 'y' : 1, '?' : 2} 
    f = open('data/house-votes-84.data', 'r')

    training = None
    test = None
    lines = f.readlines()
    train_index = int(len(lines)*training_ratio)
    random.shuffle(lines)
    for k, line in enumerate(lines):
        data = line.split(',')
        vector = [float(label_conversions[i.rstrip('\n')]) for i in data]
        vector = np.array(vector)
        if k < train_index:
            if training is None:
                training = vector
            else:
                training = np.vstack((training, vector))
        else:
            if test is None:
                test = vector
            else:
               test = np.vstack((test, vector))
    return (training, test)

def load_monks(data_set_number):
    """Load the MONK's data.

    Args:
        data_set_number: which MONK's data set to use (1, 2, or 3)
    Returns:
        a tuple of numpy matrices, the first in the tuple is the training 
        data, second is test data. Each matrix row represents a data point 
        as a row vector: the first element of the row vector corresponds to 
        the label and the following elements correspond to attributes.
    """

    if data_set_number not in [1, 2, 3]:
        print "Invalid argument"
        return

    s = 'data/monks-'+str(data_set_number)+'.train'
    f = open(s, 'r')
    training = None
    for line in f.readlines():
        data = line.split(' ')
        vector = [int(i) for i in data[1:-1]]
        vector = np.array(vector)
        if training is None:
            training = vector
        else:
            training = np.vstack((training, vector))

    s = 'data/monks-'+str(data_set_number)+'.test'
    f = open(s, 'r')
    test = None
    for line in f.readlines():
        data = line.split(' ')
        vector = [int(i) for i in data[1:-1]]
        vector = np.array(vector)
        if test is None:
            test = vector
        else:
            test = np.vstack((test, vector))
    
    return (training, test)

# just to test the methods 
if __name__=="__main__":
    load_iris(.6)
    load_congress_data(.6)
    for i in [1, 2, 3]:
        load_monks(i)
