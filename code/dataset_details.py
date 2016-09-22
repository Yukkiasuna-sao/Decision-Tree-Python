import pandas as pd
from os import path
import math
import numpy as np


# read data sets
def dataset_read():
    # user parameter editing section started

    # make sure your class column is the first column of in input train and test file
    # Load training data and assign column names
    train_path = "http://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.train"
    # use below if you are reading train data from local file
    #train_path = path.abspath(path.curdir) + "your_file_name"
    train_separator = " "

    test_path = "http://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.test"
    # use below if you are reading test data from local file
    #test_path = path.abspath(path.curdir) + "your_file_name"

    test_separator = " "
    maxdepth_parameters = [-1]  # if given -1, default decision tree with the depth 1,2,4,...16 build, else given depth size tree build
    col_names = ["class", "a1", "a2", "a3", "a4", "a5",
                 "a6"]  # change the column name in this format. 'class' attribute should be first column of dataset with name 'class'
    # user parameter editing section ended. Don't edit anything further


    train = pd.read_table(train_path,
                          sep=train_separator, header=None)
    train = train.drop([0, 8],
                       1)  # put column number inside [ ],if you don't want to include columns in model building, else delete this line
    train.columns = col_names

    test = pd.read_table(test_path,
                         sep=test_separator, header=None)
    test = test.drop([0, 8],
                     1)  # put column number inside [ ],if you don't want to include columns in model building, else delete this line
    test.columns = col_names

    ## Convert the columns datatype into category
    for i in range(len(col_names)):
        train[col_names[i]] = train[col_names[i]].astype("category")
        test[col_names[i]] = test[col_names[i]].astype("category")
    return train, test, maxdepth_parameters
