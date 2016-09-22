# ﻿Code execution instructions

1. Edit 'dataset_details.py' between 'user parameter editing section started' and 'user parameter editing section ended' part. Here below parameters can be edited  

*train_path*: provide the training data URL or local file path  
*train_separato*r: provide the training file seperator  
*test_path*: provide the test data URL or local file path  
*test_separator*: provide the test file seperator
*maxdepth_parameters*: if accepts below two parameters
		-1 : default decision tree with the depth 1,2,4,...16 will build
		any positive integer: Decision tree with given depth size will build
*col_names*: give input dataset columns name inside list. Make sure that first column in dataset is target column with name 'class', else code won't work

2. Once all the inputs are given correctly, decision tree with the given depth size would developed.
3. By default decision tree will run on 'monks-1.train' and 'monks-1.test'. Respective output along with performance matrix will be printed on screen/console
