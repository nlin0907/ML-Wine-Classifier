#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:02:37 2020

@author: nicolelin
"""

import numpy as np
from matplotlib import pyplot as plt

def euclidean_distance(a,b):
    diff = a - b
    return np.sqrt(np.dot(diff, diff))

def load_data(csv_filename):
    data = np.genfromtxt(csv_filename, delimiter=';', skip_header=1, usecols=(0,1,2,3,4,5,6,7,8,9,10,11))
    return data


# file = '/Users/nicolelin/Downloads/whitewine.csv'
# dataset = load_data(file)
# print(len(dataset))

    
def split_data(dataset, ratio = 0.9):
    test_ratio = int(ratio * dataset.shape[0])
    train = dataset[:test_ratio]
    # print(len(train))
    test = dataset[test_ratio:]
    # print(len(test))
    return (train, test)

# print(len(split_data(dataset,0.9)))


def compute_centroid(data):
    return sum(data) / len(data)
    
def experiment(ww_train, rw_train, ww_test, rw_test):
    total = 0
    correct = 0
    white_centroid = compute_centroid(ww_train)
    red_centroid = compute_centroid(rw_train)
    
    for row in ww_test:
        total+=1
        white_distance = euclidean_distance(row, white_centroid)
        red_distance = euclidean_distance(row, red_centroid)
        if white_distance < red_distance:
            correct +=1
            
    for row in rw_test:
        total+=1
        white_distance = euclidean_distance(row, white_centroid)
        red_distance = euclidean_distance(row, red_centroid)
        if red_distance < white_distance:
            correct +=1
    
    accuracy = correct/total

    print("Accuracy: {},  ({}/{} correct predictions)".format(accuracy, correct, total))
    return accuracy
    
def cross_validation(ww_data, rw_data, k):
    accuracy = 0.0
    size = ww_data.shape[0]/k
    for j in range(k):
        start = int(j * size)
        end = int(start + size + 1)
        ww_test = ww_data[start:end]
        ww_train1 = ww_data[:start]
        ww_train2 = ww_data[end:]
        ww_train = np.concatenate((ww_train1, ww_train2))
        
        rw_test = rw_data[start:end]
        rw_train1 = rw_data[:start]
        rw_train2 = rw_data[end:]
        rw_train = np.concatenate((rw_train1, rw_train2))
        
        accuracy += experiment(ww_train, rw_train, ww_test, rw_test)
    
    k_accuracy = accuracy/k
    return k_accuracy
        
    
    
if __name__ == "__main__":
    
    ww_data = load_data('/Users/nicolelin/Downloads/whitewine.csv')
    rw_data = load_data('/Users/nicolelin/Downloads/redwine.csv')

    #Uncomment the following lines for step 2: 
    ww_train, ww_test = split_data(ww_data, 0.9)
    rw_train, rw_test = split_data(rw_data, 0.9)
    print("testing experiment function")
    experiment(ww_train, rw_train, ww_test, rw_test)
    
    #Uncomment the following lines for step 3:
    k = 10
    print("testing cross-validation")
    acc = cross_validation(ww_data, rw_data,k)
    print("{}-fold cross-validation accuracy: {}".format(k,acc))
    
