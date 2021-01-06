# ML-Wine-Classifier
Building a machine learning classifier that can predict whether a Portuguese (Vinho Verde) wine is red or white from the chemical measurements: fixed acidity, volatile acidity, citric acid, residual sugar chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol

Overview:
In this project, I will apply the Nearest Centroid Classifier and a technique called cross-validation to get a more reliable measure of the performance of the machine learning model.

The Data Set:
Each row in the redwine.csv and whitewine.csv files contains information about a type of wine (these are all Portuguese wines of the Vinho Verde DOC), either red or white wine. The attributes for each wine are different chemical measurements. The columns in the files are:

fixed acidity
volatile acidity
citric acid
residual sugar
chlorides
free sulfur dioxide
total sulfur dioxide
density
pH
sulphates
alcohol
quality (score between 1 and 10. We won't be using this column)
Note that the fields are separated by semicolons. 


Part 1 - Reading and Preprocessing the Data

First, write the function load_data(csv_filename), which should read in wine data from a file. This function should return a numpy ndarray in which each row represents a wine and each column represents a chemical measurement. The last column ("quality") in the csv file is not a chemical measurement and should not be part of the ndarray. The ndarray will therefore only have 11 columns.

Next, write the function split_data(dataset, ratio), which splits the data set into a training and testing portion. The function should return a (training set, testing set) tuple of ndarrays. The ratio parameter determines how much of the data should be assigned to training and how much should be used for testing. For example, if ratio == 0.9, the first 90% of the data should be placed in the training set. Make sure that the training and testing set do not overlap.


Part 2 - Nearest Centroid Classifier

Write the function experiment(ww_training, rw_training, ww_test, rw_test), which reads in training and test data sets and performs the following steps:

1) Create a centroid for each class (red or white) using the compute_centroid function.
2) For each of the items in the two test, make a prediction for the color by measuring the euclidean distance of this data item to each of the centroids. The template already contains a euclidean_distance method you can use.
3) Keep track of how many correct predictions this model makes. Know the correct color of each data item from the data set that it is in. 
4) The function should print out the total number of predictions made, the number of correct predictions, and the accuracy of the model. 
5) The function should also return the accuracy. 


Part 3 - Cross Validation

When working with relatively small testing sets results may not be reliable. For example, it is possible that, by chance, a small testing set contains only input instances that are difficult to predict, or only instances that are easy to predict. Selecting a different test data set of the same size might lead to different results.

One common solution to this problem is to split the available data into k-partitions of equal size; perform k training/testing repetitions. In each repetition, test on one of the k partitions after training on the remaining k-1 partitions. In this way, each of the partitions will be tested on once; then, average the accuracy for each of the k repetitions. This is called k-fold cross validation. 

Implement the function cross_validation(ww_data, rw_data, k), that performs k-fold cross validation and returns the average accuracy. Use the experiment function from step 2 to run each training/testing repetition. 
