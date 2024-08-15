Introduction

In this assignment, I developed a Convolutional Neural Network (CNN) using Keras in R to classify the Fashion MNIST dataset. The aim is to create a CNN with six layers and make predictions on at least two images from the dataset.

Libraries and Data Preparation

 Load necessary libraries
library(keras)
library(tensorflow)

 Load Fashion MNIST dataset
fashion_mnist <- dataset_fashion_mnist()
c(c(x_train, y_train), c(x_test, y_test)) %<-% fashion_mnist
x_train <- x_train / 255
x_test <- x_test / 255

 Reshape the data
x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), 28, 28, 1))

 Convert class vectors to binary class matrices
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)
