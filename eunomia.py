import tensorflow as tf
import numpy as np
from src.Utilities import Utilities
from src.Autoencoder import InputLayer
from src.Autoencoder import HiddenLayer
from src.Autoencoder import OutputLayer

numSamples = 1000
numFeatures = 1000000
numEpochs = 100
batchSize = 10000

import random
inputArray = []
for i in range(numSamples // 2):
    Utilities.progress(i + 1, numSamples, status='Building sample ' + str(i + 1))
    inputInternal = []
    for j in range(numFeatures):
        inputInternal.append(random.uniform(0.0, 0.1))
    inputArray.append(inputInternal)
for i in range(numSamples // 2):
    Utilities.progress(i + (numSamples / 2) + 1, numSamples, 
                            status='Building sample ' + str(i + (numSamples / 2) + 1))
    inputInternal = []
    for j in range(numFeatures):
        inputInternal.append(random.uniform(0.9, 1.0))
    inputArray.append(inputInternal)
random.shuffle(inputArray)

# Read in data from csv file
# Utilities.progress(1, 7, status='Reading in data        ')
# inputArray = Utilities.readData()

# Build input layer
Utilities.progress(2, 7, status='Building input layer   ')
with tf.variable_scope("input"):
    iLayer = InputLayer(len(inputArray[1]))

# Build hidden layer 1
Utilities.progress(3, 7, status='Building hidden layer 1')
with tf.variable_scope("hidden1"):
    hidden1 = HiddenLayer(100000, iLayer.inputLayer)
    hidden1.buildTrainer()

# Build hidden layer 2
Utilities.progress(4, 7, status='Building hidden layer 2')
with tf.variable_scope("hidden2"):
    hidden2 = HiddenLayer(5000, hidden1.y1)
    hidden2.buildTrainer()

# Build hidden layer 3
Utilities.progress(5, 7, status='Building hidden layer 3')
with tf.variable_scope("hidden3"):
    hidden3 = HiddenLayer(160, hidden2.y1)
    hidden3.buildTrainer()

# Build output layer
Utilities.progress(6, 7, status='Building output layer  ')
with tf.variable_scope("output"):
    oLayer = OutputLayer(2, hidden3.y1)
    oLayer.buildTrainer()

Utilities.progress(7, 7, status='Starting session       ')
sess = Utilities.startSession()

iLayer.printLayerShape()
hidden1.printLayerShape()
hidden2.printLayerShape()
hidden3.printLayerShape()
oLayer.printLayerShape()

# Training the hidden layers and output layer on the data.
for i in range(numEpochs):
    Utilities.progress(i + 1, numEpochs, status='Training Layer 1 ')
    sess.run(hidden1.trainStep, 
             feed_dict = {iLayer.inputLayer: Utilities.batchBuilder(inputArray, batchSize)})

for i in range(numEpochs):
    Utilities.progress(i + 1, numEpochs, status='Training Layer 2 ')
    sess.run(hidden2.trainStep, 
             feed_dict = {iLayer.inputLayer: Utilities.batchBuilder(inputArray, batchSize)})

for i in range(numEpochs):
    Utilities.progress(i + 1, numEpochs, status='Training Layer 3 ')
    sess.run(hidden3.trainStep, 
             feed_dict = {iLayer.inputLayer: Utilities.batchBuilder(inputArray, batchSize)})

for i in range(numEpochs):
    Utilities.progress(i + 1, numEpochs, status='Training Ouput Layer')
    for j in range(len(inputArray)):
        if inputArray[j][0] > 0.5:
            labels = [1.0, 0.0]
        else:
            labels = [0.0, 1.0]
        sess.run(oLayer.trainStep,
                 feed_dict = {iLayer.inputLayer: Utilities.numpyReshape(inputArray[j]), 
                              oLayer.labelTensor: Utilities.numpyReshape(labels)})

# Gathers the results for analysis
outputList = []
for i in range(len(inputArray)):
    Utilities.progress(i + 1, len(inputArray), status='Gathering Output')
    outputList.append(sess.run(oLayer.yo, 
                      feed_dict = {iLayer.inputLayer: Utilities.numpyReshape(inputArray[i])}))

num1 = 0
num2 = 0

for i in outputList:
    if i[0][0] > i[0][1]:
        num1 += 1
    else:
        num2 += 1

print("\nNumber of 1: ", num1)
print("Number of 2: ", num2)

print("\nHidden Layer 1:")
print("Squared Difference: ", sess.run(hidden1.squareDifference, 
                              feed_dict = {iLayer.inputLayer: Utilities.numpyReshape(inputArray[0])}))
print("\nHidden Layer 2:")
print("Squared Difference: ", sess.run(hidden2.squareDifference, 
                              feed_dict = {iLayer.inputLayer: Utilities.numpyReshape(inputArray[0])}))
print("\nHidden Layer 3:")
print("Squared Difference: ", sess.run(hidden3.squareDifference, 
                              feed_dict = {iLayer.inputLayer: Utilities.numpyReshape(inputArray[0])}))
