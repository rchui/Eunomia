import tensorflow as tf
import numpy as np
from src.Utilities import Utilities
from src.Autoencoder import InputLayer
from src.Autoencoder import HiddenLayer
from src.Autoencoder import OutputLayer

# Number of epochs to run
numEpochs = 100
# Size of each batch
batchSize = 10
# Scaling factor for sparsity cost function
beta = 0.01

# Read in data from csv file
Utilities.progress(1, 7, status='Reading in data        ')
inputArray = Utilities.readData()

# Build input layer
Utilities.progress(2, 7, status='Building input layer   ')
with tf.variable_scope("input"):
    iLayer = InputLayer(len(inputArray[1]))

# Build hidden layer 1
Utilities.progress(3, 7, status='Building hidden layer 1')
with tf.variable_scope("hidden1"):
    hidden1 = HiddenLayer(100, iLayer.inputLayer)
    hidden1.buildTrainer(beta)

# Build hidden layer 2
Utilities.progress(4, 7, status='Building hidden layer 2')
with tf.variable_scope("hidden2"):
    hidden2 = HiddenLayer(50, hidden1.y1)
    hidden2.buildTrainer(beta)

# Build hidden layer 3
Utilities.progress(5, 7, status='Building hidden layer 3')
with tf.variable_scope("hidden3"):
    hidden3 = HiddenLayer(16, hidden2.y1)
    hidden3.buildTrainer(beta)

# Build output layer
Utilities.progress(6, 7, status='Building output layer  ')
with tf.variable_scope("output"):
    oLayer = OutputLayer(2, hidden3.y1)
    oLayer.buildTrainer()

Utilities.progress(7, 7, status='Starting session       ')
sess = Utilities.startSession()

# Print the shape of each layer
iLayer.printLayerShape()
hidden1.printLayerShape()
hidden2.printLayerShape()
hidden3.printLayerShape()
oLayer.printLayerShape()

# Training the hidden layers
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

outputList = []
for i in range(len(inputArray)):
    Utilities.progress(i + 1, len(inputArray), status='Gathering Output')
    outputList.append(sess.run(hidden3.y1, feed_dict = {iLayer.inputLayer: Utilities.numpyReshape(inputArray[i])}))

print(outputList[0][0][0])

for i in outputList:
    outputString = str(i[0])
    count = 0
    for j in range(len(i)):
        if count != 0:
            outputString += ", " + str(i[j])
        else:
            count = 1
    print(outputString)

# Training the output layer
# for i in range(numEpochs):
    # Utilities.progress(i + 1, numEpochs, status='Training Ouput Layer')
    # logits = Utilities.batchBuilder(inputArray, batchSize)
    # labels = []
    # for i in logits:
        # if i[0] > 0.5:
            # labels.append([1.0, 0.0])
        # else:
            # labels.append([0.0, 1.0])
    # sess.run(oLayer.trainStep, feed_dict = {iLayer.inputLayer: logits, oLayer.labelTensor: labels})

# Gathers the results for analysis
# for i in range(len(inputArray)):
    # Utilities.progress(i + 1, len(inputArray), status='Gathering Output')
    # outputList.append(sess.run(oLayer.yo, 
                      # feed_dict = {iLayer.inputLayer: Utilities.numpyReshape(inputArray[i])}))

num1 = 0
num2 = 0

for i in outputList:
    if i[0][0] > i[0][1]:
        num1 += 1
    else:
        num2 += 1

# Output results
print("\nNumber of 1: ", num1)
print("Number of 2: ", num2)

testCase = Utilities.numpyReshape(inputArray[0])

print("\nHidden Layer 1:")
print("Squared Difference: ", sess.run(hidden1.squareDifference, 
                                       feed_dict = {iLayer.inputLayer: testCase}))
print("\nHidden Layer 2:")
print("Squared Difference: ", sess.run(hidden2.squareDifference, 
                                       feed_dict = {iLayer.inputLayer: testCase}))
print("\nHidden Layer 3:")
print("Squared Difference: ", sess.run(hidden3.squareDifference, 
                                       feed_dict = {iLayer.inputLayer: testCase}))
