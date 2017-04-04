import tensorflow as tf
from src.Utilities import Utilities
from src.Autoencoder import InputLayer
from src.Autoencoder import HiddenLayer
from src.Autoencoder import OutputLayer

Utilities.progress(1, 10, status='Reading in data')

# Read in data from csv file
inputArray = Utilities.readData()

Utilities.progress(2, 10, status='Building input layer')

# Build input layer
with tf.variable_scope("input"):
    iLayer = InputLayer(len(inputArray[1]))
iLayer.printLayerShape()

Utilities.progress(3, 10, status='Building hidden layer 1')

# Build hidden layer 1
with tf.variable_scope("hidden1"):
    hidden1 = HiddenLayer(100, iLayer.inputLayer)
    hidden1.buildTrainer()
hidden1.printLayerShape()

# Build hidden layer 2
with tf.variable_scope("hidden2"):
    hidden2 = HiddenLayer(50, hidden1.y1)
    hidden2.buildTrainer()
hidden2.printLayerShape()

# Build hidden layer 3
with tf.variable_scope("hidden3"):
    hidden3 = HiddenLayer(16, hidden2.y1)
    hidden3.buildTrainer()
hidden3.printLayerShape()

# Build output layer
with tf.variable_scope("output"):
    oLayer = OutputLayer(2, hidden3.y1)
    oLayer.buildTrainer()
oLayer.printLayerShape()

sess = Utilities.startSession()

for i in range(len(inputArray)):
    sess.run(hidden1.trainStep, 
             feed_dict = {iLayer.inputLayer: Utilities.numpyReshape(inputArray[i])})

for i in range(len(inputArray)):
    sess.run(hidden2.trainStep, 
             feed_dict = {iLayer.inputLayer: Utilities.numpyReshape(inputArray[i])})

for i in range(len(inputArray)):
    sess.run(hidden3.trainStep, 
             feed_dict = {iLayer.inputLayer: Utilities.numpyReshape(inputArray[i])})

for i in range(len(inputArray)):
    if inputArray[i][0] > 0.5:
        labels = [1.0, 0.0]
    else:
        labels = [0.0, 1.0]
    sess.run(oLayer.trainStep,
             feed_dict = {iLayer.inputLayer: Utilities.numpyReshape(inputArray[i]), 
                          oLayer.labelTensor: Utilities.numpyReshape(labels)})

outputList = []
for i in range(len(inputArray)):
    outputList.append(sess.run(oLayer.yo, 
                      feed_dict = {iLayer.inputLayer: Utilities.numpyReshape(inputArray[i])}))

num1 = 0
num2 = 0

for i in outputList:
    if abs(i[0][0]) > abs(i[0][1]):
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
