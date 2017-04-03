import tensorflow as tf
from src.Utilities import Utilities
from src.Autoencoder import InputLayer
from src.Autoencoder import HiddenLayer

inputArray = Utilities.readData()

with tf.variable_scope("input"):
    iLayer = InputLayer(len(inputArray[1]))
iLayer.printLayerShape()

with tf.variable_scope("hidden1"):
    hidden1 = HiddenLayer(100, iLayer.input)
hidden1.printLayerShape()

with tf.variable_scope("hidden2"):
    hidden2 = HiddenLayer(50, hidden1.y2)

sess = Utilities.startSession()
