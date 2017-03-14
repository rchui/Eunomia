import tensorflow as tf
import numpy as np

class autoencoder:
    def startSession():
        return tf.InteractiveSession()

    def listToTensor(inputList):
        return tf.convert_to_tensor(inputLIst, dtype=tf.float32)

    def printTensor(inputTensor):
        inputTensor = tf.Print(inputTensor, [inputTensor], message = "Printing Tensor: ")
        scratch = tf.add(inputTensor, inputTensor).eval()

    
