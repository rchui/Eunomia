import csv
import random
import sys
import tensorflow as tf

class Utilities:
    def readData():
        count = 0
        fileName = sys.argv[1]
        inputLayer = []
        with open(fileName) as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                row.pop()
                if count != 0:
                    floatRow = [float(i) for i in row]
                    inputLayer.append(floatRow)
                count += 1
        csvFile.close()
        return inputLayer

    def startSession():
        sess = tf.InteractiveSession()
        tf.global_variables_initializer.run()
        return sess
