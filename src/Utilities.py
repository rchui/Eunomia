import csv
import sys
import random
import tensorflow as tf
import numpy as np

class Utilities:
    def readData():
        """ Reads in data passed by the user from a CSV file. """
        fileName = sys.argv[1]
        inputArray = []
        with open(fileName) as csvFile:
            reader = csv.reader(csvFile)
            arraySlice = []
            for row in reader:
                arraySlice = (row[48:235])
                for i in range(len(arraySlice)):
                    if arraySlice[i] == "":
                        arraySlice[i] = -10000
                arraySlice = [float(i) for i in arraySlice]
                inputArray.append(arraySlice)
        csvFile.close()
        return inputArray

    def batchBuilder(array, batchSize):
        """
        Builds batches of samples from the read in data.
        @params:
            array -- holds samples that batches are built from
            batchSize -- the size of each batch
        """
        # np.random.shuffle(array)
        dictFeeder = []
        numBatches = len(array) // batchSize
        for i in range(numBatches):
            start = i * batchSize
            dictFeeder.append(array[start:start + batchSize])
        dictFeeder.append(array[(numBatches * batchSize):(len(array) - 1)])
        return dictFeeder

    def numpyReshape(array):
        """
        Reshapes a given array to match tensor dimensions.
        @params:
            array -- the array to be reshaped and turned into a numpy array
        """
        return np.array(array, dtype = float).reshape(1, len(array))

    def startSession():
        """ Starts an interactive session to run the tensorflow graph. """
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        return sess

    def progress(count, total, status=''):
        """
        Displays a progress bar that updates with the program's progress.
        @params:
            count -- distance traveled so far
            total -- total distance to travel
            status -- updates on the progress of the program
        """
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)
        
        if count >= total: 
            sys.stdout.write('[%s] %s%s ...%s%s\r' % (bar, percents, '%', status, '\n'))
            sys.stdout.flush()
        else:
            sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
            sys.stdout.flush()

    def dataRecorder(layerNum, squareDifference ,epoch):
        """ 
        Records layer accuracy against the epoch
        @params:
            layerNum -- hidden layer being recorded
            squareDifference -- squareDifference for layer
            epoch -- epoch training is on
        """
        writeStream = open('layer.' + str(layerNum) + '.csv', 'a')
        writeStream.write(str(epoch) + "," + str(squareDifference) + "\n")
        writeStream.close()
