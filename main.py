from src.Utilities import Utilities
from src.Autoencoder import InputLayer
from src.Autoencoder import HiddenLayer

inputArray = Utilities.readData()

iLayer = InputLayer(len(inputArray[1]))
iLayer.printLayerShape()
hidden1 = HiddenLayer(100, iLayer.self.input)
hidden1.printLayerShape()
hidden2 = HiddenLayer(50, hidden1.self.z2)
hidden2.printLayerShape()

sess = Utilities.startSession()
