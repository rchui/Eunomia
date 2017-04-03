from src.Utilities import Utilities
from src.Autoencoder import InputLayer
from src.Autoencoder import HiddenLayer

inputArray = Utilities.readData()

iLayer = InputLayer(len(inputArray[1]))
hidden1 = HiddenLayer(100, iLayer.input)

sess = Utilities.startSession()
