from src.Utilities import Utilities
from src.Autoencoder import InputLayer

inputArray = Utilities.readData()

iLayer = InputLayer(len(inputArray))
hidden1 = HiddenLAyer(100, iLayer.input)

sess = Utilities.startSession()
