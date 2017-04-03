from src.Utilities import Utilities
from src.Autoencoder import InputLayer

inputArray = Utilities.readData()

iLayer = InputLayer(len(inputArray))

sess = Utilities.startSession()
