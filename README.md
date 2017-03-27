# Eunomia

## Introduction
Eunomia is a project being developed at the National Center for Supercomputing Applications in collaboration with Matthew Weber and John Capozzo. The aim of the project is to answer two questions. First, are mutation transition porbaiblities indiciative of known cancer subtypes and can they be used to find more distinc cancer subtypes? Seocnd, can the MDMR algorithm developed by John and Alex Lipka, effectively pick out features to increase the predictive power of deep neural networks.

## Methods
We are utilizing Google's TensorFlow dataflow graph framework to construct a stacked autoencoder. Each hidden layer of the neural network undergoes unsupervised training in a greedy layer-by-layer progression with each decoding layer being thrown away as successive hidden layers are trained. The output layer is currently being trained using supervised learning but the eventual goal is to perform unsupervised multi-label classification on the given data to elucidate hidden subtypes in the cancer data.
