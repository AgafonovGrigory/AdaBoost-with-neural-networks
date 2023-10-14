# AdaBoost-with-neural-networks
In this repo we compare the performance of two algorithms in classifying problem using data from [Fermi GBM Burst Catalog](https://heasarc.gsfc.nasa.gov/w3browse/fermi/fermigbrst.html). The first algorithm is a simple neural network with one hidden layer, and the second algorithm is AdaBoost in which the same neural network is taken as a weak classifier. This is a binary classification problem. If parameter $t_{90}$ is more than 5 seconds, then the burst is considered “long” and is assigned a value equal to 1, otherwise a value equal to 0 is assigned.

Main results you can see [here](https://github.com/AgafonovGrigory/AdaBoost-with-neural-networks/blob/main/main.ipynb)

[Additional_func.py](https://github.com/AgafonovGrigory/AdaBoost-with-neural-networks/blob/main/additional_func.py) file contains get_and_prep_func that processes raw data, adding new features, scaling and decorrelating data. It also contains train_test_split function, accuracy function and some other functions.

[Classes.py](https://github.com/AgafonovGrigory/AdaBoost-with-neural-networks/blob/main/classes.py) file contains two classes. The first class NeuralNet is a simple neural network with one hidden layer, and the second class AdaBoost is an ensemble of neural networks.

