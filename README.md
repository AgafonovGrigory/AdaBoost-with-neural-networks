# AdaBoost-with-neural-networks
In this repo we compare the performance of two algorithms in classifying problem using data from [Fermi GBM Burst Catalog](https://heasarc.gsfc.nasa.gov/w3browse/fermi/fermigbrst.html). The first algorithm is a simple neural network with one hidden layer, and the second algorithm is AdaBoost in which the same neural network is taken as a weak classifier. This is a binary classification problem. If parameter $t_{90}$ is more than 5 seconds, then the burst is considered “long” and is assigned a value equal to 1, otherwise a value equal to 0 is assigned.

Main results you can see [here](https://github.com/AgafonovGrigory/AdaBoost-with-neural-networks/blob/main/main.ipynb)

[Additional_func.py](https://github.com/AgafonovGrigory/AdaBoost-with-neural-networks/blob/main/additional_func.py) file contains get_and_prep_func that processes raw data, adding new features, scaling and decorrelating data. It also contains train_test_split function, accuracy function and some other functions.

[Classes.py](https://github.com/AgafonovGrigory/AdaBoost-with-neural-networks/blob/main/classes.py) file contains two classes. The first class NeuralNet is a simple neural network with one hidden layer, and the second class AdaBoost is an ensemble of neural networks.
### AdaBoost
Suppose we have a data set $\\{(x_1, y_1) \ldots (x_N, y_N)\\}$ where each item $x_i$ has an associated class $y_i \in \\{-1, 1\\}$, and a set of weak classifiers $\\{h_1, \ldots, h_T\\}$ each of which outputs a classification $h_j(x_i) \in \\{-1, 1\\}$ for each item. After the $(m-1)$-th iteration our boosted classifier is a linear combination of the weak classifiers of the form:
```math
\begin{equation}
H_{m-1}(x_i) = \alpha_1h_1(x_i) + \cdots + \alpha_{m-1}h_{m-1}(x_i)
\end{equation}
```
where the class will be the sign of $H_{m-1}(x_i)$. At the $m$-th iteration we want to extend this to a better boosted classifier by adding another weak classifier $h_m$, with another weight $\alpha_m$:
```math
\begin{equation}
H_{m}(x_i) = H_{m-1}(x_i) + \alpha_{m}h_{m}(x_i)
\end{equation}
```
Algorithm:
- Initialize $w_1(i)=1/N$ where $i=1\ldots N$
- For each $t = 1\ldots T$
  + Find a classifier $h_t$ that minimizes the weighted classification error $h_t = \underset{h_j\in \mathcal{H}}{\mathrm{argmin}}(N_j)$ where $N_j = \sum_{i} w_{t}(i)[y_i \neq h_{j}(x_{i})]$
