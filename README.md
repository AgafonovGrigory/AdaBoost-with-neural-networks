# AdaBoost-with-neural-networks
In this repo we compare the performance of two algorithms in a binary classification problem using the data from [Fermi GBM Burst Catalog](https://heasarc.gsfc.nasa.gov/w3browse/fermi/fermigbrst.html). This data contains information about gamma-ray bursts observed by a subset of the 14 GBM detectors. If parameter $t_{90}$ is more than 5 seconds, then the burst is considered “long” and is assigned a value equal to 1, otherwise a value equal to 0 is assigned.

The first algorithm is a simple neural network with one hidden layer, and the second algorithm is AdaBoost in which the same neural network is taken as a weak classifier. Main results can be seen [here](https://github.com/AgafonovGrigory/AdaBoost-with-neural-networks/blob/main/main.ipynb)

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
  + Choose $\alpha_t = \frac{1}{2}\text{ln}\left(\frac{1-N_t}{N_t}\right)$
  + Add to ensemble:
    - $H_t(x) = H_{t-1}(x) + \alpha_th_t(x)$
  + Update weights:
    - Compute $M(i) = y_ih_t(x_i)$
    - $w_{t+1}(i) = w_{t}(i)e^{-\alpha_tM(i)}$
    - Renormalize $w_{t+1}(i)$ such that $\sum_{i}w_{t+1}(i) = 1$
- Building the resulting classifier $H(x) = \text{sign}(\sum_{t}\alpha_th_t(x))$
