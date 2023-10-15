import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np


class NeuralNet(nn.Module):
    """ Class for 1 layer neural network

        Attributes:
        `  ----------
                input size: int
                    number of input features
                layer_length: int 
                    size of hidden layer

    """

    def __init__(self, input_size: int, layer_length: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(
            input_size, layer_length), nn.ReLU(), nn.Linear(layer_length, 1), nn.Sigmoid())

    def forward(self, x: torch.tensor) -> float:
        """
         This method returns the probability of belonging to a class 1 for input data x

         Parameters
         `  ----------
                 x: torch.tensor
                     input data
         Returns:
         `  ----------
                 p: float
                     probability of class with target equals 1
         """

        return self.layers(x)





class AdaBoost():
    """ Class for AdaBoost with NeuralNet as weak classifier

        Attributes:
        `  ----------
                weak_classifier: NeuralNet
                    model of weak classifier
                T: int
                    number of weak classifiers
                N: int
                    size of train data
                X_train: torch.tensor
                   train data
                y_train: torch.tensor
                    target for train data
                weights: torch.tensor
                    weights of train data
                alphas: list
                    list of coefficents for each weak classifier
                trained_models: list
                    list of trained weak classifiers

    """

    def __init__(self, weak_classifier: NeuralNet, T: int, df_train: pd.DataFrame) -> None:
        self.weak_classifier = weak_classifier
        self.T = T
        self.N = len(df_train)
        self.X_train = torch.tensor(
            df_train.iloc[:, :-1].to_numpy(), dtype=torch.float32)
        self.y_train = torch.tensor(
            df_train.iloc[:, [-1]].to_numpy(), dtype=torch.float32)
        self.weights = 1/self.N * torch.ones(self.N, dtype=torch.float32)
        self.alphas = []
        self.trained_models = []

    def criterion_for_weak_classifier(self, weak_classifier: NeuralNet) -> torch.tensor:
        """
            This method implements loss function for a weak classifier

            Parameters
            `  ----------
                    weak_classifier: NeuralNet
                        model of a weak classifier
            Returns:
            `  ----------
                    res: torch.tensor
                        weighted MSE loss
            """
        X_train = self.X_train
        y_train = self.y_train
        weights = self.weights
        y_pred = weak_classifier(X_train)
        squared_diff = ((y_pred - y_train).squeeze(-1))**2
        return torch.dot(weights, squared_diff)

    def train_weak_classifier(self, weak_classifier: NeuralNet, n_steps: int, learning_rate: float) -> None:
        """
            This method implements training of a weak classifier

            Parameters
            `  ----------
                    weak_classifier: NeuralNet
                        model of a weak classifier
                    n_steps: int
                        number of optimization steps
                    learning_rate: float
                        learning rate
            Returns:
            `  ----------
                    None
            """
        optimizer = torch.optim.Adam(
            weak_classifier.parameters(), lr=learning_rate)
        for step in range(n_steps):
            loss = self.criterion_for_weak_classifier(weak_classifier)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def N_(self, weak_classifier: NeuralNet) -> float:
        """
            Weighted proportion of incorrect predictions for a weak classifier

            Parameters
            `  ----------
                    weak_classifier: NeuralNet
                        model of a weak classifier
            Returns:
            `  ----------
                    res: float
                        weighted proportion of incorrect predictions
            """
        with torch.no_grad():
            X_train = self.X_train
            y_train = self.y_train
            weights = self.weights
            y = weak_classifier(X_train)
            y_pred = (y > 0.5)
        return torch.dot((y_pred != y_train).to(torch.float32).squeeze(-1), weights).item()

    def compute_alpha(self, weak_classifier: NeuralNet) -> float:
        """
            Method computes coefficent alpha for a weak classifier in ensemble

            Parameters
            `  ----------
                    weak_classifier: NeuralNet
                        model of a weak classifier
            Returns:
            `  ----------
                    res: float
                        coefficent for a weak classifier in ensemble
            """
        with torch.no_grad():
            err = self.N_(weak_classifier)
            return 0.5*np.log((1-err)/err)

    def update_weights(self, weak_classifier) -> None:
        """
            Method updates weights in AdaBoost algorithm

            Parameters
            `  ----------
                    weak_classifier: NeuralNet
                        model of weak classifier
            Returns:
            `  ----------
                   None
            """
        y_pred = weak_classifier(self.X_train)
        alpha = self.compute_alpha(weak_classifier).item()
        self.alphas.append(alpha)
        new_weights = torch.zeros_like(self.weights)
        for i in range(self.N):
            shifted_y_train = 2*self.y_train[i] - 1
            shifted_y_pred = 2*y_pred[i] - 1
            margin = shifted_y_pred.item() * shifted_y_train.item()
            new_weight = self.weights[i] * np.exp(-alpha*margin)
            new_weights[i] = new_weight

        norm = new_weights.sum()
        self.weights = new_weights/norm

    def main_train(self, layer_length: int, n_steps: int, lr: float) -> None:
        """
            Method trains AdaBoost

            Parameters
            `  ----------
                    layer_length: int
                        size of hidden layer in weak classifier
                    n_steps: int
                        number of steps in optimization of weak classifier
                    lr: float
                        learning rate in optimization of weak classifier
            Returns:
            `  ----------
                   None
            """
        for i in range(self.T):
            model = self.weak_classifier(10, layer_length)
            self.train_weak_classifier(model, n_steps, lr)
            self.trained_models.append(model)
            self.update_weights(model)

    def predict(self, X):
        """
         Method returns target for input data X

         Parameters
         `  ----------
                 X: torch.tensor
                     input data
         Returns:
         `  ----------
                 res: int
                     target
         """
        pred = 0
        for i in range(self.T):
            model_pred = 2*self.trained_models[i](X) - 1
            pred += model_pred*self.alphas[i]

        return 0.5*(torch.sign(pred) + 1)






