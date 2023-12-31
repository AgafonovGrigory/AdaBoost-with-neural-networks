
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from classes import NeuralNet


def get_and_prep_func() -> pd.DataFrame:
    """
    This function adds new features, scales and decorrelates old ones

    Parameters
        ----------

    Returns
        ----------
           X_prime: pd.Dataframe
            new corrected DataFrame
    """

    # downloading data
    data = pd.read_csv('./fermigbrst.txt', sep='|')

    # deleting missing values
    for i in data.columns:
        data = data.loc[data[i] != '     null']

    # adding new features
    data['sin(lii)'] = np.sin(data['lii     ']*np.pi/180)
    data['cos(lii)'] = np.cos(data['lii     ']*np.pi/180)
    data['sin(bii)'] = np.sin(data['bii     ']*np.pi/180)
    data['cos(bii)'] = np.cos(data['bii     ']*np.pi/180)
    data['flux_log'] = np.log(1 + np.abs(data['flux_64  '].astype('float')))
    data['fluence_log'] = np.log(
        1e-6 + np.abs(data['fluence  '].astype('float')))

    # scaling data
    x = np.array([data[i] for i in ['t90_start', 'lii     ', 'bii     ', 'fluence_log', 'flux_log',
                 'flux_64_time', 'sin(lii)', 'cos(lii)', 'sin(bii)', 'cos(bii)']]).astype('float')
    for i in range(len(x)):
        x[i] = (x[i] - x[i].mean())/x[i].std()
    x = x.transpose()

    # decorrelating data
    X = pd.DataFrame(x, columns=['t90_start', 'lii', 'bii', 'fluence', 'flux_64',
                     'flux_64_time', 'sin(lii)', 'cos(lii)', 'sin(bii)', 'cos(bii)'])
    Xcov = X.cov()
    eig_values, U = np.linalg.eigh(Xcov)
    x = np.dot(x, U)
    Xprime = pd.DataFrame(
        x, columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'])

    # adding target to dataset
    y = (np.array(data['t90    ']).astype('float') > 5).astype('float')
    Xprime['target'] = y

    return Xprime




def train_test_split(df: pd.DataFrame) -> tuple:
    """
    This function splits the dataset into a test and split dataframes

    Parameters
        ----------
            df: pd.DataFrame
                original DataFrame

    Returns
        ----------
            df_train: pd.Dataframe
                train DataFrame
            df_test: pd.Dataframe
                test DataFrame
    """
    n = len(df.index)
    shuffled_indices = np.random.permutation(n)
    df = df.iloc[shuffled_indices]
    num_train = int(n * 0.8)
    df_train = df.iloc[:num_train, :]
    df_test = df.iloc[num_train:, :]
    return df_train, df_test




class Dataset(Dataset):
    """ Standard torch Dataset class
    """

    def __init__(self, df: pd.DataFrame):
        self.features = torch.tensor(
            df.iloc[:, :-1].to_numpy(), dtype=torch.float32)
        self.target = torch.tensor(
            df.iloc[:, [-1]].to_numpy(), dtype=torch.float32)
        self.n_samples = df.shape[0]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> tuple:
        features = self.features[index]
        target = self.target[index]
        return features, target





def create_torch_dataloader(df: pd.DataFrame, batch_size: int, shuffle: bool) -> torch.utils.data.dataloader.DataLoader:
    """
    This function creates torch DataLoader

    Parameters
        ----------
            df: pd.DataFrame
                original DataFrame
            batch_size: int
                batch_size
            shuffle: bool
                random permutation of dataset indices

    Returns
        ----------
            dataloader: DataLoader
    """
    dataset = Dataset(df)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle)
    return dataloader





def accuracy_on_batch(y_pred: torch.tensor, y_true: torch.tensor) -> float:
    """
    This function calculates the accuracy for any two tensors

    Parameters
        ----------
            y_pred: torch.tensor
                predicted tensor
            y_true: torch.tensor
                true values

    Returns
        ----------
            acc: float
    """
    y_pred = (y_pred > 0.5).to(torch.float32)
    true_pos = torch.sum((y_pred == y_true))
    acc = true_pos/len(y_true)*100
    return acc.item()


def accuracy(model: NeuralNet, data_loader: DataLoader) -> tuple:
    """
    This function calculates the average accuracy and deviation for the whole dataset

    Parameters
        ----------
            model: NeuralNet
                model we want to check
            dataloader: DataLoader
                dataset for checking the model

    Returns
        ----------
            mean_accuracy: float
                mean accuracy on whole dataset
            std_accuracy: float
                standard deviation
    """
    acc = np.array([])
    for i, (X, label) in enumerate(data_loader):
        y_pred = model(X)
        acc = np.append(acc, accuracy_on_batch(y_pred, label))
    return np.mean(acc), np.std(acc)



