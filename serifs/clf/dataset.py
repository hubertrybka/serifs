import pandas as pd
import torch
from torch.utils.data import Dataset


class ClfDataset(Dataset):
    """
    Dataset for the discriminator model
    Args:
        mu_path (str): path to the mu parquet file containing the mu values and activity labels in 'label' column
        logvar_path (str): path to the logvar parquet file
    """

    def __init__(self, mu_path, logvar_path=None):
        super().__init__()
        self.mu, self.activity = self.load_mu_n_labels(mu_path)
        if logvar_path is not None:
            self.use_logvar = True
            self.logvar = self.load_logvar(logvar_path)
        else:
            self.use_logvar = False
            self.logvar = torch.zeros(self.mu.shape)

    def __getitem__(self, idx):
        if self.use_logvar:
            encoding = reparameterize(self.mu[idx], self.logvar[idx])
        else:
            encoding = self.mu[idx]
        activity = self.activity[idx].float()
        return encoding, activity

    def __len__(self):
        return len(self.mu)

    @staticmethod
    def load_mu_n_labels(path):
        df = pd.read_parquet(path)
        labels = torch.tensor(df.label.to_numpy())
        df = df.drop(columns=['label', 'smiles'])
        tensor = torch.tensor(df.to_numpy())
        return tensor, labels

    @staticmethod
    def load_logvar(path):
        df = pd.read_parquet(path)
        tensor = torch.tensor(df.to_numpy())
        return tensor


def reparameterize(mu, logvar):
    """
    Reparameterization trick for sampling VAE latent space
    Args:
        mu (torch.Tensor): tensor of mu values
        logvar (torch.Tensor): tensor of logvar values
    Returns:
        torch.Tensor: tensor of sampled values
    """

    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)
