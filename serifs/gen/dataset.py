import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import selfies as sf
import torch
from torch.utils.data import Dataset


class GRUDataset(Dataset):
    def __init__(self, df, vectorizer, fp_len=4860, smiles_enum=False):
        """
        Dataset class for handling GRU training data.
        Args:
            df (pd.DataFrame): dataframe containing SMILES and fingerprints,
                               SMILES must be contained in ['smiles'] column as strings,
                               fingerprints must be contained in ['fps'] column as lists
                               of integers (dense vectors).
            vectorizer: SELFIES vectorizer instantiated from vectorizer.py
        """
        self.smiles = df["smiles"]
        self.fps = df["fps"]
        self.fps = self.prepare_X(self.fps)
        self.smiles = self.prepare_y(self.smiles)
        self.alphabet = vectorizer.read_alphabet()
        self.vectorizer = vectorizer
        self.fp_len = fp_len
        self.smiles_enum = smiles_enum

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, idx):
        """
        Get item from dataset.
        Args:
            idx (int): index of item to get
        Returns:
            X (torch.Tensor): reconstructed fingerprint
            y (torch.Tensor): vectorized SELFIES
        """
        raw_smile = self.smiles[idx]
        raw_selfie = ""  # placeholder
        if self.smiles_enum:
            successful = False
            n_tries = 0
            while not successful and n_tries < 3:
                randomized_smile = self.randomize_smiles(raw_smile)
                raw_selfie = sf.encoder(randomized_smile, strict=False)
                tokens = self.vectorizer.split_selfi(raw_selfie)
                all_good = True
                for token in tokens:
                    if token not in self.alphabet:
                        all_good = False
                if all_good:
                    successful = True
                else:
                    n_tries += 1
                    print("error")
            if n_tries == 3:
                raw_selfie = sf.encoder(raw_smile, strict=False)
        else:
            raw_selfie = sf.encoder(raw_smile, strict=False)
        vectorized_selfie = self.vectorizer.vectorize(raw_selfie)
        if len(vectorized_selfie) > 128:
            vectorized_selfie = vectorized_selfie[:128]
        raw_X = self.fps[idx]
        X = np.array(raw_X, dtype=int)
        X_reconstructed = self.reconstruct_fp(X)
        return (
            torch.from_numpy(X_reconstructed).float(),
            torch.from_numpy(vectorized_selfie).float(),
        )

    def randomize_smiles(self, smiles):
        """
        Randomize SMILES string.
        Args:
            smiles: SMILES string
        Returns:
            str: randomized SMILES string
        """
        m = Chem.MolFromSmiles(smiles)
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m, ans)
        return Chem.MolToSmiles(nm, canonical=False)

    def reconstruct_fp(self, fp):
        fp_rec = np.zeros(self.fp_len)
        fp_rec[fp] = 1
        return fp_rec

    def prepare_X(self, fps):
        fps = fps.apply(lambda x: np.array(x, dtype=int))
        return fps.values

    @staticmethod
    def prepare_y(selfies):
        return selfies.values


class VAEDataset(Dataset):
    """
    Dataset for variational autoencoder
    Args:
        df (pd.DataFrame): pandas DataFrame object containing 'fps' column, which contains fingerprints
        in the form of lists of integers (dense representation)
        fp_len (int): length of fingerprints
    """

    def __init__(self, df, fp_len):
        self.fps = pd.DataFrame(df["fps"])
        self.fp_len = fp_len

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, idx):
        raw_X = self.fps.iloc[idx]
        X_prepared = self.prepare_X(raw_X).values[0]
        X = np.array(X_prepared, dtype=int)
        X_reconstructed = self.reconstruct_fp(X)
        return torch.from_numpy(X_reconstructed).float()

    def reconstruct_fp(self, fp):
        fp_rec = np.zeros(self.fp_len)
        fp_rec[fp] = 1
        return fp_rec

    @staticmethod
    def prepare_X(fps):
        fps = fps.apply(lambda x: np.array(x, dtype=int))
        return fps
