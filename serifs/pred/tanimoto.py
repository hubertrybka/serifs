import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from scipy.spatial.distance import cdist


def fp2bitstring(fp):
    """
    Changes the molecules fingerprint into a bitstring.

    Args:
        fp (list): list containing active bits of a vector

    Returns:
        bitstring (str): bitstring vector

    """
    bitstring = ['0'] * 512
    for x in fp:
        bitstring[x] = '1'
    return ''.join(bitstring)


def get_smiles_from_train(idx):
    """
    Returns smiles of a molecule by idx in the training set

    Args:
        idx (int): index of the molecule in the training set

    Returns:
        smiles (str): SMILES of the molecule
    """

    data_path = 'data/train_data/train_dataset.parquet'
    train = pd.read_parquet(data_path).smiles
    smiles = train.iloc[idx]
    return smiles


def unpack(ls: list):
    vec = np.zeros(512)
    vec[ls] = 1
    return vec


class TanimotoSearch:

    def __init__(self, return_smiles=False, progress_bar=True):
        data_path = 'data/train_morgan_512bits.parquet'
        self.fps = pd.read_parquet(data_path).fps.apply(eval)
        self.fps = np.array(self.fps.apply(unpack).to_list()).reshape(-1, 512)
        self.fps = pd.DataFrame(self.fps, columns=[f'FP_{i + 1}' for i in range(512)])
        self.return_smiles = return_smiles
        self.progress_bar = progress_bar
        self.XB = self.fps.to_numpy(dtype=np.int8)

    def __call__(self, mols):
        if isinstance(mols, list):
            query_fps = [Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(x, radius=2, nBits=512) for x in mols]
            ln = len(query_fps)
        else:
            query_fps = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mols, radius=2, nBits=512)
            ln = 1
        XA = np.array(query_fps, dtype=np.int8).reshape(-1, 512)
        distances = cdist(XA, self.XB, metric='jaccard').T
        tanimoto_indices = distances.argmax(axis=0)
        select_array = [[x, y] for x, y in zip(tanimoto_indices, range(ln))]
        metrics = [distances[x[0], x[1]] for x in select_array]
        return metrics
