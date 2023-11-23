import numpy as np
import pandas as pd
import rdkit.Chem.Crippen as Crippen
import selfies as sf
import torch
from rdkit import Chem
from rdkit.Chem import QED, rdMolDescriptors

from serifs.utils.vectorizer import SELFIESVectorizer


def predict_with_dropout(
    model,
    latent_vectors: np.array,
    n_samples: int = 10,
    device: torch.device = torch.device("cpu"),
):
    """
    Generate molecules from latent vectors with dropout.
    Args:
        model (torch.nn.Module): EncoderDecoderv3 model.
        latent_vectors (np.array): numpy array of latent vectors. Shape = (n_samples, latent_size).
        n_samples (int): number of samples to generate for each latent vector.
        device: device to use for prediction. Can be 'cpu' or 'cuda'.

    Returns:
        pd.DataFrame: Dataframe containing smiles and scores.
    """
    vectorizer = SELFIESVectorizer(pad_to_len=128)
    device = torch.device(device)
    dataframes = []

    with torch.no_grad():
        for n in range(n_samples):
            df = pd.DataFrame(columns=["idx", "smiles"])
            latent_tensor = torch.Tensor(latent_vectors).to(device)
            model = model.to(device)
            preds, _ = model(latent_tensor, None, omit_encoder=True)
            preds = preds.detach().cpu().numpy()
            df["selfies"] = [
                vectorizer.devectorize(pred, remove_special=True) for pred in preds
            ]
            df["smiles"] = df["selfies"].apply(sf.decoder)
            df["mols"] = df["smiles"].apply(Chem.MolFromSmiles)
            # sanitize mols
            df["mols"] = df["mols"].apply(try_sanitize)
            # drop redundant columns
            df.drop(columns=["mols", "selfies"], inplace=True)
            df["idx"] = range(len(df))
            dataframes.append(df)
    df = pd.concat(dataframes)
    df = df.sort_values(by=["idx"])
    df = df.drop_duplicates(subset=["smiles"])
    return df


def get_largest_ring(mol):
    ri = mol.GetRingInfo()
    rings = []
    for b in mol.GetBonds():
        ring_len = [len(ring) for ring in ri.BondRings() if b.GetIdx() in ring]
        rings += ring_len
    return max(rings) if rings else 0


def try_sanitize(mol):
    try:
        output = mol
        Chem.SanitizeMol(output)
        return output
    except:
        return mol


def filter_dataframe(df, config):
    df_copy = df.copy()
    df_copy["mols"] = df_copy["smiles"].apply(Chem.MolFromSmiles)

    # filter by largest ring
    df_copy["largest_ring"] = df_copy["mols"].apply(get_largest_ring)
    if config["RING_SIZE"]["min"].lower() != "none":
        df_copy = df_copy[df_copy["largest_ring"] >= int(config["RING_SIZE"]["min"])]
    if config["RING_SIZE"]["max"].lower() != "none":
        df_copy = df_copy[df_copy["largest_ring"] <= int(config["RING_SIZE"]["max"])]
    print(f"Number of molecules after filtering by ring size: {len(df_copy)}")

    # filter by num_rings
    df_copy["num_rings"] = df_copy["mols"].apply(Chem.rdMolDescriptors.CalcNumRings)
    if config["NUM_RINGS"]["min"].lower() != "none":
        df_copy = df_copy[df_copy["num_rings"] >= int(config["NUM_RINGS"]["min"])]
    if config["NUM_RINGS"]["max"].lower() != "none":
        df_copy = df_copy[df_copy["num_rings"] <= int(config["NUM_RINGS"]["max"])]
    print(f"Number of molecules after filtering by num_rings: {len(df_copy)}")

    # filter by QED
    df_copy["qed"] = df_copy["mols"].apply(QED.default)
    if config["QED"]["min"].lower() != "none":
        df_copy = df_copy[df_copy["qed"] >= float(config["QED"]["min"])]
    if config["QED"]["max"].lower() != "none":
        df_copy = df_copy[df_copy["qed"] <= float(config["QED"]["max"])]
    print(f"Number of molecules after filtering by QED: {len(df_copy)}")

    # filter by mol_wt
    df_copy["mol_wt"] = df_copy["mols"].apply(Chem.rdMolDescriptors.CalcExactMolWt)
    if config["MOL_WEIGHT"]["min"].lower() != "none":
        df_copy = df_copy[df_copy["mol_wt"] >= float(config["MOL_WEIGHT"]["min"])]
    if config["MOL_WEIGHT"]["max"].lower() != "none":
        df_copy = df_copy[df_copy["mol_wt"] <= float(config["MOL_WEIGHT"]["max"])]
    print(f"Number of molecules after filtering by mol_wt: {len(df_copy)}")

    # filter by num_HBA
    df_copy["num_HBA"] = df_copy["mols"].apply(rdMolDescriptors.CalcNumHBA)
    if config["NUM_HBA"]["min"].lower() != "none":
        df_copy = df_copy[df_copy["num_HBA"] >= int(config["NUM_HBA"]["min"])]
    if config["NUM_HBA"]["max"].lower() != "none":
        df_copy = df_copy[df_copy["num_HBA"] <= int(config["NUM_HBA"]["max"])]
    print(f"Number of molecules after filtering by num_HBA: {len(df_copy)}")

    # filter by num_HBD
    df_copy["num_HBD"] = df_copy["mols"].apply(rdMolDescriptors.CalcNumHBD)
    if config["NUM_HBD"]["min"].lower() != "none":
        df_copy = df_copy[df_copy["num_HBD"] >= int(config["NUM_HBD"]["min"])]
    if config["NUM_HBD"]["max"].lower() != "none":
        df_copy = df_copy[df_copy["num_HBD"] <= int(config["NUM_HBD"]["max"])]
    print(f"Number of molecules after filtering by num_HBD: {len(df_copy)}")

    # filter by logP
    df_copy["logP"] = df_copy["mols"].apply(Crippen.MolLogP)
    if config["LOGP"]["min"].lower() != "none":
        df_copy = df_copy[df_copy["logP"] >= float(config["LOGP"]["min"])]
    if config["LOGP"]["max"].lower() != "none":
        df_copy = df_copy[df_copy["logP"] <= float(config["LOGP"]["max"])]
    print(f"Number of molecules after filtering by logP: {len(df_copy)}")

    # filter by num_rotatable_bonds
    df_copy["num_rotatable_bonds"] = df_copy["mols"].apply(
        rdMolDescriptors.CalcNumRotatableBonds
    )
    if config["NUM_ROT_BONDS"]["min"].lower() != "none":
        df_copy = df_copy[
            df_copy["num_rotatable_bonds"] >= int(config["NUM_ROTATABLE_BONDS"]["min"])
        ]
    if config["NUM_ROT_BONDS"]["max"].lower() != "none":
        df_copy = df_copy[
            df_copy["num_rotatable_bonds"] <= int(config["NUM_ROTATABLE_BONDS"]["max"])
        ]
    print(f"Number of molecules after filtering by num_rotatable_bonds: {len(df_copy)}")

    # filter by TPSA
    df_copy["tpsa"] = df_copy["mols"].apply(rdMolDescriptors.CalcTPSA)
    if config["TPSA"]["min"].lower() != "none":
        df_copy = df_copy[df_copy["tpsa"] >= float(config["TPSA"]["min"])]
    if config["TPSA"]["max"].lower() != "none":
        df_copy = df_copy[df_copy["tpsa"] <= float(config["TPSA"]["max"])]
    print(f"Number of molecules after filtering by TPSA: {len(df_copy)}")

    # filter by bridgehead atoms
    df_copy["bridgehead_atoms"] = df_copy["mols"].apply(
        rdMolDescriptors.CalcNumBridgeheadAtoms
    )
    if config["NUM_BRIDGEHEAD_ATOMS"]["min"].lower() != "none":
        df_copy = df_copy[
            df_copy["bridgehead_atoms"] >= int(config["NUM_BRIDGEHEAD_ATOMS"]["min"])
        ]
    if config["NUM_BRIDGEHEAD_ATOMS"]["max"].lower() != "none":
        df_copy = df_copy[
            df_copy["bridgehead_atoms"] <= int(config["NUM_BRIDGEHEAD_ATOMS"]["max"])
        ]
    print(f"Number of molecules after filtering by bridgehead atoms: {len(df_copy)}")

    # filter by spiro atoms
    df_copy["spiro_atoms"] = df_copy["mols"].apply(rdMolDescriptors.CalcNumSpiroAtoms)
    if config["NUM_SPIRO_ATOMS"]["min"].lower() != "none":
        df_copy = df_copy[
            df_copy["spiro_atoms"] >= int(config["NUM_SPIRO_ATOMS"]["min"])
        ]
    if config["NUM_SPIRO_ATOMS"]["max"].lower() != "none":
        df_copy = df_copy[
            df_copy["spiro_atoms"] <= int(config["NUM_SPIRO_ATOMS"]["max"])
        ]
    print(f"Number of molecules after filtering by spiro atoms: {len(df_copy)}")

    # drop redundant columns
    df_copy.drop(columns=["mols"], inplace=True)

    return df_copy
