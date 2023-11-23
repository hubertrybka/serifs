import random

from rdkit.Chem.Scaffolds import MurckoScaffold


# scaffold split


def get_scaffold(smiles, include_chirality=False):
    """
    Returns the Murcko scaffold of a molecule
    Args:
        smiles (str): SMILES of the molecule
        include_chirality (bool): whether to include chirality in the scaffold
    Returns:
        scaffold (str): SMILES of the scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality
    )
    return scaffold


def scaffold_split(df, frac_train, shuffle=False, seed=42):
    """
    Splits a dataframe into train and validation sets by scaffold
    Args:
        df (pandas.DataFrame): dataframe to split
        frac_train (float): fraction of data to use for training
        shuffle (bool): whether to shuffle the scaffold sets before splitting
        seed (int): random seed for shuffling
    Returns:
        train_df (pandas.DataFrame): training set dataframe
        val_df (pandas.DataFrame): validation set dataframe
    """
    print("Grouping compounds by murcko scaffold...")
    # create dict of the form {scaffold_i: [idx1, idx....]}

    all_scaffolds = {}
    for i, smiles in enumerate(df.smiles):
        scaffold = get_scaffold(smiles, include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    print("Sorting keys...")
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set
        for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
        )
    ]

    # get train, valid indices
    cutoff = frac_train * len(df)

    train_idx, valid_idx = [], []

    if shuffle:
        print("Shuffling...")
        random.seed(seed)
        all_scaffold_sets = random.sample(all_scaffold_sets, len(all_scaffold_sets))

    print("Splitting...")
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > cutoff:
            valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0

    train_df = df[df.index.isin(train_idx)]
    val_df = df[df.index.isin(valid_idx)]

    return train_df, val_df
