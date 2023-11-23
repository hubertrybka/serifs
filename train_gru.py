# import packages
import argparse
import configparser
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from serifs.gen.dataset import GRUDataset
from serifs.gen.generator import EncoderDecoderV3
from serifs.gen.train import train
from serifs.utils.split import scaffold_split
from serifs.utils.vectorizer import SELFIESVectorizer


def main(config_path):
    """
    Training script for model with fully-connected encoder and GRU decoder
    """

    # read config file

    config = configparser.ConfigParser()
    config.read(config_path)
    train_size = float(config['RUN']['train_size'])
    random_seed = int(config['RUN']['random_seed'])
    run_name = str(config['RUN']['run_name'])
    batch_size = int(config['RUN']['batch_size'])
    data_path = str(config['RUN']['data_path'])
    NUM_WORKERS = int(config['RUN']['num_workers'])
    smiles_enum = config.getboolean('RUN', 'smiles_enum')
    encoding_size = int(config['MODEL']['encoding_size'])
    hidden_size = int(config['MODEL']['hidden_size'])
    num_layers = int(config['MODEL']['num_layers'])
    dropout = float(config['MODEL']['dropout'])
    teacher_ratio = float(config['MODEL']['teacher_ratio'])
    fp_len = int(config['MODEL']['fp_len'])
    fc1_size = int(config['MODEL']['fc1_size'])
    fc2_size = int(config['MODEL']['fc2_size'])
    fc3_size = int(config['MODEL']['fc3_size'])
    encoder_activation = str(config['MODEL']['encoder_activation'])
    use_cuda = config.getboolean('RUN', 'use_cuda')

    val_size = round(1 - train_size, 1)
    train_percent = int(train_size * 100)
    val_percent = int(val_size * 100)

    cuda_available = (torch.cuda.is_available() and use_cuda)
    device = torch.device('cuda' if cuda_available else 'cpu')

    print('Using device:', device)

    vectorizer = SELFIESVectorizer(pad_to_len=128)

    # read dataset

    dataset = pd.read_parquet(data_path)

    # create a directory for this model weights if not there

    if not os.path.isdir(f'models/{run_name}'):
        os.mkdir(f'models/{run_name}')

    with open(f'models/{run_name}/hyperparameters.ini', 'w') as configfile:
        config.write(configfile)

    # if train_dataset not generated, perform scaffold split

    if (not os.path.isfile(data_path.split('.')[0] + f'_train_{train_percent}.parquet')
            or not os.path.isfile(data_path.split('.')[0] + f'_val_{val_percent}.parquet')):
        train_df, val_df = scaffold_split(dataset, train_size, seed=random_seed, shuffle=True)
        train_df.to_parquet(data_path.split('.')[0] + f'_train_{train_percent}.parquet')
        val_df.to_parquet(data_path.split('.')[0] + f'_val_{val_percent}.parquet')
        print("Scaffold split complete")
    else:
        train_df = pd.read_parquet(data_path.split('.')[0] + f'_train_{train_percent}.parquet')
        val_df = pd.read_parquet(data_path.split('.')[0] + f'_val_{val_percent}.parquet')
    scoring_df = val_df.sample(frac=0.1, random_state=random_seed)

    # prepare dataloaders

    train_dataset = GRUDataset(train_df, vectorizer, fp_len, smiles_enum=smiles_enum)
    val_dataset = GRUDataset(val_df, vectorizer, fp_len, smiles_enum=False)
    scoring_dataset = GRUDataset(scoring_df, vectorizer, fp_len, smiles_enum=False)

    print("Dataset size:", len(dataset))
    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))
    print("Scoring size:", len(scoring_dataset))

    val_batch_size = batch_size if batch_size < len(val_dataset) else len(val_dataset)
    scoring_batch_size = batch_size if batch_size < len(scoring_dataset) else len(scoring_dataset)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                              drop_last=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=val_batch_size,
                            drop_last=True, num_workers=NUM_WORKERS)
    scoring_loader = DataLoader(scoring_dataset, shuffle=False, batch_size=scoring_batch_size,
                                drop_last=True, num_workers=NUM_WORKERS)

    # Init model

    model = EncoderDecoderV3(
        fp_size=fp_len,
        encoding_size=encoding_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        teacher_ratio=teacher_ratio,
        output_size=31,  # alphabet length
        fc1_size=fc1_size,
        fc2_size=fc2_size,
        fc3_size=fc3_size,
        encoder_activation=encoder_activation
    ).to(device)

    _ = train(config, model, train_loader, val_loader, scoring_loader)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='config_files/RNN_config.ini',
                        help='Path to config file')
    config_path = parser.parse_args().config
    main(config_path)
