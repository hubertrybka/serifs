import argparse
import configparser
import os
import time

import pandas as pd
import rdkit.Chem as Chem
import rdkit.Chem.Draw as Draw
import torch

from serifs.pred.pred import predict_with_dropout, filter_dataframe
from serifs.utils.modelinit import initialize_model


def main(file_path, model_path, config_path, n_samples, use_cuda, verbosity):
    """
    Predicting molecules using the trained model.

    Args:
        file_path (str): Path to the file containing latent vectors.
        model_path (str): Path to the model weights.
        config_path: Path to the config file.
    Returns: None
    """

    # setup
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(config_path)

    # get file name
    dirname = os.path.dirname(file_path)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    print(f'Using {device} device') if verbosity > 0 else None

    model_epoch = model_path.split('/')[-1]
    model_config_path = model_path.replace(model_epoch, 'hyperparameters.ini')
    if not os.path.exists(model_config_path):
        raise ValueError(f'Model config file {model_config_path} not found')

    # load model

    model = initialize_model(config_path=model_config_path,
                             dropout=True if n_samples > 1 else False,
                             device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f'Loaded model from {model_path}') if verbosity > 1 else None

    # load data
    if file_path.endswith('.csv'):
        query_df = pd.read_csv(file_path)
    elif file_path.endswith('.parquet'):
        query_df = pd.read_parquet(file_path)
    else:
        raise ValueError('Data file format not supported (must be .csv or .parquet)')

    for col in ['smiles', 'label', 'score', 'activity', 'norm']:
        if col in query_df.columns:
            query_df = query_df.drop(columns=[col])
    input_vector = query_df.to_numpy()
    print(f'Loaded data from {file_path}') if verbosity > 1 else None

    # get predictions
    print(f'Getting predictions for file {file_path}...') if verbosity > 1 else None
    df = predict_with_dropout(model,
                              input_vector,
                              n_samples=n_samples,
                              device=device,
                              fix_mols=config['SUBSTRUCTURE_FIX'].getboolean('apply')
                              )

    # filter dataframe
    df = filter_dataframe(df, config)

    # save data as csv
    os.mkdir(f'{dirname}/preds_{timestamp}')
    with open(f'{dirname}/preds_{timestamp}/config.ini', 'w') as configfile:
        config.write(configfile)
    df.to_csv(f'{dirname}/preds_{timestamp}/predictions.csv', index=False)

    print(f'Saved data to {dirname}/preds_{timestamp} directory') if verbosity > 0 else None

    # save images
    os.mkdir(f'{dirname}/preds_{timestamp}/imgs')
    for n, (idx, smiles) in enumerate(zip(df['idx'], df['smiles'])):
        mol = Chem.MolFromSmiles(smiles)
        Draw.MolToFile(mol, f'{dirname}/preds_{timestamp}/imgs/{idx}_{n}.png', size=(300, 300))

    time_elapsed = time.time() - start_time
    print(f'File processed in {(time_elapsed / 60):.2f} minutes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        '-c',
                        type=str,
                        default='config_files/pred_config.ini',
                        help='Path to config file')
    parser.add_argument('-d',
                        '--data_path',
                        type=str,
                        required=True,
                        help='Path to data file')
    parser.add_argument('-m',
                        '--model_path',
                        required=True,
                        type=str,
                        help='Path to model weights')
    parser.add_argument('-v',
                        '--verbosity',
                        type=int,
                        default=1,
                        help='Verbosity level (0 - silent, 1 - progress, 2 - verbose)')
    parser.add_argument('-n',
                        '--n_samples',
                        type=int,
                        default=10,
                        help='Number of samples to generate for each latent vector. If > 1, the variety of the generated molecules will be increased by using dropout.')
    parser.add_argument('-u',
                        '--use_cuda',
                        type=bool,
                        default=True,
                        help='Use cuda if available')

    args = parser.parse_args()
    main(file_path=args.data_path,
         model_path=args.model_path,
         config_path=args.config,
         n_samples=args.n_samples,
         use_cuda=args.use_cuda,
         verbosity=args.verbosity)
