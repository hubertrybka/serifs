import argparse
import os
import pickle
import time
import configparser

import numpy as np
import pandas as pd
import sklearn.model_selection
import torch
import torch.utils.data as D
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.svm import SVC

from serifs.gen.dataset import VAEDataset
from serifs.gen.generator import EncoderDecoderV3
from serifs.utils.modelinit import initialize_model


def main(config_path):
    """
    Trains an SVM classifier on the latent space of the model.
    """

    # read config file
    
    config = configparser.ConfigParser()
    config.read(config_path)
    data_path = str(config['SVC']['data_path'])
    model_path = str(config['SVC']['model_path'])
    c_param = float(config['SVC']['c_param'])
    kernel = str(config['SVC']['kernel'])
    gamma = str(config['SVC']['gamma'])
    use_cuda = config.getboolean('SVC', 'use_cuda')

    start_time = time.time()

    cuda_available = (torch.cuda.is_available() and use_cuda)
    device = torch.device('cuda' if cuda_available else 'cpu')
    print(f'Using device: {device}')

    # read dataset

    data = pd.read_parquet(data_path, columns=['smiles', 'activity', 'fps'])
    data.reset_index(drop=True, inplace=True)
    print(f'Loaded data from {data_path}')
    activity = data['activity']

    # load model

    config_path = '/'.join(model_path.split('/')[:-1]) + '/hyperparameters.ini'
    model = initialize_model(config_path,
                             dropout=False,
                             device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # encode data into latent space vectors

    print('Encoding data...')
    mus, _ = encode(data, model, device)
    data = pd.DataFrame(mus)
    data['activity'] = activity
    data.reset_index(drop=True, inplace=True)

    # split into train and test set

    train, test = sklearn.model_selection.train_test_split(data, test_size=0.1, random_state=42)

    # train SVM

    SV_params = {'C': c_param,
                 'kernel': kernel,
                 'gamma': gamma,
                 'shrinking': True,
                 'probability': True,
                 'max_iter': -1}

    svc = SVC(**SV_params)

    train_X = train.drop('activity', axis=1)
    train_y = train['activity']
    test_X = test.drop('activity', axis=1)
    test_y = test['activity']
    print('Training set size:', train_X.shape[0])
    print('Test set size:', test_X.shape[0])

    print('Training...')
    svc.fit(train_X, train_y)
    model_name = model_path.split('/')[-2]
    clf_name = f'SVC_{model_name}'

    # save model

    if not os.path.exists(f'models/{clf_name}'):
        os.mkdir(f'models/{clf_name}')
    with open(f'./models/{clf_name}/clf.pkl', 'wb') as file:
        pickle.dump(svc, file)

    # evaluate

    print('Evaluating...')
    metrics = evaluate(svc, test_X, test_y)

    metrics_df = pd.DataFrame(metrics, index=[0])
    metrics_df.to_csv(f'models/{clf_name}/metrics.csv', index=False)

    time_elapsed = round((time.time() - start_time), 2)
    if time_elapsed < 60:
        print(f'Executed in {time_elapsed} seconds')
    else:
        print(f'Executed in {round(time_elapsed / 60, 2)} minutes')
    return


def encode(df, model, device):
    """
    Encodes the fingerprints of the molecules in the dataframe using VAE encoder.
    Args:
        df (pd.DataFrame): dataframe containing 'fps' column with Klekota&Roth fingerprints
            in the form of a list of integers (dense representation)
        model (EncoderDecoderV3): model to be used for encoding
        device (torch.device): device to be used for encoding
    Returns:
        mus (np.ndarray): array of means of the latent space
        logvars (np.ndarray): array of logvars of the latent space
    """
    dataset = VAEDataset(df, fp_len=model.fp_size)
    dataloader = D.DataLoader(dataset, batch_size=1024, shuffle=False)
    mus = []
    logvars = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            X = batch.to(device)
            mu, logvar = model.encoder(X)
            mus.append(mu.cpu().numpy())
            logvars.append(logvar.cpu().numpy())

        mus = np.concatenate(mus, axis=0)
        logvars = np.concatenate(logvars, axis=0)
    return mus, logvars


def evaluate(model, test_X, test_y):
    predictions = model.predict_proba(test_X)[:, 1]
    df = pd.DataFrame()
    df['pred'] = predictions
    df['label'] = test_y.values
    df['pred'] = df['pred'].apply(lambda x: 1 if x > 0.5 else 0)
    accuracy = df[df['pred'] == df['label']].shape[0] / df.shape[0]
    roc_auc = roc_auc_score(df['label'], df['pred'])
    tn, fp, fn, tp = confusion_matrix(df['label'], df['pred']).ravel()
    metrics = {
        'accuracy': round(accuracy, 4),
        'roc_auc': round(roc_auc, 4),
        'true_positive': round(tp / df.shape[0], 4),
        'true_negative': round(tn / df.shape[0], 4),
        'false_positive': round(fp / df.shape[0], 4),
        'false_negative': round(fn / df.shape[0], 4)
    }
    return metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='config_files/SVC_config.ini',
                        help='Path to SVC config file')
    args = parser.parse_args()
    config_path = args.config
    main(config_path)
