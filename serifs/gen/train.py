import time

import pandas as pd
import rdkit.Chem as Chem
import rdkit.Chem.QED as QED
import selfies as sf
import torch

from serifs.gen.loss import CCE
from serifs.utils.annealing import Annealer
from serifs.utils.vectorizer import SELFIESVectorizer


def train(config, model, train_loader, val_loader, scoring_loader):
    """
    Training loop for the model consisting of a VAE encoder and GRU decoder
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    epochs = int(config['RUN']['epochs'])
    run_name = str(config['RUN']['run_name'])
    learn_rate = float(config['RUN']['learn_rate'])
    kld_backward = config.getboolean('RUN', 'kld_backward')
    start_epoch = int(config['RUN']['start_epoch'])
    kld_weight = float(config['RUN']['kld_weight'])
    kld_annealing = config.getboolean('RUN', 'kld_annealing')
    annealing_max_epoch = int(config['RUN']['annealing_max_epoch'])
    annealing_shape = str(config['RUN']['annealing_shape'])

    annealing_agent = Annealer(annealing_max_epoch, annealing_shape)

    # Define dataframe for logging progress
    epochs_range = range(start_epoch, epochs + start_epoch)
    metrics = pd.DataFrame(columns=['epoch',
                                    'kld_loss',
                                    'kld_weighted',
                                    'train_loss',
                                    'val_loss',
                                    'mean_qed',
                                    'mean_fp_recon'])

    # Define loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    criterion = CCE()

    print("Starting Training of GRU")
    print(f"Device: {device}")

    # Start training loop
    for epoch in epochs_range:
        model.train()
        start_time = time.time()
        print(f'Epoch: {epoch}')
        epoch_loss = 0
        kld_loss = 0
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output, kld_loss = model(X, y, teacher_forcing=True)
            loss = criterion(y, output)
            kld_weighted = kld_loss * kld_weight
            if kld_annealing:
                kld_weighted = annealing_agent(kld_weighted)
            if kld_backward:
                (loss + kld_weighted).backward()
            else:
                loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        val_loss = evaluate(model, val_loader)

        if epoch % 10 == 0:
            start = time.time()
            mean_qed, mean_fp_recon = get_scores(model, scoring_loader)
            end = time.time()
            print(f'QED + fp evaluated in {(end - start) / 60} minutes')
        else:
            mean_qed = None
            mean_fp_recon = None

        metrics_dict = {'epoch': epoch,
                        'kld_loss': kld_loss.item(),
                        'kld_weighted': kld_weighted.item(),
                        'train_loss': avg_loss,
                        'val_loss': val_loss,
                        'mean_qed': mean_qed,
                        'mean_fp_recon': mean_fp_recon
                        }

        if kld_annealing:
            annealing_agent.step()

        # Update metrics df
        metrics.loc[len(metrics)] = metrics_dict
        if epoch % 10 == 0:
            save_path = f"./models/{run_name}/epoch_{epoch}.pt"
            torch.save(model.state_dict(), save_path)

        metrics.to_csv(f"./models/{run_name}/metrics.csv", index=False)
        end_time = time.time()
        loop_time = (end_time - start_time) / 60  # in minutes
        print(f'Epoch {epoch} completed in {loop_time} minutes')

    return None


def evaluate(model, val_loader):
    """
    Evaluates the model on the validation set
    Args:
        model (nn.Module): EncoderDecoderV3 model
        val_loader (DataLoader): validation set loader
    Returns:
        float: average loss on the validation set

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        criterion = CCE()
        epoch_loss = 0
        for batch_idx, (X, y) in enumerate(val_loader):
            X = X.to(device)
            y = y.to(device)
            output, _ = model(X, y, teacher_forcing=False)
            loss = criterion(y, output)
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(val_loader)
        return avg_loss


def get_scores(model, scoring_loader):
    """
    Calculates the QED and FP reconstruction score for the model
    Args:
        model (nn.Module): EncoderDecoderV3 model
        scoring_loader (DataLoader): scoring set loader

    Returns:
        mean_qed (float): average QED score
        mean_fp_recon (float): average FP reconstruction score

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vectorizer = SELFIESVectorizer(pad_to_len=128)
    model.eval()
    with torch.no_grad():
        mean_qed = 0
        mean_fp_recon = 0
        for batch_idx, (X, y) in enumerate(scoring_loader):
            X = X.to(device)
            y = y.to(device)
            output, _ = model(X, y, teacher_forcing=False)
            selfies_list = [vectorizer.devectorize(ohe.detach().cpu().numpy(),
                                                   remove_special=True) for ohe in output]
            smiles_list = [sf.decoder(x) for x in selfies_list]
            mol_list = [Chem.MolFromSmiles(x) for x in smiles_list]

            # Calculate QED
            batch_qed = 0
            for mol in mol_list:
                batch_qed += QED.qed(mol)
            batch_qed = batch_qed / len(mol_list)
            mean_qed += batch_qed

            # Calculate FP recon score
            batch_fp_recon = 0
            for mol, x in zip(mol_list, X):
                fp = x.detach().cpu()
                batch_fp_recon += fp_score(mol, fp)
            batch_fp_recon = batch_fp_recon / len(mol_list)
            mean_fp_recon += batch_fp_recon

        mean_fp_recon = mean_fp_recon / len(scoring_loader)
        mean_qed = mean_qed / len(scoring_loader)
        return mean_qed, mean_fp_recon


def fp_score(mol, fp: torch.Tensor):
    """
    Calculates the fingerprint reconstruction score for a molecule
    Args:
        mol: rdkit mol object
        fp: torch tensor of size (fp_len)
    Returns:
        score: float (0-1)
    """
    score = 0
    key = pd.read_csv('data/KlekFP_keys.txt', header=None)
    fp_len = fp.shape[0]
    for i in range(fp_len):
        if fp[i] == 1:
            frag = Chem.MolFromSmarts(key.iloc[i].values[0])
            score += mol.HasSubstructMatch(frag)
    return score / torch.sum(fp).item()
