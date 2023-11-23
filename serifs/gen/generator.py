import random

import torch
import torch.nn as nn


class VAEEncoder(nn.Module):
    """
    Encoder net, part of VAE.
    Parameters:
        input_size (int): size of the fingerprint vector
        output_size (int): size of the latent vectors mu and logvar
        fc1_size (int): size of the first fully connected layer
        fc2_size (int): size of the second fully connected layer
        fc3_size (int): size of the third fully connected layer
        activation (str): activation function ('relu', 'elu', 'gelu' or 'leaky_relu')
    """

    def __init__(
        self, input_size, output_size, fc1_size, fc2_size, fc3_size, activation="relu"
    ):
        super(VAEEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.fc41 = nn.Linear(fc3_size, output_size)
        self.fc42 = nn.Linear(fc3_size, output_size)
        if activation == "relu":
            self.relu = nn.ReLU()
        elif activation == "leaky_relu":
            self.relu = nn.LeakyReLU()
        elif activation == "elu":
            self.relu = nn.ELU()
        elif activation == "gelu":
            self.relu = nn.GELU()
        else:
            raise ValueError("Activation must be one of: relu, leaky_relu, elu, gelu")

    def forward(self, x):
        """
        Args:
            x (torch.tensor): fingerprint vector
        Returns:
            mu (torch.tensor): mean
            logvar: (torch.tensor): log variance
        """
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))
        mu = self.fc41(h3)
        logvar = self.fc42(h3)
        return mu, logvar

    @staticmethod
    def kld_loss(mu, logvar):
        kld = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0
        )
        return kld


class GRUDecoder(nn.Module):
    """
    Decoder class based on GRU.

    Parameters:
        hidden_size (int): GRU hidden size
        num_layers (int): GRU number of layers
        output_size (int): GRU output size (alphabet size)
        dropout (float): GRU dropout
    """

    def __init__(
        self,
        hidden_size,
        num_layers,
        output_size,
        dropout,
        input_size,
        encoding_size,
        teacher_ratio,
        device,
    ):
        super(GRUDecoder, self).__init__()

        # GRU parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_size = input_size
        self.device = device
        self.teacher_ratio = teacher_ratio
        self.encoding_size = encoding_size
        self.output_size = output_size

        # start token initialization
        self.start_ohe = torch.zeros(output_size, dtype=torch.float32)
        self.start_ohe[output_size - 1] = 1.0  # start token

        # pytorch.nn
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
        )
        self.fc1 = nn.Linear(self.encoding_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, latent_vector, y_true=None, teacher_forcing=False):
        """
        Args:
            latent_vector (torch.tensor): latent vector of size [batch_size, encoding_size]
            y_true (torch.tensor): batched SELFIES of target molecules
            teacher_forcing: (bool): whether to use teacher forcing (training only)

        Returns:
            out (torch.tensor): GRU output of size [batch_size, seq_len, alphabet_size]
        """
        batch_size = latent_vector.shape[0]

        # matching GRU hidden state shape
        latent_transformed = self.fc1(latent_vector)  # shape (batch_size, hidden_size)

        # initializing hidden state
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(
            self.device
        )
        hidden[0] = latent_transformed.unsqueeze(0)
        hidden[1] = latent_transformed.unsqueeze(0)
        # shape (num_layers, batch_size, hidden_size)

        # initializing input (batched start token)
        x = (
            self.start_ohe.repeat(batch_size, 1).unsqueeze(1).to(self.device)
        )  # shape (batch_size, 1, 42)

        # generating sequence
        outputs = []
        for n in range(128):
            out, hidden = self.gru(x, hidden)
            out = self.fc2(out)  # shape (batch_size, 1, 31)
            outputs.append(out)
            out = self.softmax(out)
            random_float = random.random()
            if (
                teacher_forcing
                and random_float < self.teacher_ratio
                and y_true is not None
            ):
                out = y_true[:, n, :].unsqueeze(1)  # shape (batch_size, 1, 31)
            x = out
        out_cat = torch.cat(outputs, dim=1)
        return out_cat


class EncoderDecoderV3(nn.Module):
    """
    Encoder-Decoder class based on VAE and GRU. The samples from VAE latent space are passed
    to the GRU decoder as initial hidden state.

    Parameters:
        fp_size (int): size of the fingerprint vector
        encoding_size (int): size of the latent vectors mu and logvar
        hidden_size (int): GRU hidden size
        num_layers (int): GRU number of layers
        output_size (int): GRU output size (alphabet size)
        dropout (float): GRU dropout
        teacher_ratio (float): teacher forcing ratio
        random_seed (int): random seed for reproducibility
        use_cuda (bool): whether to use cuda
        fc1_size (int): size of the first fully connected layer in the encoder
        fc2_size (int): size of the second fully connected layer in the encoder
        fc3_size (int): size of the third fully connected layer in the encoder
        encoder_activation (str): activation function for the encoder ('relu', 'elu', 'gelu' or 'leaky_relu')
    """

    def __init__(
        self,
        fp_size,
        encoding_size,
        hidden_size,
        num_layers,
        output_size,
        dropout,
        teacher_ratio,
        random_seed=42,
        use_cuda=True,
        fc1_size=2048,
        fc2_size=1024,
        fc3_size=512,
        encoder_activation="relu",
    ):
        super(EncoderDecoderV3, self).__init__()
        self.fp_size = fp_size
        self.encoder = VAEEncoder(
            fp_size, encoding_size, fc1_size, fc2_size, fc3_size, encoder_activation
        )
        self.decoder = GRUDecoder(
            hidden_size,
            num_layers,
            output_size,
            dropout,
            input_size=output_size,
            teacher_ratio=teacher_ratio,
            encoding_size=encoding_size,
            device=torch.device(
                "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
            ),
        )
        random.seed(random_seed)

    def forward(self, X, y, teacher_forcing=False, omit_encoder=False):
        """
        Args:
            X (torch.tensor): batched fingerprint vector of size [batch_size, fp_size]
            y (torch.tensor): batched SELFIES of target molecules
            teacher_forcing: (bool): whether to use teacher forcing
            omit_encoder (bool): if true, the encoder is omitted and the input is passed directly to the decoder

        Returns:
            decoded (torch.tensor): batched prediction tensor [batch_size, seq_len, alphabet_size]
            kld_loss (torch.tensor): KL divergence loss
        """
        if omit_encoder:
            encoded = X
            kld_loss = torch.tensor(0.0)
        else:
            mu, logvar = self.encoder(X)
            kld_loss = self.encoder.kld_loss(mu, logvar)
            encoded = self.reparameterize(
                mu, logvar
            )  # shape (batch_size, encoding_size)

        decoded = self.decoder(
            latent_vector=encoded, y_true=y, teacher_forcing=teacher_forcing
        )
        # shape (batch_size, selfie_len, alphabet_len)

        return decoded, kld_loss  # out_cat.shape (batch_size, selfie_len, alphabet_len)

    @staticmethod
    def reparameterize(mu, logvar):
        """
        Reparametrization trick for sampling from VAE latent space.
        Args:
            mu (torch.tensor): mean
            logvar: (torch.tensor): log variance
        Returns:
            z (torch.tensor): latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
