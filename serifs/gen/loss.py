import torch
import torch.nn as nn


class CCE(nn.Module):
    def __init__(self, ignore_index=29):
        """
        Conscious Cross-Entropy. Calculates cross-entropy loss on two SELFIES,
        ignoring indices of padding tokens.
        Parameters:
            ignore_index: index of the padding token to ignore when calculating loss
        """
        super(CCE, self).__init__()
        self.idx_ignore = ignore_index  # index of token to ignore
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, target, predictions):
        """
        Forward pass
        Args:
            target (torch.Tensor): target tensor of shape [batch_size, seq_len, token_idx]
            predictions (torch.Tensor): predictions tensor of shape [batch_size, seq_len, token_idx]
        Returns:
            loss (torch.Tensor): loss value
        """
        # target.shape [batch_size, seq_len, token_idx]
        batch_size = predictions.shape[0]
        one_hot = target.argmax(dim=-1)
        mask = (one_hot != self.idx_ignore)
        weights = (mask.T / mask.sum(axis=1)).T[mask]
        loss = torch.nn.functional.cross_entropy(predictions[mask], one_hot[mask], reduction='none')
        return (weights * loss).sum() / batch_size
