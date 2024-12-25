import torch
import torch.nn as nn

class SparseCategoricalCrossentropy(nn.Module):
    def __init__(self, reduction='mean', ignore_index=None):
        """
        Custom Sparse Categorical Crossentropy implementation with ignore_index support.
        
        Args:
            reduction (str): Specifies the reduction to apply to the output. Options: 'mean', 'sum', or 'none'.
            ignore_index (int, optional): Specifies a target value that should be ignored during loss computation.
        """
        super(SparseCategoricalCrossentropy, self).__init__()
        assert reduction in ['mean', 'sum', 'none'], "Invalid reduction type"
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Forward pass for sparse categorical crossentropy.
        
        Args:
            logits (torch.Tensor): Predicted logits of shape (batch_size, num_classes).
            targets (torch.Tensor): Ground truth labels of shape (batch_size,) with integer class indices.
        
        Returns:
            torch.Tensor: Loss value.
        """
        # Apply LogSoftmax to logits
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Gather log probabilities corresponding to target labels
        valid_mask = targets != self.ignore_index
        selected_log_probs = log_probs[range(len(targets)), targets]
        
        # Mask out invalid indices
        selected_log_probs = selected_log_probs[valid_mask]
        
        # Compute loss (negative log probability of the correct class)
        loss = -selected_log_probs
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean() if len(loss) > 0 else torch.tensor(0.0, device=logits.device)
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
