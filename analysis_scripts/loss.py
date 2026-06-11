import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """
    logits:  Tensor of shape [B, C] (B: number of batches, C: number of classes)
    targets: Tensor of shape [B]
    """

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)


class WeightedCrossEntropyLoss(nn.Module):
    """
    logits:  Tensor of shape [B, C] (B: number of batches, C: number of classes)
    targets: Tensor of shape [B]

    class_weights should be a tensor/list of shape [C]
    """

    def __init__(self, class_weights):
        super().__init__()
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer("class_weights", class_weights) # class_weights is not a trainable parameter
        # (note) buffer: part of the model's state, not trainable, moved with the model, saved in state_dict()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
        )


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """
    logits:  Tensor of shape [B, C] (B: number of batches, C: number of classes)
    targets: Tensor of shape [B]

    smoothing:
        0.0 means normal Cross-Entropy.
        Common values: 0.05, 0.1
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            targets,
            label_smoothing=self.smoothing,
        )


class FocalLoss(nn.Module):
    """
    logits:  Tensor of shape [B, C] (B: number of batches, C: number of classes)
    targets: Tensor of shape [B]

    Formula:
        FL = -alpha * (1 - p_t)^gamma * log(p_t)

    gamma:
        Higher gamma focuses more on hard examples.
        Common value: 2.0

    alpha:
        Optional class weights of shape [C].
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha=None
    ):
        super().__init__()
        self.gamma = gamma

        if alpha is not None:
            alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        targets = targets.long()

        # Get probability of the true class
        batch_indices = torch.arange(targets.size(0), device=log_probs.device)
        target_log_probs = log_probs[batch_indices, targets]
        target_probs = probs[batch_indices, targets]

        focal_factor = (1.0 - target_probs) ** self.gamma
        loss = -focal_factor * target_log_probs

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        return loss.mean()


class ClassBalancedLoss(nn.Module):
    """
    logits:  Tensor of shape [B, C] (B: number of batches, C: number of classes)
    targets: Tensor of shape [B]

    Formula:
        weight = (1 - beta) / (1 - beta^n)

    samples_per_class:
        Number of training samples in each class.
        Example:
            samples_per_class = [500, 100, 20]

    beta:
        Common values:
            0.9, 0.99, 0.999, 0.9999

        For larger datasets, use beta closer to 1.
    """

    def __init__(
        self,
        samples_per_class,
        beta: float = 0.9999
    ):
        super().__init__()

        samples_per_class = torch.tensor(samples_per_class, dtype=torch.float32)

        class_weights = (1.0 - beta) / (1.0 - torch.pow(beta, samples_per_class))

        # Normalize weights so average weight is around 1
        class_weights = class_weights / class_weights.sum() * len(samples_per_class) # shape [C]

        self.register_buffer("class_weights", class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights
        )