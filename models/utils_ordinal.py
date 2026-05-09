import torch
import torch.nn.functional as F


def build_threshold_targets(targets: torch.Tensor, num_thresholds: int) -> torch.Tensor:
    """Return cumulative binary targets where column k indicates y > k."""
    thresholds = torch.arange(num_thresholds, device=targets.device)
    return (targets.unsqueeze(1) > thresholds).float()


def ordinal_logistic_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Binary cross entropy on cumulative logits for ordinal regression."""
    target_matrix = build_threshold_targets(targets, logits.size(1))
    return F.binary_cross_entropy_with_logits(logits, target_matrix)


def cumulative_to_labels(probs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Convert cumulative probabilities to discrete labels."""
    return (probs > threshold).sum(dim=1)


def threshold_accuracy(probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Compute accuracy for each ordinal threshold."""
    target_matrix = build_threshold_targets(targets, probs.size(1))
    preds = probs > threshold
    correct = (preds == target_matrix.bool()).float().sum(dim=0)
    counts = target_matrix.size(0)
    return correct / counts
