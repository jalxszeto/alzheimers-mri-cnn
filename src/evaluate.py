"""Evaluate a trained model on the held-out test set."""
import torch


def evaluate(test_loader, model, num_classes):
    """Return overall accuracy, per-class accuracy (both in %) and the confusion matrix.

    confusion[i][j] counts test images of true class i predicted as class j.
    """
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            predicted = model(X_batch).argmax(dim=1)
            for true, pred in zip(y_batch, predicted):
                confusion[true, pred] += 1

    correct = confusion.diag()
    accuracy = 100 * correct.sum().item() / confusion.sum().item()
    per_class = (100 * correct.float() / confusion.sum(dim=1).clamp(min=1)).tolist()
    return accuracy, per_class, confusion
