"""Training and validation loop with class-weighted loss, AUROC/accuracy metrics and W&B logging."""
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch import nn
from torcheval.metrics import MulticlassAccuracy, MulticlassAUROC


def train_epoch(train_loader, optimizer, model, loss_fn, auc, accuracy):
    total_loss = 0.

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)

        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        y_prob = F.softmax(y_pred, dim=1)
        total_loss += loss.item()
        auc.update(y_prob, y_batch)
        accuracy.update(y_prob, y_batch)
        wandb.log({"batch_train_loss": loss.item()})
    return total_loss


def validate_epoch(val_loader, model, loss_fn, auc, accuracy):
    val_loss = 0.

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            val_loss += loss.item()

            y_prob = F.softmax(y_pred, dim=1)
            auc.update(y_prob, y_batch)
            accuracy.update(y_prob, y_batch)
            wandb.log({"batch_val_loss": loss.item()})
    return val_loss


def train(train_loader, val_loader, model, lr, n_epochs, optimizer, class_weights, checkpoint, num_classes=4):
    auc = MulticlassAUROC(num_classes=num_classes)
    accuracy = MulticlassAccuracy(num_classes=num_classes)

    # Weighted loss counteracts the heavy class imbalance (e.g. very few ModerateDemented scans)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr, momentum=0.9)

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = train_epoch(train_loader, optimizer, model, loss_fn, auc, accuracy)
        train_auc = auc.compute().item()
        train_acc = accuracy.compute().item()
        auc.reset()
        accuracy.reset()

        val_loss = validate_epoch(val_loader, model, loss_fn, auc, accuracy)
        val_auc = auc.compute().item()
        val_acc = accuracy.compute().item()
        auc.reset()
        accuracy.reset()

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_auc": train_auc,
                "train_acc": train_acc * 100,

                "val_loss": val_loss,
                "val_auc": val_auc,
                "val_acc": val_acc * 100,

                "lr": optimizer.param_groups[0]["lr"]
            }
        )

        checkpoint(model, epoch, val_auc)

        print(f"Epoch {epoch}: Train Acc: {train_acc*100:.2f} Train Loss: {train_loss:.4f} Train AUC: {train_auc:.4f} "
              f"Val Acc: {val_acc*100:.2f} Val Loss: {val_loss:.4f} Val AUC: {val_auc:.4f} LR: {optimizer.param_groups[0]['lr']}")
