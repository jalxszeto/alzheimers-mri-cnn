"""Saves the best model weights by validation metric and logs them to Weights & Biases."""
import logging
import os

import torch
import wandb


class WandbModelCheckpoint:
    def __init__(self, dirpath, decreasing=False, top_n=1):
        """
        dirpath: Directory path where to store all model weights
        decreasing: If decreasing is `True`, then lower metric is better
        top_n: Total number of models to track based on validation metric value
        """
        os.makedirs(dirpath, exist_ok=True)
        self.dirpath = dirpath
        self.top_n = top_n
        self.decreasing = decreasing
        self.top_model_paths = []
        self.best_metric_val = float("inf") if decreasing else float("-inf")

    def __call__(self, model, epoch, metric_val):
        model_path = os.path.join(self.dirpath, f"{model.__class__.__name__}_{wandb.run.name}_epoch{epoch}.pt")
        save = metric_val < self.best_metric_val if self.decreasing else metric_val > self.best_metric_val
        if save:
            logging.info(f"Current metric value {metric_val} better than best {self.best_metric_val}, saving model at {model_path}, & logging model weights to W&B.")
            self.best_metric_val = metric_val
            torch.save(model.state_dict(), model_path)
            self.log_artifact(f"model-ckpt-epoch-{epoch}.pt", model_path, metric_val)
            self.top_model_paths.append({"path": model_path, "score": metric_val})
            self.top_model_paths = sorted(self.top_model_paths, key=lambda o: o["score"], reverse=not self.decreasing)
        if len(self.top_model_paths) > self.top_n:
            self.cleanup()

    def log_artifact(self, filename, model_path, metric_val):
        artifact = wandb.Artifact(filename, type="model", metadata={"Validation score": metric_val})
        artifact.add_file(model_path)
        wandb.run.log_artifact(artifact)

    def cleanup(self):
        to_remove = self.top_model_paths[self.top_n:]
        logging.info(f"Removing extra models.. {to_remove}")
        for o in to_remove:
            os.remove(o["path"])
        self.top_model_paths = self.top_model_paths[:self.top_n]
