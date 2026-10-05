"""Command-line entry point: preprocess data, train / sweep with W&B, and evaluate checkpoints.

Examples:
    python main.py preprocess
    python main.py train --lr 5e-5 --batch-size 32 --epochs 50 --optimizer adam
    python main.py sweep --count 30
    python main.py evaluate models/CNNModel_*.pt
"""
import argparse
import functools

import torch
import wandb
from torch.utils.data import DataLoader, TensorDataset

from src import checkpoint as cp
from src.data import IMAGE_SIZE, class_dirs, class_names, create_test_data, create_train_data
from src.evaluate import evaluate
from src.model import CNNModel
from src.train import train

TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
PROCESSED_PATH = "data/processed.pt"
MODELS_DIR = "models"
WANDB_PROJECT = "frproj"

SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val_auc", "goal": "maximize"},
    "parameters": {
        "lr": {"min": 0.00001, "max": 0.0001},
        "batch_size": {"values": [16, 32, 64]},
        "epochs": {"values": [40, 50, 60, 70]},
        "optimizer": {"values": ["adam", "sgd"]},
    },
}


def build_model(binary=False):
    return CNNModel([IMAGE_SIZE, IMAGE_SIZE], len(class_names(binary)))


def preprocess(binary=False):
    """Load, split and augment the training images once and cache them to disk."""
    torch.save(create_train_data(class_dirs(TRAIN_DIR), binary), PROCESSED_PATH)
    print(f"Saved processed training data to {PROCESSED_PATH}")


def run_training(config=None, binary=False):
    """Train one model. `config` is ignored during a sweep, where W&B supplies it."""
    wandb.init(project=WANDB_PROJECT, config=config)
    config = wandb.config

    X_train, y_train, X_val, y_val, class_weights = torch.load(PROCESSED_PATH)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=config.batch_size, shuffle=False)

    train(train_loader, val_loader, build_model(binary), config.lr, config.epochs, config.optimizer,
          class_weights, cp.WandbModelCheckpoint(MODELS_DIR), num_classes=len(class_names(binary)))


def run_evaluation(checkpoints, binary=False):
    names = class_names(binary)
    X_test, y_test = create_test_data(class_dirs(TEST_DIR), binary)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

    model = build_model(binary)
    print(" | ".join(["checkpoint", "overall"] + names))
    for path in checkpoints:
        model.load_state_dict(torch.load(path))
        accuracy, per_class, confusion = evaluate(test_loader, model, len(names))
        print(" | ".join([path, f"{accuracy:.2f}"] + [f"{acc:.2f}" for acc in per_class]))
        print(f"confusion matrix (rows = true, cols = predicted):\n{confusion}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", action="store_true", help="NonDemented vs. Demented instead of 4 classes")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("preprocess", help=f"build and cache the train/val tensors to {PROCESSED_PATH}")

    train_parser = commands.add_parser("train", help="train a single model")
    train_parser.add_argument("--lr", type=float, default=5e-5)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")

    sweep_parser = commands.add_parser("sweep", help="run a Bayesian hyperparameter sweep with W&B")
    sweep_parser.add_argument("--count", type=int, default=30, help="number of sweep runs")

    eval_parser = commands.add_parser("evaluate", help="report test-set accuracy for saved checkpoints")
    eval_parser.add_argument("checkpoints", nargs="+", help="paths to .pt state dicts")

    args = parser.parse_args()
    if args.command == "preprocess":
        preprocess(args.binary)
    elif args.command == "train":
        config = {"lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs, "optimizer": args.optimizer}
        run_training(config, args.binary)
    elif args.command == "sweep":
        sweep_id = wandb.sweep(SWEEP_CONFIG, project=WANDB_PROJECT)
        wandb.agent(sweep_id, function=functools.partial(run_training, binary=args.binary), count=args.count)
    elif args.command == "evaluate":
        run_evaluation(args.checkpoints, args.binary)


if __name__ == "__main__":
    main()
