"""Generate the figures in assets/: an augmentation example and the model architecture diagram.

Run from the repository root:
    python scripts/visualize.py augmentation data/train/MildDemented/mildDem15.jpg
    python scripts/visualize.py architecture
    python scripts/visualize.py onnx
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import IMAGE_SIZE, apply_transforms  # noqa: E402
from src.model import CNNModel  # noqa: E402

ASSETS_DIR = "assets"


def save_augmentation_example(image_path):
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    Image.fromarray(resized).save(os.path.join(ASSETS_DIR, "sample_resized.jpeg"))

    tensor = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1)
    augmented = apply_transforms(tensor).permute(1, 2, 0).clamp(0, 255)
    Image.fromarray(augmented.numpy().astype(np.uint8)).save(os.path.join(ASSETS_DIR, "sample_augmented.jpeg"))


def save_architecture_diagram():
    from torchview import draw_graph

    model_graph = draw_graph(CNNModel([IMAGE_SIZE, IMAGE_SIZE], 4), input_size=(32, 3, IMAGE_SIZE, IMAGE_SIZE),
                             graph_dir="TD", depth=10, expand_nested=True, hide_module_functions=True, show_shapes=False)
    model_graph.visual_graph.render(filename=os.path.join(ASSETS_DIR, "model_architecture"), format="svg", cleanup=True)


def export_onnx(path="models/cnn.onnx"):
    model = CNNModel([IMAGE_SIZE, IMAGE_SIZE], 4).eval()
    dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(model, dummy, path, input_names=["image"], output_names=["logits"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("figure", choices=["augmentation", "architecture", "onnx"])
    parser.add_argument("image", nargs="?", default="data/train/MildDemented/mildDem15.jpg")
    args = parser.parse_args()

    if args.figure == "augmentation":
        save_augmentation_example(args.image)
    elif args.figure == "architecture":
        save_architecture_diagram()
    else:
        export_onnx()
