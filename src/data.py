"""Loading, labelling, splitting and augmenting the MRI image dataset."""
import os
import random

import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

# Folder names under data/{train,test}/. The index of each name is its label.
CLASS_NAMES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
NON_DEMENTED_INDEX = CLASS_NAMES.index("NonDemented")
IMAGE_SIZE = 224


def class_names(binary=False):
    return ["NonDemented", "Demented"] if binary else CLASS_NAMES


def class_dirs(split_dir):
    return [os.path.join(split_dir, name) for name in CLASS_NAMES]


def read_and_resize_images(folder, dim=IMAGE_SIZE):
    """Read every image in `folder` as a float32 CxHxW tensor of size dim x dim."""
    image_list = []
    for image_file in sorted(os.listdir(folder)):
        if image_file.startswith("."):
            continue
        image = cv2.imread(os.path.join(folder, image_file))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(image_rgb, (dim, dim))
        resized_image = torch.tensor(resized_image, dtype=torch.float32)
        image_list.append(resized_image.permute(2, 0, 1))
    return image_list


# Augmentations: each one is applied independently with the given probability
def random_rotation(image, probability=0.75, min_degree=3, max_degree=6):
    if random.random() < probability:
        degrees = random.uniform(min_degree, max_degree)
        return transforms.RandomRotation(degrees=degrees)(image)
    return image


def random_horizontal_flip(image, probability=0.75):
    if random.random() < probability:
        return transforms.RandomHorizontalFlip()(image)
    return image


def random_affine(image, probability=0.75, translate=(0.03, 0.03), scale=(0.95, 1.05), shear=8):
    if random.random() < probability:
        return transforms.RandomAffine(degrees=0, translate=translate, scale=scale, shear=shear)(image)
    return image


def random_color_jitter(image, probability=0.75, contrast=(0.8, 1), brightness=(0.8, 1)):
    if random.random() < probability:
        return transforms.ColorJitter(contrast=contrast, brightness=brightness)(image)
    return image


def apply_transforms(image):
    image = random_rotation(image)
    image = random_horizontal_flip(image)
    image = random_affine(image)
    # ColorJitter expects values in [0, 1]
    image = random_color_jitter(image / 255) * 255
    return image


def augment_images(pairs, proportion_augmented=0.1):
    """Append augmented copies of randomly chosen original (image, label) pairs."""
    num_original = len(pairs)
    for _ in range(int(num_original * proportion_augmented)):
        image, label = pairs[random.randrange(num_original)]
        pairs.append((apply_transforms(image), label))
    return pairs


def show_samples(X, y, binary=False):
    """Plot a 2x2 grid of images with their class names."""
    names = class_names(binary)
    fig = plt.figure(figsize=(10, 10))
    for i in range(4):
        fig.add_subplot(2, 2, i + 1)
        plt.imshow(X[i].permute(1, 2, 0).numpy().astype("uint8"))
        plt.title(names[int(y[i])])
    plt.show()


def create_labels(images, binary=False):
    """`images` is a list of per-class image lists, in CLASS_NAMES order."""
    labels = []
    for index, class_images in enumerate(images):
        label = int(index != NON_DEMENTED_INDEX) if binary else index
        labels.extend([torch.tensor([label], dtype=torch.int64)] * len(class_images))
    return labels


def split_val(dataset, split_ratio=0.8):
    random.shuffle(dataset)
    split_index = int(len(dataset) * split_ratio)
    return dataset[:split_index], dataset[split_index:]


def _stack(pairs):
    X, y = zip(*pairs)
    return torch.stack(X), torch.stack(y).squeeze(1)


def create_train_data(filepaths, binary=False):
    """Build augmented train and val tensors plus inverse-frequency class weights."""
    images = [read_and_resize_images(path) for path in filepaths]

    class_sizes = [len(class_images) for class_images in images]
    if binary:
        demented = sum(size for i, size in enumerate(class_sizes) if i != NON_DEMENTED_INDEX)
        class_sizes = [class_sizes[NON_DEMENTED_INDEX], demented]
    class_weights = 1. / torch.Tensor(class_sizes)
    class_weights = class_weights / class_weights.sum()

    labels = create_labels(images, binary)
    flat_images = [image for class_images in images for image in class_images]

    pairs_train, pairs_val = split_val(list(zip(flat_images, labels)))
    # Only the training split is augmented so validation reflects real images
    pairs_train = augment_images(pairs_train)

    X_train, y_train = _stack(pairs_train)
    X_val, y_val = _stack(pairs_val)
    return X_train, y_train, X_val, y_val, class_weights


def create_test_data(filepaths, binary=False):
    images = [read_and_resize_images(path) for path in filepaths]
    X_test = torch.stack([image for class_images in images for image in class_images])
    y_test = torch.stack(create_labels(images, binary)).squeeze(1)
    return X_test, y_test
