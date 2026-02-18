"""Data loading and preprocessing for image classification."""
import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """Data augmentation and normalization for training."""
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    """Normalization for validation / inference (No augmentation)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def get_dataloaders(data_dir: str, batch_size: int = 32,
                     img_size: int = 224, num_workers: int = 4) -> tuple:
    """Create dataloaders for training and validation.
    Assumes data directory has 'train' and 'val' subdirectories,
    each with one subfolder per class (ImageFoler convention).

    Args:
        data_dir: Root directory containing 'train' and 'val' folders.
        batch_size: Number of samples per batch.
        img_size: Size to resize images to (img_size x img_size).
        num_workers: Number of subprocesses for data loading.

    Returns:
        Tuple of (train_loader, val_loader) for training and validation.
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    train_dataset = datasets.ImageFolder(train_dir, transform=get_train_transforms(img_size))
    val_dataset = datasets.ImageFolder(val_dir, transform=get_val_transforms(img_size))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,pin_memory=True)

    return train_loader, val_loader

def load_image(image_path: str, img_size: int = 224):
    """Load and preprocess a single image for inference. Return batch-ready tensor.
    Args:
        image_path: Path to the image file.
        img_size: Size to resize the image to (img_size x img_size).

    Returns:
        Preprocessed image tensor with shape (1, 3, img_size, img_size).
    """
    transform = get_val_transforms(img_size)
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension