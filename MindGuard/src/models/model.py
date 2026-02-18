"""Model factory for Image Classification"""
import torch.nn as nn
import torchvision.models as models

from torchvision.models import (
    ResNet18_Weights, 
    ResNet50_Weights, 
    EfficientNet_B0_Weights
)

MODEL_REGISTRY = {
    'resnet18': (models.resnet18, ResNet18_Weights.DEFAULT),
    'resnet50': (models.resnet50, ResNet50_Weights.DEFAULT),
    'efficientnet_b0': (models.efficientnet_b0, EfficientNet_B0_Weights.DEFAULT)
}

def get_model(num_classes: int, model_name: str = 'resnet18', pretrained=True) -> nn.Module:
    """Get a model instance by name and number of classes with specified architecture.
    
    Args:
        num_classes (int): Number of output classes.
        model_name (str): Name of the model to retrieve.
        pretrained (bool): Whether to load pretrained weights.

    Returns:
        torch.nn.Module: The model instance ready for training / inference.

    Raises:
        KeyError: If the model name is not in the registry.
    """
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Model '{model_name}' not in registry. Choose from: {list(MODEL_REGISTRY.keys())}")

    model_func, weights = MODEL_REGISTRY[model_name]
    
    if pretrained:
        model = model_func(weights=weights)
    else:
        model = model_func(weights=None)

    # Modify the final layer to match num_classes
    if model_name.startswith('resnet'):
        # For ResNet models
        num_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=0.3),
                                 nn.Linear(num_features, num_classes))
    elif model_name.startswith('efficientnet'):
        # For EfficientNet models
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(nn.Dropout(p=0.3),
                                nn.Linear(num_features, num_classes))

    return model