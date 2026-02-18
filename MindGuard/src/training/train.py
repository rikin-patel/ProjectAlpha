"""Training script for model with validation, LR scheduling, early stoppint, and checkpointing."""
import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm

from src.data.dataloader import get_dataloaders
from src.models.model import get_model
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger('train')

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_scheduler(optimizer: optim.Optimizer, config: dict) :
    """Get learning rate scheduler based on config."""
    if config['scheduler'] == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=config['epochs'])
    elif config['scheduler'] == 'step':
        return StepLR(optimizer, step_size=config.get('step_size', 7), gamma=config.get('gamma', 0.1))
    else:
        raise ValueError(f"Unsupported scheduler type: {config['scheduler']}")
    
def validate(model: nn.Module, val_loader, criterion: nn.Module, device: torch.device) -> float:
    """Validate the model and return average loss.
    
    Args:
        model: The model to validate.
        val_loader: The validation data loader.
        criterion: The loss function.
        device: The device to run validation on.

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return total_loss / total, 100.0 * correct / total

def train(config_path: str = 'src/training/config.yaml') -> None:
    """Main training loop with validation, checkpointing, and early stopping."""
    config = load_config(config_path)
    set_seed(config.get('seed', 42))
    logger.info(f"Seed set to {config.get('seed', 42)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    #loading data into dataloaders
    train_loader, val_loader = get_dataloaders(config['data_dir'], 
        config['batch_size'], config['img_size'], config['num_workers'])
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = get_model(config['num_classes'],
                      model_name=config.get('model_name', 'resnet18'),
                      pretrained=config['pretrained']).to(device)
    logger.info(f"Model {config.get('model_name', 'resnet18')} initialized with classes= {config['num_classes']}, pretrained={config['pretrained']}")
    
    #Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'],
                           weight_decay=config.get('weight_decay', 1e-4))
    scheduler = get_scheduler(optimizer, config)

    #Checkpointing and early stopping variables
    save_dir = os.path.dirname(config['model_save_path'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        logger.info(f"Created checkpoint directory: {save_dir}")

    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience = config.get('early_stopping_patience', 5)
    patience_counter = 0

    for epoch in range(config['epochs']):
        # ==== Training Phase ====
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}', unit='batch')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({'loss': loss.item(), 'acc': 100.0 * correct / total})
            
        avg_train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # ==== Validation Phase ====
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        logger.info(f'Epoch [{epoch+1}/{config["epochs"]}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # ===Scheduler step and checkpointing===
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f'Learning rate adjusted to {current_lr:.6f}')
        else:
            current_lr = config['learning_rate']
            logger.info('No scheduler used, skipping learning rate adjustment.')

        logger.info(f"Epoch {epoch+1}/{config['epochs']} completed. | "
                     f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, | "
                     f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, | "            
                     f'Current learning rate: {current_lr:.6f}')

        # === Checkpointing ===
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0  # reset counter if we get a new best accuracy
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': config
            }, config['model_save_path'])
            logger.info(f'New best model saved with val acc: {val_acc:.2f}%')
        else:
            patience_counter += 1
            logger.info(f'No improvement in val acc. Patience counter: {patience_counter}/{patience}')
        
        # ==== Early stopping check ====    
        if patience_counter >= patience:
                logger.info(f'Early stopping triggered after {epoch+1} epochs.')
                break


if __name__ == "__main__":
    train()