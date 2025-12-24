import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
def train_dual_input_model(model, train_loader, val_loader, num_epochs=300, lr=0.0005, 
                           weight_decay=5e-3, l1_lambda=5e-3, patience=25, 
                           print_every=10, save_plot=True, plot_filename='training_metrics.png'):
    """
    Train a dual-input neural network model with early stopping and comprehensive metrics tracking.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The neural network model to train
    train_loader : DataLoader
        DataLoader for training data (should yield vi_batch, binder_batch, y_batch)
    val_loader : DataLoader
        DataLoader for validation data
    num_epochs : int, default=300
        Maximum number of training epochs
    lr : float, default=0.0005
        Learning rate for Adam optimizer
    weight_decay : float, default=5e-3
        L2 regularization weight decay
    l1_lambda : float, default=5e-3
        L1 regularization lambda
    patience : int, default=25
        Number of epochs to wait for improvement before early stopping
    print_every : int, default=10
        Print metrics every N epochs
    save_plot : bool, default=True
        Whether to save training metrics plot
    plot_filename : str, default='training_metrics.png'
        Filename for saved plot
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'model': Trained model with best weights loaded
        - 'train_losses': List of training losses per epoch
        - 'val_losses': List of validation losses per epoch
        - 'train_maes': List of training MAE per epoch
        - 'val_maes': List of validation MAE per epoch
        - 'train_r2s': List of training R² scores per epoch
        - 'val_r2s': List of validation R² scores per epoch
        - 'best_val_loss': Best validation loss achieved
        - 'best_epoch': Epoch where best validation loss was achieved
    """

    
    # Setup training components
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Initialize tracking lists
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    train_r2s = []
    val_r2s = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        train_preds_list = []
        train_targets_list = []
        
        for vi_batch, binder_batch, y_batch in train_loader:
            predictions = model(vi_batch, binder_batch)
            mse_loss = criterion(predictions, y_batch)
            
            # L1 regularization
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = mse_loss + l1_lambda * l1_norm
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
            # Store predictions for metrics
            train_preds_list.append(predictions.detach().numpy())
            train_targets_list.append(y_batch.numpy())
        
        # Calculate training metrics
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        train_preds = np.concatenate(train_preds_list).flatten()
        train_targets = np.concatenate(train_targets_list).flatten()
        train_mae = mean_absolute_error(train_targets, train_preds)
        train_r2 = r2_score(train_targets, train_preds)
        train_maes.append(train_mae)
        train_r2s.append(train_r2)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0
        val_preds_list = []
        val_targets_list = []
        
        with torch.no_grad():
            for vi_batch, binder_batch, y_batch in val_loader:
                predictions = model(vi_batch, binder_batch)
                loss = criterion(predictions, y_batch)
                epoch_val_loss += loss.item()
                
                # Store predictions for metrics
                val_preds_list.append(predictions.numpy())
                val_targets_list.append(y_batch.numpy())
        
        # Calculate validation metrics
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        val_preds = np.concatenate(val_preds_list).flatten()
        val_targets = np.concatenate(val_targets_list).flatten()
        val_mae = mean_absolute_error(val_targets, val_preds)
        val_r2 = r2_score(val_targets, val_preds)
        val_maes.append(val_mae)
        val_r2s.append(val_r2)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            if (epoch + 1) % print_every == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}, Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f} ✓ (Best)')
        else:
            patience_counter += 1
            if (epoch + 1) % print_every == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}, Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f} (Patience: {patience_counter}/{patience})')
            
            if patience_counter >= patience:
                print(f'\nEarly stopping at epoch {epoch+1}')
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    print(f'\nBest validation loss: {best_val_loss:.4f} at epoch {best_epoch}')
    
    # Plot training metrics
    if save_plot:
        plot_training_metrics(train_losses, val_losses, train_maes, val_maes, 
                            train_r2s, val_r2s, best_val_loss, filename=plot_filename)
    
    # Print final metrics
    print(f"\nFinal Metrics:")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Final Train MAE: {train_maes[-1]:.4f}")
    print(f"Final Val MAE: {val_maes[-1]:.4f}")
    print(f"Final Train R²: {train_r2s[-1]:.4f}")
    print(f"Final Val R²: {val_r2s[-1]:.4f}")
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_maes': train_maes,
        'val_maes': val_maes,
        'train_r2s': train_r2s,
        'val_r2s': val_r2s,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch
    }


def plot_training_metrics(train_losses, val_losses, train_maes, val_maes, 
                         train_r2s, val_r2s, best_val_loss, filename='training_metrics.png'):
    """
    Create comprehensive training metrics visualization.
    
    Parameters:
    -----------
    train_losses : list
        Training losses per epoch
    val_losses : list
        Validation losses per epoch
    train_maes : list
        Training MAE per epoch
    val_maes : list
        Validation MAE per epoch
    train_r2s : list
        Training R² scores per epoch
    val_r2s : list
        Validation R² scores per epoch
    best_val_loss : float
        Best validation loss achieved
    filename : str, default='training_metrics.png'
        Filename to save the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Training and Validation Loss
    axes[0, 0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0, 0].plot(val_losses, label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('MSE Loss', fontsize=12)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: MAE (Mean Absolute Error)
    axes[0, 1].plot(train_maes, label='Train MAE', linewidth=2, color='orange')
    axes[0, 1].plot(val_maes, label='Validation MAE', linewidth=2, color='red')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('MAE', fontsize=12)
    axes[0, 1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: R² Score
    axes[1, 0].plot(train_r2s, label='Train R²', linewidth=2, color='green')
    axes[1, 0].plot(val_r2s, label='Validation R²', linewidth=2, color='darkgreen')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('R² Score', fontsize=12)
    axes[1, 0].set_title('R² Score (Coefficient of Determination)', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Plot 4: Loss Difference (Overfitting indicator)
    loss_diff = np.array(val_losses) - np.array(train_losses)
    axes[1, 1].plot(loss_diff, linewidth=2, color='purple')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Val Loss - Train Loss', fontsize=12)
    axes[1, 1].set_title('Overfitting Monitor (Val - Train Loss)', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlot saved as '{filename}'")
