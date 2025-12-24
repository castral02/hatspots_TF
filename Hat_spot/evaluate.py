import torch.nn as nn
import torch
import pandas as pd

import torch
import pandas as pd
import numpy as np
import joblib

def get_predictions(model, X_vi_tensor, X_binder_tensor, y_true, original_data):
    """
    Get predictions from the model and return them with original data.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained model
    X_vi_tensor : torch.Tensor
        Variant input tensor
    X_binder_tensor : torch.Tensor
        Binder input tensor
    y_true : np.array
        True log-transformed Kd values
    original_data : pd.DataFrame
        Original dataframe with Variant_ID and Binder columns
        
    Returns:
    --------
    pd.DataFrame with predictions and original values
    """
    # Load the scaler
    scaler_y = joblib.load('scaler_y.pkl')
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Get scaled predictions
        predictions_scaled = model(X_vi_tensor, X_binder_tensor)
        
        # Convert to numpy and inverse transform
        predictions_scaled_np = predictions_scaled.numpy()
        predictions_log = scaler_y.inverse_transform(predictions_scaled_np).flatten()
        
        # Convert from log space to original Kd values
        predictions_original = np.exp(predictions_log)
    
    # Create results dataframe
    results = pd.DataFrame({
        'Variant_ID': original_data['Variant_ID'].values,
        'Binder': original_data['Binder'].values,
        'Kd': y_true,  # Log-transformed true values
        'Predicted_log': predictions_log,  # Log-transformed predictions
        'Predicted': predictions_original  # Original scale predictions
    })
    
    return results
