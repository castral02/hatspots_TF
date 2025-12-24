import esm
import numpy as np
import torch
from tqdm import tqdm
    
def get_esm_embeddings(data_list, batch_size=8, model_name='esm2_t33_650M_UR50D'):
    """
    Generate ESM embeddings for a list of protein sequences.
    
    Parameters:
    -----------
    data_list : list of tuples
        List of (variant_id, sequence) tuples
    batch_size : int, default=8
        Number of sequences to process in each batch
    model_name : str, default='esm2_t33_650M_UR50D'
        ESM model to use for embeddings
        
    Returns:
    --------
    list of dict
        List of dictionaries containing 'Variant_ID' and 'pooled_embedding'
    """
    # Load ESM model
    if model_name == 'esm2_t33_650M_UR50D':
        model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    else:
        raise ValueError(f"Model {model_name} not implemented")
    
    model_esm.eval()  # Set the model to evaluation mode
    batch_converter = alphabet.get_batch_converter()
    
    print('Grabbing embeddings')
    
    # Process in smaller batches
    sequence_embeddings = []
    
    for i in tqdm(range(0, len(data_list), batch_size)):
        batch_data = data_list[i:i+batch_size]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        
        with torch.no_grad():
            results = model_esm(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]
        
        # Process each sequence in the batch
        for j, (variant_id, seq) in enumerate(batch_data):
            seq_len = batch_lens[j] - 2
            full_repr = token_representations[j, 1:seq_len+1]
            pooled_repr = full_repr.mean(0)
            
            sequence_embeddings.append({
                'Variant_ID': variant_id,
                'pooled_embedding': pooled_repr.cpu().numpy()
            })
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return sequence_embeddings
