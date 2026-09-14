import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm
import math

import torch
import torch.nn as nn
import torch.optim as optim
import threading

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import median_heuristic
from src.classical_utils import biased_sample_mmd, get_sample_covariance_matrix
"""
Training of transformer models.
"""

PATH = str(REPO_ROOT / "data" / "benchmark_datasets" / "molecular") + "/"

CONFIG = dict(
    n_reps = 5, # number of repetitions to average over
    test_length_cutoff = 5, # maximal Z-string length
    num_samples = 1000, # number of test samples for evaluation
    learning_rate = 0.0001, # learning rate for training
    ITER = 1000, # number of epochs for tuning and training
    embed_dim = 16, # hidden dimension for the first layer of the MLP
    num_heads = 2, # number of attention heads in the transformer
    num_layers = 2, # number of transformer layers
    track_mmd = False, # whether to track MMD during training
)

class BitTransformer(nn.Module):
    def __init__(self, max_seq_len, embed_dim=128, num_heads=4, num_layers=4):
        super().__init__()
        # Vocabulary size is 2 (for bits 0 and 1)
        self.token_emb = nn.Embedding(2, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        
        # We use batch_first=True so inputs are (batch, seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            batch_first=True,
            norm_first=True # Pre-LN is standard for stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # A single output node per sequence position (predicts the logit of '1')
        self.fc_out = nn.Linear(embed_dim, 1)

    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # Create a causal mask so position 'i' cannot see 'i+1' or beyond
        # -inf masks out future tokens, 0.0 allows looking at past tokens
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        
        # Combine token embeddings (0s and 1s) with their positions
        emb = self.token_emb(x) + self.pos_emb(positions)
        
        # Pass through the causal transformer
        out = self.transformer(emb, mask=mask, is_causal=True)
        
        # Project down to a single logit per position and squeeze the last dimension
        logits = self.fc_out(out).squeeze(-1)
        return logits
    

def generate_bitstrings(model, device, bit_length, num_samples):
    model.eval()
    
    # We need a starting point. Since we don't have a special <START> token,
    # we initialize the very first bit of our sequence randomly.
    generated = torch.randint(0, 2, (num_samples, 1), dtype=torch.long, device=device)
    
    with torch.no_grad():
        # Loop to generate the remaining (bit_length - 1) bits
        for _ in range(bit_length - 1):
            # Pass the sequence generated so far into the model
            logits = model(generated)
            
            # We only care about the prediction for the VERY LAST bit
            next_bit_logit = logits[:, -1]
            
            # Convert the logit to a probability (0.0 to 1.0)
            prob = torch.sigmoid(next_bit_logit)
            
            # Sample from a Bernoulli distribution based on that probability
            next_bit = torch.bernoulli(prob).long().unsqueeze(1)
            
            # Append the new bit to our sequence and repeat
            generated = torch.cat([generated, next_bit], dim=1)
            
    return generated


def generate_large_dataset(model, device, bit_length, total_samples, micro_batch_size=10000):
    all_generated = []
    
    for i in range(0, total_samples, micro_batch_size):
        # Calculate how many to generate this loop (handles the remainder)
        current_batch_size = min(micro_batch_size, total_samples - i)
        
        # Generate the small batch
        batch_samples = generate_bitstrings(model, device, bit_length, current_batch_size)
        
        # IMMEDIATELY move to CPU RAM so the GPU forgets it
        all_generated.append(batch_samples.cpu())
        
        # Force PyTorch to clear fragmented memory
        torch.cuda.empty_cache() 
        
    # Stitch them all together on the CPU at the very end
    return torch.cat(all_generated, dim=0)


def main():
    data = torch.tensor(np.loadtxt(PATH + "training_set.txt"), dtype=torch.float32)
    test_data = torch.tensor(np.loadtxt(PATH + "test_set.txt"), dtype=torch.float32)

    n_reps = CONFIG["n_reps"]
    test_length_cutoff = CONFIG["test_length_cutoff"]
    num_samples = CONFIG["num_samples"]
    learning_rate = CONFIG["learning_rate"]
    ITER = CONFIG["ITER"]
    embed_dim = CONFIG["embed_dim"]
    num_heads = CONFIG["num_heads"]
    num_layers = CONFIG["num_layers"]
    track_mmd = CONFIG["track_mmd"]
    n_bits = test_data.shape[1]
    sigma = np.sqrt(median_heuristic(data.numpy()))
    sigmas = np.arange(sigma, sigma**2, 0.5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BitTransformer(max_seq_len=n_bits, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters in the model: {total_params}")

    batch_size = 128

    training_mmds = []
    losses = []
    
    for epoch in tqdm(range(ITER)):
        model.train()
        indices = torch.randint(0, len(data), (batch_size,))
        batch = data[indices].to(device)
        
        # THE AUTOREGRESSIVE SHIFT
        # Input: All bits except the very last one
        # Target: All bits except the very first one
        inputs = batch[:, :-1].int()
        targets = batch[:, 1:].float() # Targets must be float for BCE loss
        
        # Forward pass
        logits = model(inputs)
        
        # Calculate loss
        loss = criterion(logits, targets)
        losses.append(loss.item())

        if track_mmd:
            samples = generate_bitstrings(model, device, n_bits, 1000).cpu()
            training_mmds.append(biased_sample_mmd(test_data.numpy(), samples.numpy(), [sigma, 2*sigma], test_length_cutoff))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if track_mmd:
        np.savetxt(PATH + "transformer_training_mmds.txt", training_mmds)

    np.savetxt(PATH + "transformer_training_losses.txt", losses)

    mmds = []
    for n in tqdm(range(n_reps)):
        samples = generate_large_dataset(model, device, n_bits, num_samples, micro_batch_size=10000).cpu()
        mmds.append(biased_sample_mmd(test_data.numpy(), samples.numpy(), sigmas, test_length_cutoff))

    np.savetxt(PATH + "transformer_mmds.txt", mmds)

    
    covariance_matrix = get_sample_covariance_matrix(n_bits // 3, samples.numpy())
    np.savetxt(PATH + "transformer_covariance_matrix.txt", covariance_matrix)


if __name__ == "__main__":
    main()
