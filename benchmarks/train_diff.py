import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm
import math

import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import median_heuristic
from src.classical_utils import biased_sample_mmd, get_sample_covariance_matrix
"""
Training of diffusion models.
"""

PATH = str(REPO_ROOT / "data" / "benchmark_datasets" / "molecular") + "/"

CONFIG = dict(
    n_reps = 5, # number of repetitions to average over
    test_length_cutoff = 5, # maximal Z-string length
    num_samples = 100000, # number of test samples for evaluation
    learning_rate = 0.0001, # learning rate for training
    ITER = 10000, # number of epochs for tuning and training
    T = 100, # number of diffusion steps
    hidden_dim = 256, # hidden dimension for the first layer of the MLP
    track_mmd = False, # whether to track MMD during training
)

class DiffusionMLP(nn.Module):
    def __init__(self, data_dim, T, hidden_dim=128):
        super().__init__()
        self.time_emb = nn.Embedding(T, hidden_dim)
        self.fc1 = nn.Linear(data_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, data_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x, t):
        t_emb = self.time_emb(t)
        h = self.relu(self.fc1(x) + t_emb) 
        h = self.relu(self.fc2(h))
        return self.fc3(h)
    

def generate_bitstrings(model,alphas, betas, alphas_cumprod, bit_length, T, num_samples):
    # Start with pure noise sized to our bitstring length
    x = torch.randn(num_samples, bit_length)
    
    model.eval()
    for t_step in reversed(range(T)):
        t = torch.full((num_samples,), t_step, dtype=torch.long)
        
        with torch.no_grad():
            pred_noise = model(x, t)
            
        alpha_t = alphas[t].unsqueeze(-1)
        alpha_bar_t = alphas_cumprod[t].unsqueeze(-1)
        beta_t = betas[t].unsqueeze(-1)
        
        noise_factor = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
        mean = (1 / torch.sqrt(alpha_t)) * (x - noise_factor * pred_noise)
        
        if t_step > 0:
            z = torch.randn_like(x)
            sigma = torch.sqrt(beta_t)
            x = mean + sigma * z
        else:
            x = mean
            
    # If the float is > 0, it's a 1. If it's <= 0, it's a 0.
    generated_bits = (x > 0.0).int()
    
    return generated_bits, x


def main():
    data = torch.tensor(np.loadtxt(PATH + "training_set.txt"), dtype=torch.float32)
    test_data = torch.tensor(np.loadtxt(PATH + "test_set.txt"), dtype=torch.float32)

    n_reps = CONFIG["n_reps"]
    test_length_cutoff = CONFIG["test_length_cutoff"]
    num_samples = CONFIG["num_samples"]
    learning_rate = CONFIG["learning_rate"]
    ITER = CONFIG["ITER"]
    T = CONFIG["T"]  # number of diffusion steps
    hidden_dim = CONFIG["hidden_dim"]  

    betas = torch.linspace(1e-4, 0.02, T)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    n_bits = test_data.shape[1]
    sigma = np.sqrt(median_heuristic(data.numpy()))
    sigmas = np.arange(sigma, sigma**2, 0.5)

    
    model = DiffusionMLP(data_dim=n_bits, T=T, hidden_dim=hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters in the model: {total_params}")

    batch_size = 128

    dataset_continuous = (data * 2.0) - 1.0

    training_mmds = []
    losses = []
    
    for epoch in tqdm(range(ITER)):
        # Sample from the CONTINUOUS mapped dataset
        indices = torch.randint(0, len(dataset_continuous), (batch_size,))
        x_0 = dataset_continuous[indices]
        
        t = torch.randint(0, T, (batch_size,))
        noise = torch.randn_like(x_0)
        
        alpha_bar_t = alphas_cumprod[t].unsqueeze(-1)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        
        predicted_noise = model(x_t, t)
        
        # We still use MSE Loss! The network is predicting continuous noise.
        loss = nn.MSELoss()(predicted_noise, noise)
        losses.append(loss.item())

        if CONFIG["track_mmd"]:
            samples, _ = generate_bitstrings(model, alphas, betas, alphas_cumprod, n_bits, T, 1000)
            training_mmds.append(biased_sample_mmd(data.numpy(), samples.numpy(), [sigma, 2*sigma], test_length_cutoff))


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if CONFIG["track_mmd"]:
        np.savetxt(PATH + "diffusion_training_mmds.txt", training_mmds)

    np.savetxt(PATH + "diffusion_training_losses.txt", losses)

    mmds = []
    for n in tqdm(range(n_reps)):
        samples, _ = generate_bitstrings(model, alphas, betas, alphas_cumprod, n_bits, T, num_samples)
        mmds.append(biased_sample_mmd(test_data.numpy(), samples.numpy(), sigmas, test_length_cutoff))

    np.savetxt(PATH + "diffusion_mmds.txt", mmds)

    covariance_matrix = get_sample_covariance_matrix(n_bits // 3, samples.numpy())
    np.savetxt(PATH + "diffusion_covariance_matrix.txt", covariance_matrix)


if __name__ == "__main__":
    main()
