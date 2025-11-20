import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neighbors import KernelDensity
import normflows as nf

# Configuração
torch.manual_seed(2025)
np.random.seed(2025)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simulando dados do two moons, centralizando e normalizando
n_samples = 8000
X, _ = make_moons(n_samples=n_samples, noise=0.08, random_state=42)
X = X.astype(np.float32)
X = (X - X.mean(axis=0)) / X.std(axis=0)
X_train = torch.tensor(X, dtype=torch.float32, device=device)

D = 2  

# Fit do KDE
def fit_kde(data):
    kde = KernelDensity(bandwidth=0.15, kernel='gaussian')
    kde.fit(data)
    return kde

kde = fit_kde(X)

# Fit do NICE
base = nf.distributions.base.DiagGaussian(D)
flows = []
num_layers = 8       
hidden_units = 128     
num_hidden_layers = 4   

for i in range(num_layers):
    mask = torch.tensor([1.0, 0.0] if i % 2 == 0 else [0.0, 1.0])
    layer_sizes = [D] + [hidden_units] * num_hidden_layers + [D]
    
    param_map = nf.nets.MLP(
        layer_sizes,
        init_zeros=True,
        output_fn=None
    )
    
    # Manter RealNVP apenas com camada de translação (NICE) 
    flows.append(nf.flows.MaskedAffineFlow(b=mask, t=param_map, s=None))

model = nf.NormalizingFlow(base, flows).to(device)


optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
batch_size = 1024
epochs = 120
num_batches = int(np.ceil(len(X_train) / batch_size))

model.train()

for epoch in range(1, epochs + 1):
    perm = torch.randperm(X_train.shape[0], device=device)
    epoch_loss = 0.0
    
    for b in range(num_batches):
        idx = perm[b * batch_size : min((b + 1) * batch_size, len(X_train))]
        xb = X_train[idx]
        
        # Negative log-likelihood 
        loss = -model.log_prob(xb).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * len(xb)
    
    avg_loss = epoch_loss / len(X_train)
    if epoch % 30 == 0 or epoch == 1:
        print(f"Época {epoch:3d}/{epochs} | NLL: {avg_loss:.4f}")

with torch.no_grad():
    # Grid para densidade
    x_range = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
    y_range = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200)
    XX, YY = np.meshgrid(x_range, y_range)
    grid = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)
    grid_tensor = torch.tensor(grid, dtype=torch.float32, device=device)
    
    # Densidade NICE
    log_prob_nice = model.log_prob(grid_tensor).cpu().numpy()
    density_nice = np.exp(log_prob_nice).reshape(XX.shape)
    
    # Densidade KDE
    log_prob_kde = kde.score_samples(grid)
    density_kde = np.exp(log_prob_kde).reshape(XX.shape)
    
    # Gerar amostras do modelo
    samples, _ = model.sample(4000)
    samples = samples.cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

tamanho_fonte = 12
# Plot 1: Dados originais
axes[0].scatter(X[:, 0], X[:, 1], s=12, alpha=0.7, c='black', edgecolors='none')
axes[0].set_title("Amostra - two moons (n = 8000)", fontsize=tamanho_fonte, fontweight='bold', pad=15)
axes[0].set_xlabel("$x_0$", fontsize=14)
axes[0].set_ylabel("$x_1$", fontsize=14)
axes[0].set_xlim(-3, 3)
axes[0].set_ylim(-3, 3)
axes[0].set_aspect('equal', 'box')
axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
axes[0].tick_params(labelsize=11)

# Plot 2: Densidade KDE
im = axes[2].contourf(XX, YY, density_nice, levels=50, cmap='Greys', alpha=0.9, vmin=0)
axes[2].set_title("Densidade estimada - NICE", fontsize=tamanho_fonte, fontweight='bold', pad=15)
axes[2].set_xlabel("$x_0$", fontsize=14)
axes[2].set_ylabel("$x_1$", fontsize=14)
axes[2].set_xlim(-3, 3)
axes[2].set_ylim(-3, 3)
axes[2].set_aspect('equal', 'box')
axes[2].tick_params(labelsize=11)
cbar1 = plt.colorbar(im, ax=axes[1], label='p(x)', pad=0.05)
cbar1.ax.tick_params(labelsize=10)
cbar1.set_label('p(x)', fontsize=13)

# Plot 3: Densidade estimada NICE
im2 = axes[1].contourf(XX, YY, density_kde, levels=50, cmap='Greys', alpha=0.9, vmin=0)
axes[1].set_title(f"Densidade KDE", fontsize=tamanho_fonte, fontweight='bold', pad=15)
axes[1].set_xlabel("$x_0$", fontsize=14)
axes[1].set_ylabel("$x_1$", fontsize=14)
axes[1].set_xlim(-3, 3)
axes[1].set_ylim(-3, 3)
axes[1].set_aspect('equal', 'box')
axes[1].tick_params(labelsize=11)
cbar2 = plt.colorbar(im2, ax=axes[2], label='p(x)', pad=0.05)
cbar2.ax.tick_params(labelsize=10)
cbar2.set_label('p(x)', fontsize=13)

plt.tight_layout()

# Salvando a imagem
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
output_path = os.path.join(project_root, 'plots', 'density_estimation.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')