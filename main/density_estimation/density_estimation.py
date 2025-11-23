import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal 
from sklearn.neighbors import KernelDensity
import normflows as nf

# Configuração
torch.manual_seed(2025)
np.random.seed(2025)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_samples = 8000
D = 2

true_mu = np.array([0.0, 0.0])
true_cov = np.array([[1.0, 0.7], 
                     [0.7, 1.0]]) 
ground_truth_dist = multivariate_normal(mean=true_mu, cov=true_cov)
X = ground_truth_dist.rvs(size=n_samples).astype(np.float32)
X_train = torch.tensor(X, dtype=torch.float32, device=device)

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
        loss = -model.log_prob(xb).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(xb)
    
    avg_loss = epoch_loss / len(X_train)
    if epoch % 30 == 0 or epoch == 1:
        print(f"Época {epoch:3d}/{epochs} | NLL: {avg_loss:.4f}")

def hellinger_distance(p, q):
    p_norm = p / (np.sum(p) + 1e-10)
    q_norm = q / (np.sum(q) + 1e-10)
    bc = np.sum(np.sqrt(p_norm * q_norm))
    return np.sqrt(1 - bc)

def mse_distance(p, q):
    return np.mean((p - q) ** 2)

def euclidean_distance(p, q):
    return np.sqrt(np.sum((p - q) ** 2))

with torch.no_grad():
    x_range = np.linspace(-4, 4, 200)
    y_range = np.linspace(-4, 4, 200)
    XX, YY = np.meshgrid(x_range, y_range)
    grid = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)
    grid_tensor = torch.tensor(grid, dtype=torch.float32, device=device)
    
    # 1. Densidade REAL (Ground Truth)
    density_true = ground_truth_dist.pdf(grid).reshape(XX.shape)

    # 2. Densidade NICE
    log_prob_nice = model.log_prob(grid_tensor).cpu().numpy()
    density_nice = np.exp(log_prob_nice).reshape(XX.shape)
    
    # 3. Densidade KDE
    log_prob_kde = kde.score_samples(grid)
    density_kde = np.exp(log_prob_kde).reshape(XX.shape)
    
    # Comparação: KDE vs REAL
    h_kde = hellinger_distance(density_true, density_kde)
    mse_kde = mse_distance(density_true, density_kde)
    euclidian_kde = euclidean_distance(density_true, density_kde)
    
    # Comparação: NICE vs REAL
    h_nice = hellinger_distance(density_true, density_nice)
    mse_nice = mse_distance(density_true, density_nice)
    euclidian_nice = euclidean_distance(density_true, density_nice)
    
    print("\n" + "="*60)
    print("COMPARATIVO DE ERRO EM RELAÇÃO À DISTRIBUIÇÃO REAL")
    print("="*60)
    print(f"{'Métrica':<25} | {'KDE (Baseline)':<15} | {'NICE (Flow)':<15}")
    print("-" * 60)
    print(f"{'Hellinger Dist.':<25} | {h_kde:.6f}          | {h_nice:.6f}")
    print(f"{'MSE':<25} | {mse_kde:.6f}          | {mse_nice:.6f}")
    print(f"{'L2':<25} | {euclidian_kde:.6f}          | {euclidian_nice:.6f}")
    print("="*60 + "\n")
    
    # Gerar amostras do modelo
    samples, _ = model.sample(4000)
    samples = samples.cpu().numpy()

# PLOTAGEM
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
tamanho_fonte = 12

# Plot 1: Dados Reais (Normal Bivariada)
axes[0].scatter(X[:, 0], X[:, 1], s=12, alpha=0.5, c='black', edgecolors='none')
axes[0].set_title("Amostra Normal Bivariada", fontsize=tamanho_fonte, fontweight='bold', pad=15)
axes[0].set_xlim(-4, 4)
axes[0].set_ylim(-4, 4)
axes[0].set_aspect('equal', 'box')

# Plot 2: Densidade REAL (Ground Truth) 
im = axes[1].contourf(XX, YY, density_kde, levels=50, cmap='Greys', alpha=0.9, vmin=0)
axes[1].set_title(f"Densidade KDE", fontsize=tamanho_fonte, fontweight='bold', pad=15)
axes[1].set_xlim(-4, 4)
axes[1].set_ylim(-4, 4)
axes[1].set_aspect('equal', 'box')
plt.colorbar(im, ax=axes[1], label='p(x)')

# Plot 3: Densidade Estimada NICE
im2 = axes[2].contourf(XX, YY, density_nice, levels=50, cmap='Greys', alpha=0.9, vmin=0)
axes[2].set_title(f"Densidade NICE", fontsize=tamanho_fonte, fontweight='bold', pad=15)
axes[2].set_xlim(-4, 4)
axes[2].set_ylim(-4, 4)
axes[2].set_aspect('equal', 'box')
plt.colorbar(im2, ax=axes[2], label='p(x)')

plt.tight_layout()

script_dir = os.getcwd()
project_root = os.path.dirname(os.path.dirname(script_dir))
output_path = os.path.join(project_root, 'plots', 'density_estimation.png')
metrics_output_path = os.path.join(project_root, 'plots', 'density_metrics.txt')
with open(metrics_output_path, 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("MÉTRICAS DE COMPARAÇÃO ENTRE KDE E NICE\n")
    f.write("Dataset: Normal Bivariada\n")
    f.write("="*60 + "\n\n")
    f.write(f"{'Métrica':<25} | {'KDE (Baseline)':<15} | {'NICE (Flow)':<15}\n")
    f.write(f"{'Hellinger Dist.':<25} | {h_kde:.6f}          | {h_nice:.6f}\n")
    f.write(f"{'MSE':<25} | {mse_kde:.6f}          | {mse_nice:.6f}\n")
    f.write(f"{'L2':<25} | {euclidian_kde:.6f}          | {euclidian_nice:.6f}\n")
    f.write("="*60 + "\n")
    f.write("INTERPRETAÇÃO DAS MÉTRICAS:\n")
    f.write("-" * 60 + "\n")
    f.write("• Distância de Hellinger: varia de 0 (idênticas) a 1 (totalmente diferentes)\n")
    f.write("• MSE: quanto menor, mais próximas as densidades\n")
    f.write("• Distância Euclidiana: norma L2 da diferença entre densidades\n")