import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.utils import make_grid

SEED = 2025
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

transform = transforms.ToTensor()

mnist = datasets.MNIST(root="data", train=True, download=True, transform=transform)
imgs_mnist = torch.stack([mnist[i][0] for i in range(50)])  

fmnist = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
imgs_fmnist = torch.stack([fmnist[i][0] for i in range(50)])       

def sampling(images, save_path, rows=5, cols=10):
    images = images[:rows*cols]
    grid = make_grid(images, nrow=cols, padding=2) 
    grid = grid.permute(1, 2, 0)

    plt.figure(figsize=(cols, rows)) 
    plt.imshow(grid.squeeze(), cmap="gray")
    plt.axis("off")
    
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)

sampling(imgs_mnist, save_path="plots/imgs_mnist_sample.png")
sampling(imgs_fmnist, save_path="plots/imgs_fmnist_sample.png")