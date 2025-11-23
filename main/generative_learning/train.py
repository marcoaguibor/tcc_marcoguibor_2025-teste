# CÓDIGO ADAPTADO DE OFEKIRSH (2025)

import argparse
import logging
import os
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms

SEED = 2025
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the NICE model from the nice module
try:
    import nice
except ImportError:
    logger.error("Failed to import 'nice' module. Make sure it's installed or in your PYTHONPATH.")
    raise


class Generator(nn.Module):
    """Generator network for GAN."""
    
    def __init__(self, latent_dim: int = 100, img_shape: Tuple[int, int, int] = (1, 28, 28)):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.img_size = int(torch.prod(torch.tensor(img_shape)))
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Arquitetura profunda (8 camadas ocultas) para equiparar com NICE
        # Largura: 256 neurônios (equivalente aos 128 do NICE mas com mais expressividade)
        self.model = nn.Sequential(
            *block(latent_dim, 256, normalize=False),  
            *block(256, 256),  
            *block(256, 512),  
            *block(512, 512),  
            *block(512, 512),  
            *block(512, 512),  
            *block(512, 256),  
            *block(256, 256),  
            nn.Linear(256, self.img_size),  
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    """Discriminator network for GAN."""
    
    def __init__(self, img_shape: Tuple[int, int, int] = (1, 28, 28)):
        super(Discriminator, self).__init__()
        self.img_size = int(torch.prod(torch.tensor(img_shape)))

    
        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(self.img_size, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.utils.spectral_norm(nn.Linear(256, 256)), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.utils.spectral_norm(nn.Linear(256, 512)), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.utils.spectral_norm(nn.Linear(512, 512)), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.utils.spectral_norm(nn.Linear(512, 512)), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.utils.spectral_norm(nn.Linear(512, 256)),  
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.utils.spectral_norm(nn.Linear(256, 128)),  
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.utils.spectral_norm(nn.Linear(128, 64)),   
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(64, 1),  
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


def add_uniform_noise(x):
    """Add uniform noise for dequantization."""
    return x + torch.zeros_like(x).uniform_(0., 1. / 256.)


def train_epoch_nice(
        flow: nice.NICE,
        trainloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device
) -> float:
    """Train the NICE flow model for one epoch."""
    loss_epoch = 0.0
    flow.train()

    for inputs, _ in trainloader:
        optimizer.zero_grad()
        inputs = inputs.view(inputs.shape[0], -1).to(device)
        loss = -flow(inputs).mean()
        loss_epoch += loss.item()
        loss.backward()
        optimizer.step()

    loss_epoch /= len(trainloader)
    return loss_epoch


def train_epoch_gan(
        generator: nn.Module,
        discriminator: nn.Module,
        trainloader: torch.utils.data.DataLoader,
        optimizer_g: torch.optim.Optimizer,
        optimizer_d: torch.optim.Optimizer,
        criterion: nn.Module,
        latent_dim: int,
        device: torch.device
) -> Tuple[float, float]:
    """Train the GAN for one epoch."""
    g_loss_epoch = 0.0
    d_loss_epoch = 0.0
    
    generator.train()
    discriminator.train()

    for real_imgs, _ in trainloader:
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)

        # Adversarial ground truths
        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        # Train Discriminator
        optimizer_d.zero_grad()
        real_loss = criterion(discriminator(real_imgs), valid)
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_imgs = generator(z)
        fake_loss = criterion(discriminator(fake_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()
        z = torch.randn(batch_size, latent_dim, device=device)
        gen_imgs = generator(z)
        g_loss = criterion(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_g.step()

        g_loss_epoch += g_loss.item()
        d_loss_epoch += d_loss.item()

    g_loss_epoch /= len(trainloader)
    d_loss_epoch /= len(trainloader)

    return g_loss_epoch, d_loss_epoch


def evaluate_nice(
        flow: nice.NICE,
        testloader: torch.utils.data.DataLoader,
        sample_shape: List[int],
        device: torch.device,
        plots_dir: Path,
        filename_prefix: str,
        epoch: int,
        generate_samples: bool = False,
        sample_size: int = 64,
        nrow: int = 4,
        sample_seed: int = 2025
) -> float:
    """Evaluate the NICE flow model and optionally generate samples."""
    loss_inference = 0.0
    flow.eval()

    with torch.no_grad():
        if generate_samples:
            plots_dir.mkdir(exist_ok=True, parents=True)

            # Set seed for reproducible sampling
            torch.manual_seed(sample_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(sample_seed)

            samples = flow.sample(sample_size).to(device)
            a, b = samples.min(), samples.max()
            samples = (samples - a) / (b - a + 1e-10)
            samples = samples.view(-1, *sample_shape)

            sample_path = plots_dir / f"{filename_prefix}_epoch{epoch}.png"
            torchvision.utils.save_image(samples, sample_path, nrow=nrow, padding=2)
            logger.info(f"Generated {sample_size} NICE samples saved to {sample_path}")

        for xs, _ in testloader:
            xs = xs.view(xs.shape[0], -1).to(device)
            loss = -flow(xs).mean()
            loss_inference += loss.item()

        loss_inference /= len(testloader)
        return loss_inference


def evaluate_gan(
        generator: nn.Module,
        discriminator: nn.Module,
        testloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        latent_dim: int,
        sample_shape: List[int],
        device: torch.device,
        plots_dir: Path,
        filename_prefix: str,
        epoch: int,
        generate_samples: bool = False,
        sample_size: int = 64,
        nrow: int = 4,
        sample_seed: int = 2025
) -> Tuple[float, float]:
    """Evaluate the GAN and optionally generate samples."""
    g_loss_inference = 0.0
    d_loss_inference = 0.0
    
    generator.eval()
    discriminator.eval()

    with torch.no_grad():
        if generate_samples:
            plots_dir.mkdir(exist_ok=True, parents=True)

            # Set seed for reproducible sampling
            torch.manual_seed(sample_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(sample_seed)

            z = torch.randn(sample_size, latent_dim, device=device)
            samples = generator(z)
            samples = (samples + 1) / 2  # From [-1, 1] to [0, 1]

            sample_path = plots_dir / f"{filename_prefix}_epoch{epoch}.png"
            torchvision.utils.save_image(samples, sample_path, nrow=nrow, padding=2)
            logger.info(f"Generated {sample_size} GAN samples saved to {sample_path}")

        for real_imgs, _ in testloader:
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            real_loss = criterion(discriminator(real_imgs), valid)
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_imgs = generator(z)
            fake_loss = criterion(discriminator(fake_imgs), fake)
            d_loss = (real_loss + fake_loss) / 2

            z = torch.randn(batch_size, latent_dim, device=device)
            gen_imgs = generator(z)
            g_loss = criterion(discriminator(gen_imgs), valid)

            g_loss_inference += g_loss.item()
            d_loss_inference += d_loss.item()

        g_loss_inference /= len(testloader)
        d_loss_inference /= len(testloader)

    return g_loss_inference, d_loss_inference



def get_data_loaders(
        dataset_name: str,
        batch_size: int,
        data_root: str = "./data",
        num_workers: int = 0,
        method: str = "nice"
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create data loaders for the specified dataset."""
    

    if method == "nice":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (1.,)),
            transforms.Lambda(add_uniform_noise)
        ])
    else:  # GAN
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    data_path = Path(data_root)
    data_path.mkdir(exist_ok=True, parents=True)

    if dataset_name.lower() == 'mnist':
        trainset = torchvision.datasets.MNIST(
            root=data_path / 'MNIST', train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.MNIST(
            root=data_path / 'MNIST', train=False, download=True, transform=transform
        )
    elif dataset_name.lower() == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(
            root=data_path / 'FashionMNIST', train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.FashionMNIST(
            root=data_path / 'FashionMNIST', train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported. Use 'mnist' or 'fashion-mnist'.")

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    return trainloader, testloader


def build_model_name(args: argparse.Namespace) -> str:
    """Construct a descriptive model name from arguments."""
    if args.method == "nice":
        return (
            f"{args.dataset}_nice_"
            f"batch{args.batch_size}_"
            f"coupling{args.coupling}_"
            f"type{args.coupling_type}_"
            f"mid{args.mid_dim}_"
            f"hidden{args.hidden}"
        )
    else:  # GAN
        return (
            f"{args.dataset}_gan_"
            f"batch{args.batch_size}_"
            f"latent{args.latent_dim}"
        )



def train_and_evaluate(args: argparse.Namespace) -> None:
    """Train and evaluate the model with specified parameters."""
    
    # Set up device
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device(f"cuda:{args.gpu}")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    # Set up plots directory
    plots_dir = Path("./plots")
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Get data loaders
    trainloader, testloader = get_data_loaders(
        args.dataset, args.batch_size, args.data_root, args.num_workers, args.method
    )

    sample_shape = [1, 28, 28]  # [C, H, W]
    model_name = build_model_name(args)

    # ==================== NICE Training ====================
    if args.method == "nice":
        input_dim = sample_shape[0] * sample_shape[1] * sample_shape[2]
        logger.info(f"Initializing NICE model with input dimension {input_dim}")

        flow = nice.NICE(
            prior=args.prior,
            coupling=args.coupling,
            coupling_type=args.coupling_type,
            in_out_dim=input_dim,
            mid_dim=args.mid_dim,
            hidden=args.hidden,
            device=device
        ).to(device)

        optimizer = torch.optim.Adam(flow.parameters(), lr=args.lr)
        scheduler = None
        if args.lr_schedule:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )

        train_losses = []
        test_losses = []

        logger.info(f"Starting NICE training for {args.epochs} epochs")
        for epoch in range(args.epochs):
            train_loss = train_epoch_nice(flow, trainloader, optimizer, device)
            train_losses.append(train_loss)

            test_loss = evaluate_nice(
                flow, testloader, sample_shape, device, plots_dir, model_name, epoch,
                generate_samples=(epoch % args.sample_interval == 0),
                sample_size=args.sample_size, nrow=4, sample_seed=args.sample_seed
            )
            test_losses.append(test_loss)

            if scheduler:
                scheduler.step(test_loss)

            logger.info(f"Epoch {epoch + 1}/{args.epochs} - Train: {train_loss:.4f}, Test: {test_loss:.4f}")

        # Final evaluation
        evaluate_nice(flow, testloader, sample_shape, device, plots_dir, model_name,
                     args.epochs, True, args.sample_size, 4, args.sample_seed)
        
        logger.info(f"Training complete. Train loss: {train_losses[-1]:.4f}, Test loss: {test_losses[-1]:.4f}")

    else:  # method == "gan"
        logger.info(f"Initializing GAN with latent dimension {args.latent_dim}")

        generator = Generator(args.latent_dim, tuple(sample_shape)).to(device)
        discriminator = Discriminator(tuple(sample_shape)).to(device)
        criterion = nn.BCELoss()

        optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

        g_train_losses = []
        d_train_losses = []
        g_test_losses = []
        d_test_losses = []

        logger.info(f"Starting GAN training for {args.epochs} epochs")
        for epoch in range(args.epochs):
            g_loss, d_loss = train_epoch_gan(
                generator, discriminator, trainloader, optimizer_g, optimizer_d,
                criterion, args.latent_dim, device
            )
            g_train_losses.append(g_loss)
            d_train_losses.append(d_loss)

            g_test_loss, d_test_loss = evaluate_gan(
                generator, discriminator, testloader, criterion, args.latent_dim,
                sample_shape, device, plots_dir, model_name, epoch,
                generate_samples=(epoch % args.sample_interval == 0),
                sample_size=args.sample_size, nrow=4, sample_seed=args.sample_seed
            )
            g_test_losses.append(g_test_loss)
            d_test_losses.append(d_test_loss)

            logger.info(
                f"Epoch {epoch + 1}/{args.epochs} - "
                f"G: {g_loss:.4f}/{g_test_loss:.4f}, D: {d_loss:.4f}/{d_test_loss:.4f}"
            )

        # Final evaluation
        evaluate_gan(generator, discriminator, testloader, criterion, args.latent_dim,
                    sample_shape, device, plots_dir, model_name, args.epochs,
                    True, args.sample_size, 4, args.sample_seed)
        
        logger.info(
            f"Training complete. "
            f"G loss: {g_train_losses[-1]:.4f}/{g_test_losses[-1]:.4f}, "
            f"D loss: {d_train_losses[-1]:.4f}/{d_test_losses[-1]:.4f}"
        )

    logger.info("Training and evaluation complete!")



def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='NICE Flow and GAN Model Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Method selection
    parser.add_argument('--method', type=str, default='nice', choices=['nice', 'gan'],
                       help='Model type to train')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'fashion-mnist'], help='Dataset to use')
    parser.add_argument('--data-root', type=str, default='./data',
                       help='Root directory for dataset storage')


    parser.add_argument('--prior', type=str, default='logistic',
                       choices=['logistic', 'gaussian'], help='Latent distribution for NICE')
    parser.add_argument('--coupling', type=int, default=4,
                       help='Number of coupling layers for NICE')
    parser.add_argument('--coupling-type', type=str, default='additive',
                       choices=['additive', 'affine'], help='Type of coupling layers for NICE')
    parser.add_argument('--mid-dim', type=int, default=1000,
                       help='Dimension of hidden layers for NICE')
    parser.add_argument('--hidden', type=int, default=5,
                       help='Number of hidden layers for NICE')

    parser.add_argument('--latent-dim', type=int, default=100,
                       help='Dimension of latent space for GAN')

    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--lr-schedule', action='store_true',
                       help='Use learning rate scheduler (NICE only)')

    parser.add_argument('--no-cuda', action='store_true',
                       help='Disable CUDA')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of data loading workers')

    parser.add_argument('--output-dir', type=str, default='./output',
                       help='Output directory')
    parser.add_argument('--sample-interval', type=int, default=10,
                       help='Interval for generating samples')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                       help='Interval for saving checkpoints')
    parser.add_argument('--sample-size', type=int, default=64,
                       help='Number of samples to generate')
    parser.add_argument('--sample-seed', type=int, default=42,
                       help='Random seed for generating samples (ensures reproducibility across epochs)')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    logger.info(f"{args.method.upper()} training started with configuration:")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")

    try:
        train_and_evaluate(args)
    except Exception as e:
        logger.exception(f"Error during training: {e}")
        raise


if __name__ == '__main__':
    main()