# CÓDIGO ORIGINAL DE OFEKIRSH (2035)

import torch
import torch.nn as nn
from torch.distributions.transforms import SigmoidTransform, AffineTransform
from torch.distributions import Uniform, TransformedDistribution
import numpy as np



class AdditiveCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize an additive coupling layer.

        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(AdditiveCoupling, self).__init__()
        self.in_out_dim = in_out_dim
        self.mid_dim = mid_dim
        self.hidden = hidden
        self.mask_config = mask_config

        # Define the MLP for m(x1)
        layers = nn.ModuleList()
        input_dim = in_out_dim // 2
        for _ in range(hidden):
            layers.append(nn.Linear(input_dim, mid_dim))
            layers.append(nn.ReLU())
            input_dim = mid_dim
        layers.append(nn.Linear(mid_dim, in_out_dim // 2))  # Output layer
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, log_det_J, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            log_det_J: log determinant of the Jacobian
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and updated log-determinant of Jacobian.
        """
        """
        1.x1,x2 = split the vector x to odds and evens. 
        2.y1 = x1
        3.y2 = x2 + m(x1)
        4.y = concat(y1,y2)
        5.log_det = log_det (because of the log det is 0) 
        """
        # Split x into odd (x1) and even (x2) indices
        x1, x2 = x[:, self.mask_config::2], x[:, 1 - self.mask_config::2]

        if reverse:
            # Reverse mode: x2 = y2 - m(x1)
            y1 = x1
            y2 = x2 - self.mlp(x1)
        else:
            # Forward mode: y2 = x2 + m(x1)
            y1 = x1
            y2 = x2 + self.mlp(x1)

        ordered_concat = [y1, y2] if self.mask_config == 0 else [y2, y1]
        y = torch.stack(ordered_concat, dim=2).view(-1, self.in_out_dim)
        log_det_J = log_det_J  # Jacobian determinant is 1, so no change
        return y, log_det_J


class AffineCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize an affine coupling layer.

        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(AffineCoupling, self).__init__()
        self.in_out_dim = in_out_dim
        self.mid_dim = mid_dim
        self.hidden = hidden
        self.mask_config = mask_config
 
        # Define the scale layer for m(x1)
        scale_layers = nn.ModuleList()
        input_dim = in_out_dim // 2
        for _ in range(hidden):
            scale_layers.append(nn.Linear(input_dim, mid_dim))
            scale_layers.append(nn.ReLU())
            input_dim = mid_dim
        scale_layers.append(nn.Linear(mid_dim, in_out_dim // 2))  # Output layer
        scale_layers.append(nn.Tanh())
        self.scale_network = nn.Sequential(*scale_layers)
        
        # Define the shift layer for m(x1)
        shift_layers = nn.ModuleList()
        input_dim = in_out_dim // 2
        for _ in range(hidden):
            shift_layers.append(nn.Linear(input_dim, mid_dim))
            shift_layers.append(nn.ReLU())
            input_dim = mid_dim
        shift_layers.append(nn.Linear(mid_dim, in_out_dim // 2))  # Output layer
        shift_layers.append(nn.Tanh())
        self.shift_network = nn.Sequential(*shift_layers)

    def forward(self, x, log_det_J, reverse=False):
        """Forward pass.
    
        Args:
            x: input tensor.
            log_det_J: log determinant of the Jacobian
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and updated log-determinant of Jacobian.
        """
        x1, x2 = x[:, self.mask_config::2], x[:, 1 - self.mask_config::2]
    
        # Compute b1 and b2 from MLP
        scale = torch.exp(self.scale_network(x1))
        shift = self.shift_network(x1)  # Split into scaling and translation components

        if reverse:
            # Reverse mode: x2 = (y2 - b2) / b1
            y1 = x1
            y2 = (x2 - shift) / scale
        else:
            # Forward mode: y2 = x2 * b1 + b2
            y1 = x1
            y2 = x2 * scale + shift
            log_det_J += scale.log().sum(dim=1)  # Update log-determinant
    
        # Reassemble y by interleaving y1 and y2
        ordered_concat = [y1, y2] if self.mask_config == 0 else [y2, y1]
        y = torch.stack(ordered_concat, dim=2).view(-1, self.in_out_dim)
        return y, log_det_J


class Scaling(nn.Module):
    def __init__(self, dim):
        """Initialize a (log-)scaling layer.

        Args:
            dim: input/output dimensions.
        """
        super(Scaling, self).__init__()
        self.scale = nn.Parameter(
            torch.zeros((1, dim)), requires_grad=True)
        self.eps = 1e-5

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log-determinant of Jacobian.
        """
        """
        1. y = x * S
        2. log_det = sum of the elements in the diagonal S matrix
        """
        scale = torch.exp(self.scale) + self.eps

        if reverse:
            # Inverse transformation: x = y / S
            y = x / scale
            log_det = torch.sum(torch.log(scale), dim=1) 
        else:
            # Forward transformation: y = x * S
            y = x * scale
            log_det = torch.sum(torch.log(scale), dim=1) 

        return y, log_det


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Move the parameters of Uniform to the device
uniform = Uniform(torch.tensor(0.).to(device), torch.tensor(1.).to(device))

# Move the transforms to the device
sigmoid_transform = SigmoidTransform()
affine_transform = AffineTransform(loc=0., scale=1.)

# Now create the logistic distribution with the moved components
logistic = TransformedDistribution(uniform, [sigmoid_transform.inv, affine_transform])


class NICE(nn.Module):
    def __init__(self, prior, coupling, coupling_type, in_out_dim, mid_dim, hidden, device):
        """Initialize a NICE.

        Args:
            coupling_type: 'additive' or 'adaptive'
            coupling: number of coupling layers.
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            device: run on cpu or gpu
        """
        super(NICE, self).__init__()
        self.device = device

        if prior == 'gaussian':
            self.prior = torch.distributions.Normal(
                torch.tensor(0.).to(device), torch.tensor(1.).to(device))
        elif prior == 'logistic':
            self.prior = logistic
        else:
            raise ValueError('Prior not implemented.')

        self.in_out_dim = in_out_dim
        self.mid_dim = mid_dim
        self.hidden = hidden
        self.coupling = coupling
        self.coupling_type = coupling_type
        self.scaling = Scaling(dim=in_out_dim)

        # Define coupling layers
        self.coupling_layers = nn.ModuleList()
        for i in range(self.coupling):
            if self.coupling_type == 'additive':
                self.coupling_layers.append(
                    AdditiveCoupling(in_out_dim, mid_dim, hidden, mask_config=i % 2)
                )
            elif self.coupling_type == 'adaptive':
                self.coupling_layers.append(
                    AffineCoupling(in_out_dim, mid_dim, hidden, mask_config=i % 2)
                )
            else:
                raise ValueError("Fuck that's not the option for coupling layer")

    def f_inverse(self, z):
        """Transformation g: Z -> X (inverse of f).

        Args:
            z: tensor in latent space Z.
        Returns:
            transformed tensor in data space X.
        """
        x, _ = self.scaling(z, reverse=True)
        for layer in reversed(list(self.coupling_layers)):
            x, _ = layer(x, 0, reverse=True)
        return x

    def f(self, x):
        """Transformation f: X -> Z (inverse of g).

        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor in latent space Z and log determinant Jacobian
        """
        z = x
        log_det = 0
        for layer in self.coupling_layers:
            z, log_det = layer(z, log_det, reverse=False)
        z, log_det_scale = self.scaling(z, reverse=False)
        log_det += log_det_scale
        return z, log_det

    def log_prob(self, x):
        """Computes data log-likelihood.

        (See Section 3.3 in the NICE paper.)

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        z, log_det_J = self.f(x)
        log_det_J -= np.log(256)*self.in_out_dim #log det for rescaling from [0.256] (after dequantization) to [0,1]
        prior_log_prob = self.prior.log_prob(z).to(device)
        log_ll = torch.sum(self.prior.log_prob(z), dim=1)
        
        return log_ll + log_det_J

    def sample(self, size):
        """Generates samples.

        Args:
            size: number of samples to generate.
        Returns:
            samples from the data space X.
        """
        z = self.prior.sample((size, self.in_out_dim)).to(self.device)
        return self.f_inverse(z)

    def forward(self, x):
        """Forward pass.

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        return self.log_prob(x)