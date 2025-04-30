"""
Normalizing Flow Module

This module implements normalizing flows for sampling from
posterior amplitude distributions in quantum systems.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from quantum_harmonic_oscillator import QuantumHarmonicOscillator


class PlanarFlow(nn.Module):
    """
    Planar flow transformation: f(z) = z + u * tanh(w^T * z + b)
    """
    def __init__(self, dim):
        """
        Initialize a planar flow transformation.
        
        Args:
            dim (int): Dimension of the input/output
        """
        super(PlanarFlow, self).__init__()
        
        # Parameters of the transformation
        self.w = nn.Parameter(torch.randn(dim))
        self.u = nn.Parameter(torch.randn(dim))
        self.b = nn.Parameter(torch.randn(1))
        
    def forward(self, z):
        """
        Apply the planar flow transformation.
        
        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, dim)
            
        Returns:
            tuple: (transformed_z, log_det_jacobian)
        """
        # Ensure w^T u > -1 for invertibility
        wu = torch.sum(self.w * self.u)
        m_wu = -1 + torch.log(1 + torch.exp(wu))
        u_hat = self.u + (m_wu - wu) * self.w / torch.sum(self.w**2)
        
        # Compute activation
        activation = torch.tanh(torch.matmul(z, self.w.unsqueeze(1)).squeeze(1) + self.b)
        
        # Apply transformation
        z_next = z + u_hat.unsqueeze(0) * activation.unsqueeze(1)
        
        # Compute log determinant of Jacobian
        psi = (1 - activation**2).unsqueeze(1) * self.w.unsqueeze(0)
        log_det_jacobian = torch.log(torch.abs(1 + torch.sum(psi * u_hat.unsqueeze(0), dim=1)))
        
        return z_next, log_det_jacobian


class RadialFlow(nn.Module):
    """
    Radial flow transformation: f(z) = z + β * (z - z0) / (α + |z - z0|)
    """
    def __init__(self, dim):
        """
        Initialize a radial flow transformation.
        
        Args:
            dim (int): Dimension of the input/output
        """
        super(RadialFlow, self).__init__()
        
        # Parameters of the transformation
        self.z0 = nn.Parameter(torch.randn(dim))
        self.log_alpha = nn.Parameter(torch.randn(1))
        self.beta = nn.Parameter(torch.randn(1))
        
    def forward(self, z):
        """
        Apply the radial flow transformation.
        
        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, dim)
            
        Returns:
            tuple: (transformed_z, log_det_jacobian)
        """
        # Ensure α > 0 for invertibility
        alpha = torch.exp(self.log_alpha)
        
        # Compute distance from reference point
        diff = z - self.z0.unsqueeze(0)
        r = torch.norm(diff, dim=1, keepdim=True)
        
        # Apply transformation
        h = 1 / (alpha + r)
        z_next = z + self.beta * h * diff
        
        # Compute log determinant of Jacobian
        dim = z.shape[1]
        log_det_jacobian = (dim - 1) * torch.log(1 + self.beta * h) - torch.log(alpha + r) + torch.log(alpha + r + self.beta)
        
        return z_next, log_det_jacobian


class NormalizingFlow(nn.Module):
    """
    Normalizing flow model for transforming a simple distribution
    into a more complex one.
    """
    def __init__(self, dim, flow_length=16, flow_type='planar'):
        """
        Initialize the normalizing flow model.
        
        Args:
            dim (int): Dimension of the input/output
            flow_length (int): Number of flow transformations
            flow_type (str): Type of flow transformation ('planar' or 'radial')
        """
        super(NormalizingFlow, self).__init__()
        
        self.dim = dim
        self.flow_length = flow_length
        
        # Create flow transformations
        self.flows = nn.ModuleList()
        for _ in range(flow_length):
            if flow_type == 'planar':
                self.flows.append(PlanarFlow(dim))
            elif flow_type == 'radial':
                self.flows.append(RadialFlow(dim))
            else:
                raise ValueError(f"Unknown flow type: {flow_type}")
    
    def forward(self, z):
        """
        Apply the sequence of flow transformations.
        
        Args:
            z (torch.Tensor): Input tensor from the base distribution
            
        Returns:
            tuple: (transformed_z, log_prob)
        """
        # Log probability of the base distribution (standard normal)
        log_prob = -0.5 * torch.sum(z**2, dim=1) - 0.5 * self.dim * np.log(2 * np.pi)
        
        # Apply sequence of transformations
        for flow in self.flows:
            z, log_det_jacobian = flow(z)
            log_prob = log_prob - log_det_jacobian
        
        return z, log_prob
    
    def sample(self, n_samples=1, device='cpu'):
        """
        Generate samples from the flow model.
        
        Args:
            n_samples (int): Number of samples to generate
            device (str): Device to generate samples on
            
        Returns:
            torch.Tensor: Generated samples
        """
        with torch.no_grad():
            # Sample from the base distribution
            z = torch.randn(n_samples, self.dim, device=device)
            
            # Apply flow transformations
            for flow in self.flows:
                z, _ = flow(z)
            
            return z


class AmplitudeFlow(nn.Module):
    """
    Normalizing flow model for quantum amplitude distributions.
    Transforms between latent space and quantum amplitude space.
    """
    def __init__(self, input_dim, latent_dim=8, hidden_dim=64, flow_length=16, flow_type='planar'):
        """
        Initialize the amplitude flow model.
        
        Args:
            input_dim (int): Dimension of the quantum amplitude space
            latent_dim (int): Dimension of the latent space
            hidden_dim (int): Dimension of hidden layers
            flow_length (int): Number of flow transformations
            flow_type (str): Type of flow transformation ('planar' or 'radial')
        """
        super(AmplitudeFlow, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder network (quantum state -> latent)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # *2 for complex input
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Normalizing flow in latent space
        self.flow = NormalizingFlow(latent_dim, flow_length, flow_type)
        
        # Decoder network (latent -> quantum state)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim * 2)  # *2 for complex output
        )
    
    def encode(self, x):
        """
        Encode quantum state to latent space.
        
        Args:
            x (torch.Tensor): Quantum state tensor
            
        Returns:
            torch.Tensor: Latent representation
        """
        return self.encoder(x)
    
    def decode(self, z):
        """
        Decode from latent space to quantum state.
        
        Args:
            z (torch.Tensor): Latent representation
            
        Returns:
            torch.Tensor: Quantum state tensor
        """
        return self.decoder(z)
    
    def forward(self, x, n_samples=1):
        """
        Forward pass through the amplitude flow model.
        
        Args:
            x (torch.Tensor): Input quantum state tensor
            n_samples (int): Number of samples to generate per input
            
        Returns:
            tuple: (reconstructed_x, latent_samples, log_prob)
        """
        # Encode to latent space
        z_mean = self.encode(x)
        
        # Sample from latent space using the flow
        z_samples = []
        log_probs = []
        
        for _ in range(n_samples):
            # Sample from base distribution
            eps = torch.randn_like(z_mean)
            
            # Apply flow transformations
            z_sample, log_prob = self.flow(eps)
            
            # Add to collections
            z_samples.append(z_sample)
            log_probs.append(log_prob)
        
        # Stack samples
        z_samples = torch.stack(z_samples, dim=1)  # [batch_size, n_samples, latent_dim]
        log_probs = torch.stack(log_probs, dim=1)  # [batch_size, n_samples]
        
        # Reshape for decoding
        batch_size = x.shape[0]
        z_samples_flat = z_samples.view(-1, self.latent_dim)
        
        # Decode samples
        decoded_flat = self.decode(z_samples_flat)
        decoded = decoded_flat.view(batch_size, n_samples, -1)
        
        return decoded, z_samples, log_probs
    
    def sample_posterior(self, x, n_samples=1):
        """
        Sample from the posterior distribution of quantum states.
        
        Args:
            x (torch.Tensor): Conditioning quantum state tensor
            n_samples (int): Number of samples to generate
            
        Returns:
            torch.Tensor: Sampled quantum states
        """
        with torch.no_grad():
            # Encode to latent space
            z_mean = self.encode(x)
            
            # Sample from base distribution
            eps = torch.randn(x.shape[0], n_samples, self.latent_dim, device=x.device)
            
            # Apply flow transformations
            z_samples = []
            for i in range(n_samples):
                z_sample, _ = self.flow(eps[:, i, :])
                z_samples.append(z_sample)
            
            # Stack samples
            z_samples = torch.stack(z_samples, dim=1)  # [batch_size, n_samples, latent_dim]
            
            # Reshape for decoding
            batch_size = x.shape[0]
            z_samples_flat = z_samples.view(-1, self.latent_dim)
            
            # Decode samples
            decoded_flat = self.decode(z_samples_flat)
            decoded = decoded_flat.view(batch_size, n_samples, -1)
            
            return decoded
    
    def sample_prior(self, n_samples=1, device='cpu'):
        """
        Sample from the prior distribution of quantum states.
        
        Args:
            n_samples (int): Number of samples to generate
            device (str): Device to generate samples on
            
        Returns:
            torch.Tensor: Sampled quantum states
        """
        with torch.no_grad():
            # Sample from the flow
            z = self.flow.sample(n_samples, device)
            
            # Decode to quantum states
            samples = self.decode(z)
            
            # Normalize the quantum states
            real_part = samples[:, :self.input_dim]
            imag_part = samples[:, self.input_dim:]
            
            norm = torch.sqrt(torch.sum(real_part**2 + imag_part**2, dim=1, keepdim=True))
            real_part = real_part / norm
            imag_part = imag_part / norm
            
            # Recombine
            normalized_samples = torch.cat([real_part, imag_part], dim=1)
            
            return normalized_samples


class AmplitudeFlowTrainer:
    """
    Trainer for the Amplitude Flow model.
    """
    def __init__(self, model, learning_rate=1e-3, kl_weight=0.01):
        """
        Initialize the trainer.
        
        Args:
            model (AmplitudeFlow): The amplitude flow model
            learning_rate (float): Learning rate for optimization
            kl_weight (float): Weight for the KL divergence term
        """
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.kl_weight = kl_weight
        self.device = next(model.parameters()).device
    
    def loss_function(self, recon_x, x, log_prob):
        """
        Loss function for the amplitude flow model.
        
        Args:
            recon_x (torch.Tensor): Reconstructed quantum states
            x (torch.Tensor): Original quantum states
            log_prob (torch.Tensor): Log probability from the flow
            
        Returns:
            tuple: (total_loss, reconstruction_loss, flow_loss)
        """
        # Reconstruction loss (mean squared error)
        recon_loss = F.mse_loss(recon_x, x.unsqueeze(1).expand_as(recon_x), reduction='sum')
        
        # Flow loss (negative log likelihood)
        flow_loss = -torch.sum(log_prob)
        
        # Total loss
        total_loss = recon_loss + self.kl_weight * flow_loss
        
        return total_loss, recon_loss, flow_loss
    
    def train_epoch(self, dataloader, n_samples=1):
        """
        Train for one epoch.
        
        Args:
            dataloader (DataLoader): DataLoader for training data
            n_samples (int): Number of samples to generate per input
            
        Returns:
            tuple: (average_loss, average_recon_loss, average_flow_loss)
        """
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_flow_loss = 0
        
        for batch_idx, data in enumerate(dataloader):
            data = data[0].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            recon_batch, _, log_prob = self.model(data, n_samples)
            
            # Compute loss
            loss, recon_loss, flow_loss = self.loss_function(recon_batch, data, log_prob)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_flow_loss += flow_loss.item()
        
        # Compute averages
        avg_loss = total_loss / len(dataloader.dataset)
        avg_recon_loss = total_recon_loss / len(dataloader.dataset)
        avg_flow_loss = total_flow_loss / len(dataloader.dataset)
        
        return avg_loss, avg_recon_loss, avg_flow_loss
    
    def train(self, dataloader, epochs=100, n_samples=1, print_every=10):
        """
        Train the amplitude flow model.
        
        Args:
            dataloader (DataLoader): DataLoader for training data
            epochs (int): Number of training epochs
            n_samples (int): Number of samples to generate per input
            print_every (int): Print loss every print_every epochs
            
        Returns:
            dict: Training history
        """
        history = {
            'loss': [],
            'recon_loss': [],
            'flow_loss': []
        }
        
        for epoch in range(epochs):
            avg_loss, avg_recon_loss, avg_flow_loss = self.train_epoch(dataloader, n_samples)
            
            # Store history
            history['loss'].append(avg_loss)
            history['recon_loss'].append(avg_recon_loss)
            history['flow_loss'].append(avg_flow_loss)
            
            # Print progress
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, "
                      f"Recon Loss: {avg_recon_loss:.6f}, Flow Loss: {avg_flow_loss:.6f}")
        
        # Save the model
        torch.save(self.model.state_dict(), "amplitude_flow_model.pt")
        
        return history
    
    def visualize_samples(self, dataloader, n_samples=10):
        """
        Visualize samples from the amplitude flow model.
        
        Args:
            dataloader (DataLoader): DataLoader containing quantum states
            n_samples (int): Number of samples to generate per input
        """
        self.model.eval()
        
        # Get a batch from the dataloader
        data_iter = iter(dataloader)
        data = next(data_iter)[0][:5].to(self.device)
        
        # Generate samples
        with torch.no_grad():
            samples = self.model.sample_posterior(data, n_samples)
        
        # Convert to numpy
        data = data.cpu().numpy()
        samples = samples.cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(5, n_samples + 1, figsize=(2 * (n_samples + 1), 10))
        
        for i in range(5):
            # Original
            input_dim = data.shape[1] // 2
            real_part = data[i, :input_dim]
            imag_part = data[i, input_dim:]
            prob = real_part**2 + imag_part**2
            
            axes[i, 0].plot(prob)
            axes[i, 0].set_title(f"Original {i+1}")
            
            # Samples
            for j in range(n_samples):
                real_part = samples[i, j, :input_dim]
                imag_part = samples[i, j, input_dim:]
                prob = real_part**2 + imag_part**2
                
                axes[i, j+1].plot(prob)
                axes[i, j+1].set_title(f"Sample {j+1}")
        
        plt.tight_layout()
        plt.savefig('amplitude_flow_samples.png')
        plt.close()
    
    def visualize_prior_samples(self, n_samples=10):
        """
        Visualize samples from the prior distribution.
        
        Args:
            n_samples (int): Number of samples to generate
        """
        self.model.eval()
        
        # Generate samples from the prior
        with torch.no_grad():
            samples = self.model.sample_prior(n_samples, self.device)
        
        # Convert to numpy
        samples = samples.cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(n_samples, 1, figsize=(10, 2 * n_samples))
        
        for i in range(n_samples):
            # Extract real and imaginary parts
            input_dim = samples.shape[1] // 2
            real_part = samples[i, :input_dim]
            imag_part = samples[i, input_dim:]
            
            # Calculate probability density
            prob = real_part**2 + imag_part**2
            
            axes[i].plot(np.linspace(-5, 5, input_dim), prob)
            axes[i].set_title(f"Prior Sample {i+1}")
            axes[i].set_xlabel("Position")
            axes[i].set_ylabel("Probability")
        
        plt.tight_layout()
        plt.savefig('amplitude_flow_prior_samples.png')
        plt.close()
    
    def visualize_interpolation(self, dataloader, n_steps=10):
        """
        Visualize interpolation between quantum states in latent space.
        
        Args:
            dataloader (DataLoader): DataLoader containing quantum states
            n_steps (int): Number of interpolation steps
        """
        self.model.eval()
        
        # Get two quantum states from the dataloader
        data_iter = iter(dataloader)
        data = next(data_iter)[0][:2].to(self.device)
        
        # Encode to latent space
        with torch.no_grad():
            z1 = self.model.encode(data[0:1])
            z2 = self.model.encode(data[1:2])
            
            # Interpolate in latent space
            alphas = torch.linspace(0, 1, n_steps, device=self.device)
            interpolated_z = []
            
            for alpha in alphas:
                z_interp = (1 - alpha) * z1 + alpha * z2
                interpolated_z.append(z_interp)
            
            # Stack interpolated points
            interpolated_z = torch.cat(interpolated_z, dim=0)
            
            # Decode to quantum states
            interpolated_states = self.model.decode(interpolated_z)
        
        # Convert to numpy
        interpolated_states = interpolated_states.cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(n_steps, 1, figsize=(10, 2 * n_steps))
        
        for i in range(n_steps):
            # Extract real and imaginary parts
            input_dim = interpolated_states.shape[1] // 2
            real_part = interpolated_states[i, :input_dim]
            imag_part = interpolated_states[i, input_dim:]
            
            # Calculate probability density
            prob = real_part**2 + imag_part**2
            
            axes[i].plot(np.linspace(-5, 5, input_dim), prob)
            axes[i].set_title(f"Interpolation Step {i+1}/{n_steps}")
            axes[i].set_xlabel("Position")
            axes[i].set_ylabel("Probability")
        
        plt.tight_layout()
        plt.savefig('amplitude_flow_interpolation.png')
        plt.close()


def main():
    """Train and evaluate an Amplitude Flow model."""
    print("=" * 80)
    print("Normalizing Flow for Quantum Amplitude Distributions")
    print("=" * 80)
    
    # Prepare dataset
    print("\nPreparing quantum dataset...")
    from quantum_autoencoder import prepare_quantum_dataset
    dataset, dataloader = prepare_quantum_dataset(n_states=1000, n_points=128)
    
    print(f"Generated {len(dataset)} quantum states for training")
    
    # Create the amplitude flow model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 128  # Number of spatial grid points
    latent_dim = 8   # Dimension of the latent space
    
    model = AmplitudeFlow(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=128,
        flow_length=8,
        flow_type='planar'
    ).to(device)
    
    print(f"Created Amplitude Flow model with latent dimension {latent_dim}")
    
    # Create trainer
    trainer = AmplitudeFlowTrainer(
        model=model,
        learning_rate=1e-3,
        kl_weight=0.01
    )
    
    # Train the model
    print("\nTraining the Amplitude Flow model...")
    history = trainer.train(dataloader, epochs=100, n_samples=5, print_every=10)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Total Loss')
    plt.plot(history['recon_loss'], label='Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['flow_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Flow Loss')
    plt.title('Flow Loss')
    
    plt.tight_layout()
    plt.savefig('amplitude_flow_training_history.png')
    plt.close()
    
    print("Training complete. History plot saved to 'amplitude_flow_training_history.png'")
    
    # Visualize posterior samples
    print("\nVisualizing posterior samples...")
    trainer.visualize_samples(dataloader, n_samples=5)
    print("Posterior samples visualization saved to 'amplitude_flow_samples.png'")
    
    # Visualize prior samples
    print("\nVisualizing prior samples...")
    trainer.visualize_prior_samples(n_samples=10)
    print("Prior samples visualization saved to 'amplitude_flow_prior_samples.png'")
    
    # Visualize interpolation
    print("\nVisualizing interpolation in latent space...")
    trainer.visualize_interpolation(dataloader, n_steps=10)
    print("Interpolation visualization saved to 'amplitude_flow_interpolation.png'")
    
    print("\nAmplitude Flow demonstration complete!")


if __name__ == "__main__":
    main()
