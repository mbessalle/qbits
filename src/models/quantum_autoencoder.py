"""
Quantum Autoencoder Module

This module implements variational autoencoders (VAEs) for learning
compressed latent representations of quantum states.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from quantum_harmonic_oscillator import QuantumHarmonicOscillator


class QuantumVariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder for quantum states.
    Learns a compressed latent representation of quantum wavefunctions.
    """
    def __init__(self, input_dim, latent_dim=8, hidden_dim=64, complex_input=True):
        """
        Initialize the Quantum Variational Autoencoder.
        
        Args:
            input_dim (int): Dimension of the input quantum state
            latent_dim (int): Dimension of the latent space
            hidden_dim (int): Dimension of hidden layers
            complex_input (bool): Whether the input is complex (represented as [real; imag])
        """
        super(QuantumVariationalAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.complex_input = complex_input
        
        # Determine actual input dimension (doubled for complex inputs)
        actual_input_dim = input_dim * 2 if complex_input else input_dim
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(actual_input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        
        # Mean and log variance layers for VAE
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, actual_input_dim),
        )
    
    def encode(self, x):
        """
        Encode the input to the latent space.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            tuple: (mu, logvar) parameters of the latent distribution
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from the latent distribution.
        
        Args:
            mu (torch.Tensor): Mean of the latent distribution
            logvar (torch.Tensor): Log variance of the latent distribution
            
        Returns:
            torch.Tensor: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """
        Decode from the latent space to the input space.
        
        Args:
            z (torch.Tensor): Latent vector
            
        Returns:
            torch.Tensor: Reconstructed input
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        Forward pass through the VAE.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            tuple: (reconstructed_x, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, logvar
    
    def sample(self, n_samples=1, device='cpu'):
        """
        Generate samples from the latent space.
        
        Args:
            n_samples (int): Number of samples to generate
            device (str): Device to generate samples on
            
        Returns:
            torch.Tensor: Generated samples
        """
        with torch.no_grad():
            # Sample from the latent space
            z = torch.randn(n_samples, self.latent_dim, device=device)
            samples = self.decode(z)
            
            # If complex input, reshape to complex format
            if self.complex_input:
                real_part = samples[:, :self.input_dim]
                imag_part = samples[:, self.input_dim:]
                
                # Normalize the wavefunction
                norm = torch.sqrt(torch.sum(real_part**2 + imag_part**2, dim=1, keepdim=True))
                real_part = real_part / norm
                imag_part = imag_part / norm
                
                # Recombine
                samples = torch.cat([real_part, imag_part], dim=1)
            
            return samples


class QuantumAutoencoderTrainer:
    """
    Trainer for the Quantum Variational Autoencoder.
    """
    def __init__(self, model, learning_rate=1e-3, kl_weight=0.01):
        """
        Initialize the trainer.
        
        Args:
            model (QuantumVariationalAutoencoder): The VAE model
            learning_rate (float): Learning rate for optimization
            kl_weight (float): Weight for the KL divergence term
        """
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.kl_weight = kl_weight
        self.device = next(model.parameters()).device
    
    def loss_function(self, recon_x, x, mu, logvar):
        """
        VAE loss function: reconstruction loss + KL divergence.
        
        Args:
            recon_x (torch.Tensor): Reconstructed input
            x (torch.Tensor): Original input
            mu (torch.Tensor): Mean of the latent distribution
            logvar (torch.Tensor): Log variance of the latent distribution
            
        Returns:
            tuple: (total_loss, reconstruction_loss, kl_divergence)
        """
        # Reconstruction loss (mean squared error)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + self.kl_weight * kl_div
        
        return total_loss, recon_loss, kl_div
    
    def train_epoch(self, dataloader):
        """
        Train for one epoch.
        
        Args:
            dataloader (DataLoader): DataLoader for training data
            
        Returns:
            tuple: (average_loss, average_recon_loss, average_kl_div)
        """
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_div = 0
        
        for batch_idx, data in enumerate(dataloader):
            data = data[0].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            
            # Compute loss
            loss, recon_loss, kl_div = self.loss_function(recon_batch, data, mu, logvar)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_div += kl_div.item()
        
        # Compute averages
        avg_loss = total_loss / len(dataloader.dataset)
        avg_recon_loss = total_recon_loss / len(dataloader.dataset)
        avg_kl_div = total_kl_div / len(dataloader.dataset)
        
        return avg_loss, avg_recon_loss, avg_kl_div
    
    def train(self, dataloader, epochs=100, print_every=10):
        """
        Train the VAE.
        
        Args:
            dataloader (DataLoader): DataLoader for training data
            epochs (int): Number of training epochs
            print_every (int): Print loss every print_every epochs
            
        Returns:
            dict: Training history
        """
        history = {
            'loss': [],
            'recon_loss': [],
            'kl_div': []
        }
        
        for epoch in range(epochs):
            avg_loss, avg_recon_loss, avg_kl_div = self.train_epoch(dataloader)
            
            # Store history
            history['loss'].append(avg_loss)
            history['recon_loss'].append(avg_recon_loss)
            history['kl_div'].append(avg_kl_div)
            
            # Print progress
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, "
                      f"Recon Loss: {avg_recon_loss:.6f}, KL Div: {avg_kl_div:.6f}")
        
        # Save the model
        torch.save(self.model.state_dict(), "quantum_vae_model.pt")
        
        return history
    
    def visualize_reconstruction(self, dataloader, n_samples=5, complex_input=True):
        """
        Visualize original and reconstructed quantum states.
        
        Args:
            dataloader (DataLoader): DataLoader containing quantum states
            n_samples (int): Number of samples to visualize
            complex_input (bool): Whether the input is complex
        """
        self.model.eval()
        
        # Get samples from the dataloader
        data_iter = iter(dataloader)
        samples = next(data_iter)[0][:n_samples].to(self.device)
        
        # Reconstruct
        with torch.no_grad():
            recon_samples, _, _ = self.model(samples)
        
        # Convert to numpy
        samples = samples.cpu().numpy()
        recon_samples = recon_samples.cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(n_samples, 2, figsize=(10, 2 * n_samples))
        
        for i in range(n_samples):
            # Original
            if complex_input:
                # For complex inputs, plot probability density
                input_dim = samples.shape[1] // 2
                real_part = samples[i, :input_dim]
                imag_part = samples[i, input_dim:]
                prob = real_part**2 + imag_part**2
                
                axes[i, 0].plot(prob)
                axes[i, 0].set_title(f"Original {i+1}")
                
                # Reconstructed
                real_part = recon_samples[i, :input_dim]
                imag_part = recon_samples[i, input_dim:]
                prob = real_part**2 + imag_part**2
                
                axes[i, 1].plot(prob)
                axes[i, 1].set_title(f"Reconstructed {i+1}")
            else:
                # For real inputs, plot directly
                axes[i, 0].plot(samples[i])
                axes[i, 0].set_title(f"Original {i+1}")
                
                axes[i, 1].plot(recon_samples[i])
                axes[i, 1].set_title(f"Reconstructed {i+1}")
        
        plt.tight_layout()
        plt.savefig('vae_reconstruction.png')
        plt.close()
    
    def visualize_latent_space(self, dataloader, n_samples=1000):
        """
        Visualize the latent space of the VAE.
        
        Args:
            dataloader (DataLoader): DataLoader containing quantum states
            n_samples (int): Number of samples to encode
        """
        self.model.eval()
        
        # Get samples from the dataloader
        all_mu = []
        all_logvar = []
        
        with torch.no_grad():
            for batch_idx, data in enumerate(dataloader):
                if len(all_mu) * data[0].shape[0] >= n_samples:
                    break
                    
                data = data[0].to(self.device)
                mu, logvar = self.model.encode(data)
                
                all_mu.append(mu.cpu().numpy())
                all_logvar.append(logvar.cpu().numpy())
        
        # Concatenate
        all_mu = np.concatenate(all_mu, axis=0)[:n_samples]
        all_logvar = np.concatenate(all_logvar, axis=0)[:n_samples]
        
        # Create figure
        if self.model.latent_dim >= 2:
            plt.figure(figsize=(10, 8))
            
            # Plot the first two dimensions of the latent space
            plt.scatter(all_mu[:, 0], all_mu[:, 1], alpha=0.5)
            plt.xlabel('Latent Dimension 1')
            plt.ylabel('Latent Dimension 2')
            plt.title('Latent Space Visualization')
            
            plt.savefig('vae_latent_space.png')
            plt.close()
        
        # Plot the distribution of each latent dimension
        n_dims = min(self.model.latent_dim, 8)  # Plot at most 8 dimensions
        fig, axes = plt.subplots(n_dims, 1, figsize=(10, 2 * n_dims))
        
        for i in range(n_dims):
            if n_dims == 1:
                ax = axes
            else:
                ax = axes[i]
                
            ax.hist(all_mu[:, i], bins=30, alpha=0.5, label='μ')
            
            # Generate samples from the learned distribution
            std = np.exp(0.5 * all_logvar[:, i])
            samples = np.random.normal(all_mu[:, i], std)
            
            ax.hist(samples, bins=30, alpha=0.5, label='Samples')
            ax.set_title(f'Latent Dimension {i+1}')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('vae_latent_distributions.png')
        plt.close()


def prepare_quantum_dataset(n_states=1000, n_points=128, x_range=(-5, 5)):
    """
    Prepare a dataset of quantum states for training the VAE.
    
    Args:
        n_states (int): Number of quantum states to generate
        n_points (int): Number of spatial grid points
        x_range (tuple): Range of x values (min, max)
        
    Returns:
        tuple: (dataset, dataloader)
    """
    # Create quantum harmonic oscillator
    qho = QuantumHarmonicOscillator(n_points=n_points, x_range=x_range)
    
    # Generate different initial states
    all_states = []
    
    for i in range(n_states):
        # Vary the initial state parameters
        x0 = np.random.uniform(-3, 3)
        sigma = np.random.uniform(0.3, 1.5)
        
        # Create initial state
        psi_0 = qho.initial_state(x0=x0, sigma=sigma)
        
        # Evolve for a random time
        t_max = np.random.uniform(0, 10)
        t_points, psi_t = qho.evolve(psi_0, t_max=t_max, n_steps=2)
        
        # Add final state to the collection
        all_states.append(psi_t[-1])
    
    # Convert to tensor format
    tensor_states = []
    
    for psi in all_states:
        # Extract real and imaginary parts
        psi_real = np.real(psi)
        psi_imag = np.imag(psi)
        
        # Combine into a single array
        psi_combined = np.concatenate([psi_real, psi_imag])
        
        tensor_states.append(psi_combined)
    
    # Convert to tensor
    tensor_states = torch.tensor(np.array(tensor_states), dtype=torch.float32)
    
    # Create dataset and dataloader
    dataset = TensorDataset(tensor_states)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    return dataset, dataloader


def main():
    """Train and evaluate a Quantum Variational Autoencoder."""
    print("=" * 80)
    print("Quantum Variational Autoencoder")
    print("=" * 80)
    
    # Prepare dataset
    print("\nPreparing quantum dataset...")
    dataset, dataloader = prepare_quantum_dataset(n_states=1000, n_points=128)
    
    print(f"Generated {len(dataset)} quantum states for training")
    
    # Create the VAE model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 128  # Number of spatial grid points
    latent_dim = 8   # Dimension of the latent space
    
    model = QuantumVariationalAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=128,
        complex_input=True
    ).to(device)
    
    print(f"Created Quantum VAE with latent dimension {latent_dim}")
    
    # Create trainer
    trainer = QuantumAutoencoderTrainer(
        model=model,
        learning_rate=1e-3,
        kl_weight=0.01
    )
    
    # Train the model
    print("\nTraining the Quantum VAE...")
    history = trainer.train(dataloader, epochs=100, print_every=10)
    
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
    plt.plot(history['kl_div'])
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence')
    plt.title('KL Divergence')
    
    plt.tight_layout()
    plt.savefig('vae_training_history.png')
    plt.close()
    
    print("Training complete. History plot saved to 'vae_training_history.png'")
    
    # Visualize reconstructions
    print("\nVisualizing reconstructions...")
    trainer.visualize_reconstruction(dataloader, n_samples=5)
    print("Reconstruction visualization saved to 'vae_reconstruction.png'")
    
    # Visualize latent space
    print("\nVisualizing latent space...")
    trainer.visualize_latent_space(dataloader)
    print("Latent space visualization saved to 'vae_latent_space.png' and 'vae_latent_distributions.png'")
    
    # Generate new quantum states
    print("\nGenerating new quantum states from the latent space...")
    new_states = model.sample(n_samples=5, device=device)
    
    # Plot the generated states
    plt.figure(figsize=(10, 6))
    
    for i in range(5):
        # Extract real and imaginary parts
        real_part = new_states[i, :input_dim].cpu().numpy()
        imag_part = new_states[i, input_dim:].cpu().numpy()
        
        # Calculate probability density
        prob = real_part**2 + imag_part**2
        
        plt.subplot(5, 1, i+1)
        plt.plot(np.linspace(-5, 5, input_dim), prob)
        plt.title(f"Generated Quantum State {i+1}")
        plt.xlabel("Position")
        plt.ylabel("Probability")
    
    plt.tight_layout()
    plt.savefig('vae_generated_states.png')
    plt.close()
    
    print("Generated states visualization saved to 'vae_generated_states.png'")
    print("\nQuantum VAE demonstration complete!")


if __name__ == "__main__":
    main()
