"""
Latent Quantum Models Demo

This script demonstrates the combined use of Variational Autoencoders (VAEs)
and Normalizing Flows for quantum state processing.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from quantum_harmonic_oscillator import QuantumHarmonicOscillator
from quantum_autoencoder import QuantumVariationalAutoencoder, prepare_quantum_dataset
from normalizing_flow import AmplitudeFlow, NormalizingFlow


class LatentQuantumModels:
    """
    Class for demonstrating and comparing different latent space models
    for quantum state processing.
    """
    def __init__(self, input_dim=128, latent_dim=8, hidden_dim=128):
        """
        Initialize the latent quantum models.
        
        Args:
            input_dim (int): Dimension of the quantum state
            latent_dim (int): Dimension of the latent space
            hidden_dim (int): Dimension of hidden layers
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create the VAE model
        self.vae = QuantumVariationalAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            complex_input=True
        ).to(self.device)
        
        # Create the Amplitude Flow model
        self.flow = AmplitudeFlow(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            flow_length=8,
            flow_type='planar'
        ).to(self.device)
    
    def load_models(self, vae_path="quantum_vae_model.pt", flow_path="amplitude_flow_model.pt"):
        """
        Load pretrained models.
        
        Args:
            vae_path (str): Path to the VAE model
            flow_path (str): Path to the Amplitude Flow model
        """
        try:
            self.vae.load_state_dict(torch.load(vae_path, map_location=self.device))
            print(f"Loaded VAE model from {vae_path}")
        except:
            print(f"Could not load VAE model from {vae_path}")
        
        try:
            self.flow.load_state_dict(torch.load(flow_path, map_location=self.device))
            print(f"Loaded Amplitude Flow model from {flow_path}")
        except:
            print(f"Could not load Amplitude Flow model from {flow_path}")
    
    def compare_reconstructions(self, dataloader, n_samples=5):
        """
        Compare reconstructions from VAE and Amplitude Flow.
        
        Args:
            dataloader (DataLoader): DataLoader containing quantum states
            n_samples (int): Number of samples to visualize
        """
        self.vae.eval()
        self.flow.eval()
        
        # Get samples from the dataloader
        data_iter = iter(dataloader)
        data = next(data_iter)[0][:n_samples].to(self.device)
        
        # Generate reconstructions
        with torch.no_grad():
            # VAE reconstruction
            vae_recon, _, _ = self.vae(data)
            
            # Flow reconstruction (using posterior mean)
            flow_recon, _, _ = self.flow(data, n_samples=1)
            flow_recon = flow_recon.squeeze(1)  # Remove sample dimension
        
        # Convert to numpy
        data = data.cpu().numpy()
        vae_recon = vae_recon.cpu().numpy()
        flow_recon = flow_recon.cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(n_samples, 3, figsize=(15, 3 * n_samples))
        
        for i in range(n_samples):
            # Extract real and imaginary parts for original
            input_dim = data.shape[1] // 2
            real_part = data[i, :input_dim]
            imag_part = data[i, input_dim:]
            prob = real_part**2 + imag_part**2
            
            axes[i, 0].plot(np.linspace(-5, 5, input_dim), prob)
            axes[i, 0].set_title(f"Original {i+1}")
            
            # VAE reconstruction
            real_part = vae_recon[i, :input_dim]
            imag_part = vae_recon[i, input_dim:]
            prob = real_part**2 + imag_part**2
            
            axes[i, 1].plot(np.linspace(-5, 5, input_dim), prob)
            axes[i, 1].set_title(f"VAE Reconstruction {i+1}")
            
            # Flow reconstruction
            real_part = flow_recon[i, :input_dim]
            imag_part = flow_recon[i, input_dim:]
            prob = real_part**2 + imag_part**2
            
            axes[i, 2].plot(np.linspace(-5, 5, input_dim), prob)
            axes[i, 2].set_title(f"Flow Reconstruction {i+1}")
        
        plt.tight_layout()
        plt.savefig('model_comparison_reconstructions.png')
        plt.close()
    
    def compare_samples(self, n_samples=5):
        """
        Compare samples generated from VAE and Amplitude Flow.
        
        Args:
            n_samples (int): Number of samples to generate
        """
        self.vae.eval()
        self.flow.eval()
        
        # Generate samples
        with torch.no_grad():
            # VAE samples
            vae_samples = self.vae.sample(n_samples, self.device)
            
            # Flow samples
            flow_samples = self.flow.sample_prior(n_samples, self.device)
        
        # Convert to numpy
        vae_samples = vae_samples.cpu().numpy()
        flow_samples = flow_samples.cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(n_samples, 2, figsize=(10, 3 * n_samples))
        
        for i in range(n_samples):
            # VAE sample
            input_dim = vae_samples.shape[1] // 2
            real_part = vae_samples[i, :input_dim]
            imag_part = vae_samples[i, input_dim:]
            prob = real_part**2 + imag_part**2
            
            axes[i, 0].plot(np.linspace(-5, 5, input_dim), prob)
            axes[i, 0].set_title(f"VAE Sample {i+1}")
            
            # Flow sample
            real_part = flow_samples[i, :input_dim]
            imag_part = flow_samples[i, input_dim:]
            prob = real_part**2 + imag_part**2
            
            axes[i, 1].plot(np.linspace(-5, 5, input_dim), prob)
            axes[i, 1].set_title(f"Flow Sample {i+1}")
        
        plt.tight_layout()
        plt.savefig('model_comparison_samples.png')
        plt.close()
    
    def visualize_latent_spaces(self, dataloader, n_samples=500):
        """
        Visualize and compare the latent spaces of VAE and Amplitude Flow.
        
        Args:
            dataloader (DataLoader): DataLoader containing quantum states
            n_samples (int): Number of samples to encode
        """
        self.vae.eval()
        self.flow.eval()
        
        # Get samples from the dataloader
        all_vae_mu = []
        all_flow_z = []
        
        with torch.no_grad():
            for batch_idx, data in enumerate(dataloader):
                if len(all_vae_mu) * data[0].shape[0] >= n_samples:
                    break
                    
                data = data[0].to(self.device)
                
                # VAE encoding
                mu, _ = self.vae.encode(data)
                all_vae_mu.append(mu.cpu().numpy())
                
                # Flow encoding
                flow_z = self.flow.encode(data)
                all_flow_z.append(flow_z.cpu().numpy())
        
        # Concatenate
        all_vae_mu = np.concatenate(all_vae_mu, axis=0)[:n_samples]
        all_flow_z = np.concatenate(all_flow_z, axis=0)[:n_samples]
        
        # Create figure for 2D visualization
        if self.latent_dim >= 2:
            plt.figure(figsize=(12, 6))
            
            # Plot VAE latent space
            plt.subplot(1, 2, 1)
            plt.scatter(all_vae_mu[:, 0], all_vae_mu[:, 1], alpha=0.5)
            plt.xlabel('Latent Dimension 1')
            plt.ylabel('Latent Dimension 2')
            plt.title('VAE Latent Space')
            
            # Plot Flow latent space
            plt.subplot(1, 2, 2)
            plt.scatter(all_flow_z[:, 0], all_flow_z[:, 1], alpha=0.5)
            plt.xlabel('Latent Dimension 1')
            plt.ylabel('Latent Dimension 2')
            plt.title('Flow Latent Space')
            
            plt.tight_layout()
            plt.savefig('latent_space_comparison.png')
            plt.close()
        
        # Create figure for distribution visualization
        n_dims = min(self.latent_dim, 4)  # Plot at most 4 dimensions
        fig, axes = plt.subplots(n_dims, 2, figsize=(12, 3 * n_dims))
        
        for i in range(n_dims):
            # VAE latent distribution
            axes[i, 0].hist(all_vae_mu[:, i], bins=30, alpha=0.7)
            axes[i, 0].set_title(f'VAE Latent Dim {i+1}')
            
            # Flow latent distribution
            axes[i, 1].hist(all_flow_z[:, i], bins=30, alpha=0.7)
            axes[i, 1].set_title(f'Flow Latent Dim {i+1}')
        
        plt.tight_layout()
        plt.savefig('latent_distribution_comparison.png')
        plt.close()
    
    def latent_space_arithmetic(self, dataloader):
        """
        Demonstrate latent space arithmetic with quantum states.
        
        Args:
            dataloader (DataLoader): DataLoader containing quantum states
        """
        self.vae.eval()
        self.flow.eval()
        
        # Get three quantum states from the dataloader
        data_iter = iter(dataloader)
        data = next(data_iter)[0][:3].to(self.device)
        
        # Encode to latent space
        with torch.no_grad():
            # VAE encoding
            vae_z1, _ = self.vae.encode(data[0:1])
            vae_z2, _ = self.vae.encode(data[1:2])
            vae_z3, _ = self.vae.encode(data[2:3])
            
            # Flow encoding
            flow_z1 = self.flow.encode(data[0:1])
            flow_z2 = self.flow.encode(data[1:2])
            flow_z3 = self.flow.encode(data[2:3])
            
            # Perform arithmetic: z_result = z1 - z2 + z3
            vae_z_result = vae_z1 - vae_z2 + vae_z3
            flow_z_result = flow_z1 - flow_z2 + flow_z3
            
            # Decode results
            vae_result = self.vae.decode(vae_z_result)
            flow_result = self.flow.decode(flow_z_result)
        
        # Convert to numpy
        data = data.cpu().numpy()
        vae_result = vae_result.cpu().numpy()
        flow_result = flow_result.cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Plot original states
        for i in range(3):
            # Extract real and imaginary parts
            input_dim = data.shape[1] // 2
            real_part = data[i, :input_dim]
            imag_part = data[i, input_dim:]
            prob = real_part**2 + imag_part**2
            
            axes[0, i].plot(np.linspace(-5, 5, input_dim), prob)
            axes[0, i].set_title(f"State {i+1}")
        
        # Plot arithmetic results
        # VAE result
        input_dim = vae_result.shape[1] // 2
        real_part = vae_result[0, :input_dim]
        imag_part = vae_result[0, input_dim:]
        prob = real_part**2 + imag_part**2
        
        axes[0, 3].plot(np.linspace(-5, 5, input_dim), prob)
        axes[0, 3].set_title("VAE: State 1 - State 2 + State 3")
        
        # Flow result
        real_part = flow_result[0, :input_dim]
        imag_part = flow_result[0, input_dim:]
        prob = real_part**2 + imag_part**2
        
        axes[1, 3].plot(np.linspace(-5, 5, input_dim), prob)
        axes[1, 3].set_title("Flow: State 1 - State 2 + State 3")
        
        # Add equation visualization
        axes[1, 0].text(0.5, 0.5, "State 1", ha='center', va='center', fontsize=14)
        axes[1, 0].set_xticks([])
        axes[1, 0].set_yticks([])
        
        axes[1, 1].text(0.5, 0.5, "- State 2", ha='center', va='center', fontsize=14)
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
        
        axes[1, 2].text(0.5, 0.5, "+ State 3", ha='center', va='center', fontsize=14)
        axes[1, 2].set_xticks([])
        axes[1, 2].set_yticks([])
        
        plt.tight_layout()
        plt.savefig('latent_space_arithmetic.png')
        plt.close()
    
    def generate_quantum_superpositions(self, dataloader, n_states=3, weights=None):
        """
        Generate quantum superpositions using latent space interpolation.
        
        Args:
            dataloader (DataLoader): DataLoader containing quantum states
            n_states (int): Number of states to superpose
            weights (list): Weights for the superposition (normalized if not provided)
        """
        self.vae.eval()
        self.flow.eval()
        
        # Get quantum states from the dataloader
        data_iter = iter(dataloader)
        data = next(data_iter)[0][:n_states].to(self.device)
        
        # Default to equal weights if not provided
        if weights is None:
            weights = [1.0 / n_states] * n_states
        else:
            # Normalize weights
            total = sum(weights)
            weights = [w / total for w in weights]
        
        # Encode to latent space
        with torch.no_grad():
            # VAE encoding
            vae_zs = []
            for i in range(n_states):
                mu, _ = self.vae.encode(data[i:i+1])
                vae_zs.append(mu)
            
            # Flow encoding
            flow_zs = []
            for i in range(n_states):
                z = self.flow.encode(data[i:i+1])
                flow_zs.append(z)
            
            # Create weighted superposition in latent space
            vae_z_super = torch.zeros_like(vae_zs[0])
            flow_z_super = torch.zeros_like(flow_zs[0])
            
            for i in range(n_states):
                vae_z_super += weights[i] * vae_zs[i]
                flow_z_super += weights[i] * flow_zs[i]
            
            # Decode superpositions
            vae_super = self.vae.decode(vae_z_super)
            flow_super = self.flow.decode(flow_z_super)
            
            # Also create direct superposition in wavefunction space
            direct_super = torch.zeros_like(data[0:1])
            for i in range(n_states):
                direct_super += weights[i] * data[i:i+1]
            
            # Normalize the direct superposition
            input_dim = direct_super.shape[1] // 2
            real_part = direct_super[:, :input_dim]
            imag_part = direct_super[:, input_dim:]
            norm = torch.sqrt(torch.sum(real_part**2 + imag_part**2, dim=1, keepdim=True))
            direct_super = torch.cat([real_part / norm, imag_part / norm], dim=1)
        
        # Convert to numpy
        data = data.cpu().numpy()
        vae_super = vae_super.cpu().numpy()
        flow_super = flow_super.cpu().numpy()
        direct_super = direct_super.cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(n_states + 3, 1, figsize=(10, 3 * (n_states + 3)))
        
        # Plot original states
        for i in range(n_states):
            # Extract real and imaginary parts
            input_dim = data.shape[1] // 2
            real_part = data[i, :input_dim]
            imag_part = data[i, input_dim:]
            prob = real_part**2 + imag_part**2
            
            axes[i].plot(np.linspace(-5, 5, input_dim), prob)
            axes[i].set_title(f"State {i+1} (weight: {weights[i]:.2f})")
        
        # Plot superpositions
        # Direct superposition
        input_dim = direct_super.shape[1] // 2
        real_part = direct_super[0, :input_dim]
        imag_part = direct_super[0, input_dim:]
        prob = real_part**2 + imag_part**2
        
        axes[n_states].plot(np.linspace(-5, 5, input_dim), prob)
        axes[n_states].set_title("Direct Wavefunction Superposition")
        
        # VAE superposition
        input_dim = vae_super.shape[1] // 2
        real_part = vae_super[0, :input_dim]
        imag_part = vae_super[0, input_dim:]
        prob = real_part**2 + imag_part**2
        
        axes[n_states + 1].plot(np.linspace(-5, 5, input_dim), prob)
        axes[n_states + 1].set_title("VAE Latent Space Superposition")
        
        # Flow superposition
        input_dim = flow_super.shape[1] // 2
        real_part = flow_super[0, :input_dim]
        imag_part = flow_super[0, input_dim:]
        prob = real_part**2 + imag_part**2
        
        axes[n_states + 2].plot(np.linspace(-5, 5, input_dim), prob)
        axes[n_states + 2].set_title("Flow Latent Space Superposition")
        
        plt.tight_layout()
        plt.savefig('quantum_superpositions.png')
        plt.close()


def main():
    """Demonstrate the combined use of VAE and Normalizing Flow for quantum states."""
    print("=" * 80)
    print("Latent Quantum Models: VAE and Normalizing Flow")
    print("=" * 80)
    
    # Prepare dataset
    print("\nPreparing quantum dataset...")
    dataset, dataloader = prepare_quantum_dataset(n_states=1000, n_points=128)
    
    print(f"Generated {len(dataset)} quantum states for demonstration")
    
    # Create the latent quantum models
    models = LatentQuantumModels(input_dim=128, latent_dim=8, hidden_dim=128)
    
    # Try to load pretrained models
    models.load_models()
    
    # Compare reconstructions
    print("\nComparing reconstructions from VAE and Amplitude Flow...")
    models.compare_reconstructions(dataloader, n_samples=5)
    print("Reconstruction comparison saved to 'model_comparison_reconstructions.png'")
    
    # Compare samples
    print("\nComparing samples from VAE and Amplitude Flow...")
    models.compare_samples(n_samples=5)
    print("Sample comparison saved to 'model_comparison_samples.png'")
    
    # Visualize latent spaces
    print("\nVisualizing latent spaces...")
    models.visualize_latent_spaces(dataloader, n_samples=500)
    print("Latent space visualizations saved to 'latent_space_comparison.png' and 'latent_distribution_comparison.png'")
    
    # Demonstrate latent space arithmetic
    print("\nDemonstrating latent space arithmetic...")
    models.latent_space_arithmetic(dataloader)
    print("Latent space arithmetic visualization saved to 'latent_space_arithmetic.png'")
    
    # Generate quantum superpositions
    print("\nGenerating quantum superpositions...")
    models.generate_quantum_superpositions(dataloader, n_states=3, weights=[0.5, 0.3, 0.2])
    print("Quantum superpositions visualization saved to 'quantum_superpositions.png'")
    
    print("\nLatent quantum models demonstration complete!")


if __name__ == "__main__":
    main()
