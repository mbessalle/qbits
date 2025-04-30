"""
Fourier Basis and Unitary Transformations Demo

This script demonstrates the application of Fourier and unitary transformations
as basis changes in quantum systems and neural networks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from quantum_harmonic_oscillator import QuantumHarmonicOscillator
from unitary_transforms import (
    FourierTransformLayer, 
    UnitaryLayer, 
    MeasurementBasisChange,
    FourierRecurrentUnit
)

class UnitaryHamiltonianNet(nn.Module):
    """
    Neural network that learns the Hamiltonian dynamics using unitary layers
    for basis transformations.
    """
    def __init__(self, input_dim=2, hidden_dim=64, n_unitary_layers=2):
        super(UnitaryHamiltonianNet, self).__init__()
        
        self.input_dim = input_dim
        
        # Initial embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Unitary layers for basis transformations
        self.unitary_layers = nn.ModuleList([
            UnitaryLayer(hidden_dim, complex_input=False, init='random')
            for _ in range(n_unitary_layers)
        ])
        
        # Nonlinear layers between unitary transformations
        self.nonlinear_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()
            )
            for _ in range(n_unitary_layers)
        ])
        
        # Output layer for Hamiltonian
        self.output_layer = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        """
        Forward pass to compute the Hamiltonian value.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim) containing (q, p)
            
        Returns:
            torch.Tensor: Hamiltonian value H(q, p)
        """
        # Initial embedding
        h = self.embedding(x)
        
        # Apply alternating unitary and nonlinear layers
        for unitary, nonlinear in zip(self.unitary_layers, self.nonlinear_layers):
            # Apply unitary transformation (basis change)
            h = unitary(h)
            # Apply nonlinear transformation in the new basis
            h = nonlinear(h)
            
        # Output Hamiltonian value
        return self.output_layer(h)
    
    def compute_gradients(self, q, p):
        """
        Compute the gradients of the Hamiltonian with respect to q and p.
        
        Args:
            q (torch.Tensor): Position tensor of shape (batch_size, 1)
            p (torch.Tensor): Momentum tensor of shape (batch_size, 1)
            
        Returns:
            tuple: (dH/dq, dH/dp) gradients
        """
        # Create input tensor and enable gradient tracking
        qp = torch.cat([q, p], dim=1)
        qp.requires_grad_(True)
        
        # Compute Hamiltonian
        H = self.forward(qp)
        
        # Compute gradients
        dH = torch.autograd.grad(
            H.sum(), qp, create_graph=True, retain_graph=True
        )[0]
        
        # Extract gradients
        dH_dq = dH[:, 0:1]
        dH_dp = dH[:, 1:2]
        
        return dH_dq, dH_dp
    
    def dynamics(self, qp):
        """
        Compute the Hamiltonian dynamics.
        
        Args:
            qp (torch.Tensor): Input tensor of shape (batch_size, 2) containing (q, p)
            
        Returns:
            torch.Tensor: Time derivatives (dq/dt, dp/dt)
        """
        q = qp[:, 0:1]
        p = qp[:, 1:2]
        
        # Compute gradients
        dH_dq, dH_dp = self.compute_gradients(q, p)
        
        # Hamiltonian dynamics
        dq_dt = dH_dp
        dp_dt = -dH_dq
        
        # Concatenate derivatives
        dqp_dt = torch.cat([dq_dt, dp_dt], dim=1)
        
        return dqp_dt
    
    def project_unitaries(self):
        """
        Project all unitary layers to the orthogonal group.
        This should be called during training to maintain the unitary constraint.
        """
        for layer in self.unitary_layers:
            layer.project_to_unitary()


class FourierBasisMeasurement:
    """
    Class for performing measurements in different bases using Fourier and unitary transformations.
    """
    def __init__(self, n_points=128, x_range=(-5, 5), n_bases=4):
        """
        Initialize the Fourier basis measurement system.
        
        Args:
            n_points (int): Number of spatial grid points
            x_range (tuple): Range of x values (min, max)
            n_bases (int): Number of measurement bases
        """
        self.n_points = n_points
        self.x_range = x_range
        self.x = np.linspace(x_range[0], x_range[1], n_points)
        self.dx = (x_range[1] - x_range[0]) / (n_points - 1)
        
        # For momentum space
        self.k = 2 * np.pi * np.fft.fftfreq(n_points, self.dx)
        
        # Create measurement basis change module
        self.mbc = MeasurementBasisChange(n_points, n_bases=n_bases, complex_input=True)
    
    def wavefunction_to_tensor(self, psi):
        """
        Convert a complex wavefunction to a tensor representation.
        
        Args:
            psi (ndarray): Complex wavefunction
            
        Returns:
            torch.Tensor: Tensor representation [real; imag]
        """
        # Extract real and imaginary parts
        psi_real = np.real(psi)
        psi_imag = np.imag(psi)
        
        # Combine into a single array
        psi_combined = np.concatenate([psi_real, psi_imag])
        
        # Convert to tensor
        return torch.tensor(psi_combined, dtype=torch.float32).unsqueeze(0)
    
    def measure_in_multiple_bases(self, psi):
        """
        Measure the wavefunction in multiple bases.
        
        Args:
            psi (ndarray): Complex wavefunction
            
        Returns:
            dict: Probability distributions in different bases
        """
        # Convert to tensor
        psi_tensor = self.wavefunction_to_tensor(psi)
        
        # Measure in different bases
        results = {}
        
        # Position basis (original)
        results['position'] = np.abs(psi)**2
        
        # Momentum basis (Fourier transform)
        results['momentum'] = self.mbc.measure_in_fourier_basis(psi_tensor).squeeze().detach().numpy()[:self.n_points]
        
        # Other unitary bases
        for i in range(self.mbc.n_bases):
            basis_name = f'basis_{i}'
            prob = self.mbc.measure_in_basis(psi_tensor, i).squeeze().detach().numpy()[:self.n_points]
            results[basis_name] = prob
        
        return results
    
    def visualize_basis_measurements(self, psi, title="Wavefunction in Different Bases"):
        """
        Visualize the wavefunction measurements in different bases.
        
        Args:
            psi (ndarray): Complex wavefunction
            title (str): Plot title
        """
        # Measure in different bases
        measurements = self.measure_in_multiple_bases(psi)
        
        # Number of subplots
        n_plots = len(measurements)
        
        # Create figure
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots))
        
        # Plot each measurement
        for i, (basis_name, prob) in enumerate(measurements.items()):
            ax = axes[i]
            
            if basis_name == 'position':
                ax.plot(self.x, prob)
                ax.set_xlabel('Position')
            elif basis_name == 'momentum':
                ax.plot(self.k, prob)
                ax.set_xlabel('Momentum')
            else:
                # For other bases, use indices
                ax.plot(np.arange(len(prob)), prob)
                ax.set_xlabel('Basis Index')
            
            ax.set_ylabel('Probability')
            ax.set_title(f'{basis_name.capitalize()} Basis')
        
        plt.tight_layout()
        plt.suptitle(title, y=1.02, fontsize=16)
        plt.savefig('basis_measurements.png')
        plt.close()


def train_unitary_hamiltonian_net(qp_data, derivatives, epochs=500, batch_size=32, 
                                 hidden_dim=64, n_unitary_layers=2, 
                                 learning_rate=1e-3, project_every=10):
    """
    Train a Hamiltonian Neural Network with unitary layers.
    
    Args:
        qp_data (ndarray): Input data of shape (n_samples, 2) containing (q, p)
        derivatives (ndarray): Target data of shape (n_samples, 2) containing (dq/dt, dp/dt)
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        hidden_dim (int): Dimension of hidden layers
        n_unitary_layers (int): Number of unitary layers
        learning_rate (float): Learning rate for optimization
        project_every (int): Project to unitary group every project_every epochs
        
    Returns:
        tuple: (trained_model, losses)
    """
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = UnitaryHamiltonianNet(
        input_dim=2, 
        hidden_dim=hidden_dim, 
        n_unitary_layers=n_unitary_layers
    ).to(device)
    
    # Loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Convert data to tensors
    qp_tensor = torch.tensor(qp_data, dtype=torch.float32).to(device)
    deriv_tensor = torch.tensor(derivatives, dtype=torch.float32).to(device)
    
    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(qp_tensor, deriv_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for qp_batch, deriv_batch in dataloader:
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            pred_derivatives = model.dynamics(qp_batch)
            
            # Compute loss
            loss = loss_fn(pred_derivatives, deriv_batch)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Accumulate loss
            epoch_loss += loss.item()
        
        # Project to unitary group periodically
        if (epoch + 1) % project_every == 0:
            model.project_unitaries()
        
        # Average loss for the epoch
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        # Print progress
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    # Final projection
    model.project_unitaries()
    
    # Save the model
    torch.save(model.state_dict(), "unitary_hamiltonian_nn_model.pt")
    
    return model, losses


def animate_basis_changes(qho, t_points, psi_t, n_frames=100):
    """
    Create an animation of wavefunction evolution in different bases.
    
    Args:
        qho (QuantumHarmonicOscillator): Quantum harmonic oscillator instance
        t_points (ndarray): Time points
        psi_t (ndarray): Evolved wavefunctions with shape (n_steps, n_points)
        n_frames (int): Number of frames in the animation
    """
    # Create Fourier basis measurement system
    fbm = FourierBasisMeasurement(
        n_points=qho.n_points, 
        x_range=qho.x_range, 
        n_bases=3
    )
    
    # Select frames to animate
    frame_indices = np.linspace(0, len(t_points)-1, n_frames, dtype=int)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    # Initialize plots
    lines = []
    for ax in axes:
        line, = ax.plot([], [])
        lines.append(line)
        ax.set_ylim(0, 1)
    
    # Set titles and axes
    axes[0].set_title('Position Basis')
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Probability')
    axes[0].set_xlim(qho.x_range)
    
    axes[1].set_title('Momentum Basis')
    axes[1].set_xlabel('Momentum')
    axes[1].set_ylabel('Probability')
    axes[1].set_xlim(min(fbm.k), max(fbm.k))
    
    axes[2].set_title('Basis 1')
    axes[2].set_xlabel('Basis Index')
    axes[2].set_ylabel('Probability')
    axes[2].set_xlim(0, qho.n_points)
    
    axes[3].set_title('Basis 2')
    axes[3].set_xlabel('Basis Index')
    axes[3].set_ylabel('Probability')
    axes[3].set_xlim(0, qho.n_points)
    
    # Add time annotation
    time_text = fig.text(0.5, 0.95, '', ha='center')
    
    def init():
        """Initialize the animation."""
        for line in lines:
            line.set_data([], [])
        time_text.set_text('')
        return lines + [time_text]
    
    def update(frame):
        """Update the animation for each frame."""
        # Get wavefunction at this frame
        idx = frame_indices[frame]
        psi = psi_t[idx]
        t = t_points[idx]
        
        # Measure in different bases
        measurements = fbm.measure_in_multiple_bases(psi)
        
        # Update position basis plot
        lines[0].set_data(qho.x, measurements['position'])
        
        # Update momentum basis plot
        lines[1].set_data(fbm.k, measurements['momentum'])
        
        # Update other basis plots
        lines[2].set_data(np.arange(qho.n_points), measurements['basis_0'])
        lines[3].set_data(np.arange(qho.n_points), measurements['basis_1'])
        
        # Update time text
        time_text.set_text(f'Time: {t:.2f}')
        
        return lines + [time_text]
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True)
    
    # Save animation
    anim.save('basis_changes_animation.gif', writer='pillow', fps=10)
    
    plt.close()
    
    print("Animation saved to 'basis_changes_animation.gif'")


def main():
    """Main function to demonstrate Fourier and unitary transformations."""
    print("=" * 80)
    print("Fourier and Unitary Transformations as Basis Changes")
    print("=" * 80)
    
    # Step 1: Create a quantum harmonic oscillator
    print("\nStep 1: Creating a quantum harmonic oscillator...")
    qho = QuantumHarmonicOscillator(n_points=128, x_range=(-5, 5), omega=1.0)
    
    # Create an initial state
    psi_0 = qho.initial_state(x0=-1.0, sigma=0.5)
    
    # Evolve the state
    t_points, psi_t = qho.evolve(psi_0, t_max=10.0, n_steps=100)
    
    # Step 2: Demonstrate measurement in different bases
    print("\nStep 2: Demonstrating measurement in different bases...")
    fbm = FourierBasisMeasurement(n_points=qho.n_points, x_range=qho.x_range, n_bases=4)
    
    # Visualize measurements at different time points
    for t_idx in [0, 25, 50, 75, 99]:
        psi = psi_t[t_idx]
        fbm.visualize_basis_measurements(
            psi, 
            title=f"Wavefunction in Different Bases at t={t_points[t_idx]:.2f}"
        )
    
    print("Basis measurements visualized and saved.")
    
    # Step 3: Create an animation of basis changes
    print("\nStep 3: Creating an animation of basis changes...")
    animate_basis_changes(qho, t_points, psi_t, n_frames=50)
    
    # Step 4: Generate training data for the Hamiltonian Neural Network
    print("\nStep 4: Generating training data for the Hamiltonian Neural Network...")
    qp_data, derivatives = qho.generate_training_data(n_initial_states=10, n_steps=100)
    
    print(f"Generated {len(qp_data)} training samples")
    
    # Step 5: Train a Hamiltonian Neural Network with unitary layers
    print("\nStep 5: Training a Hamiltonian Neural Network with unitary layers...")
    model, losses = train_unitary_hamiltonian_net(
        qp_data, 
        derivatives, 
        epochs=500, 
        batch_size=32, 
        hidden_dim=64, 
        n_unitary_layers=3, 
        learning_rate=1e-3, 
        project_every=10
    )
    
    # Plot the training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss for Unitary Hamiltonian Network')
    plt.yscale('log')
    plt.savefig('unitary_hnn_training_loss.png')
    plt.close()
    
    print("Unitary Hamiltonian Neural Network training complete.")
    print("Training loss plot saved to 'unitary_hnn_training_loss.png'")
    
    # Step 6: Compare with standard Hamiltonian Neural Network
    print("\nStep 6: Comparing with standard Hamiltonian Neural Network...")
    
    # Load the standard HNN model
    from hamiltonian_neural_network import HamiltonianNeuralNetwork
    standard_hnn = HamiltonianNeuralNetwork(hidden_dim=64, learning_rate=1e-3)
    standard_losses = standard_hnn.train(
        qp_data, 
        derivatives, 
        batch_size=32, 
        epochs=500, 
        print_every=100,
        save_model=True
    )
    
    # Plot comparison of losses
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Unitary HNN')
    plt.plot(standard_losses, label='Standard HNN')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.yscale('log')
    plt.legend()
    plt.savefig('hnn_comparison.png')
    plt.close()
    
    print("Comparison plot saved to 'hnn_comparison.png'")
    
    # Step 7: Demonstrate Fourier Recurrent Unit
    print("\nStep 7: Demonstrating Fourier Recurrent Unit...")
    
    # Create a simple time series from the quantum oscillator
    q_series = np.array([qho.position_expectation(psi) for psi in psi_t])
    p_series = np.array([qho.momentum_expectation(psi) for psi in psi_t])
    
    # Combine into a single time series
    time_series = np.stack([q_series, p_series], axis=1)
    
    # Convert to tensor
    time_series_tensor = torch.tensor(time_series, dtype=torch.float32).unsqueeze(0)
    
    # Create FRU
    fru = FourierRecurrentUnit(
        input_dim=2,
        hidden_dim=16,
        output_dim=2,
        n_fourier_features=8
    )
    
    # Forward pass
    outputs, hidden = fru(time_series_tensor)
    
    # Convert to numpy
    outputs_np = outputs.detach().numpy()[0]
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(t_points, q_series, 'b-', label='True')
    plt.plot(t_points, outputs_np[:, 0], 'r--', label='FRU')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.legend()
    plt.title('Position Prediction with Fourier Recurrent Unit')
    
    plt.subplot(2, 1, 2)
    plt.plot(t_points, p_series, 'b-', label='True')
    plt.plot(t_points, outputs_np[:, 1], 'r--', label='FRU')
    plt.xlabel('Time')
    plt.ylabel('Momentum')
    plt.legend()
    plt.title('Momentum Prediction with Fourier Recurrent Unit')
    
    plt.tight_layout()
    plt.savefig('fourier_recurrent_unit.png')
    plt.close()
    
    print("Fourier Recurrent Unit demonstration complete.")
    print("Plot saved to 'fourier_recurrent_unit.png'")
    
    print("\nAll demonstrations complete!")


if __name__ == "__main__":
    main()
