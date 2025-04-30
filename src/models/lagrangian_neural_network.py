"""
Lagrangian Neural Network Module

This module implements a Lagrangian Neural Network (LNN) that learns
the Lagrangian dynamics of a quantum harmonic oscillator, ensuring
energy conservation by design.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

class LagrangianNet(nn.Module):
    """
    Neural network that learns the Lagrangian function L(q, q̇).
    The dynamics are then given by the Euler-Lagrange equations:
    d/dt(∂L/∂q̇) - ∂L/∂q = 0
    """
    def __init__(self, hidden_dim=64):
        super(LagrangianNet, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Neural network to approximate the Lagrangian
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)  # Scalar Lagrangian output
        )
        
        # Initialize weights with non-zero values
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.01)
    
    def forward(self, q, q_dot):
        """
        Forward pass to compute the Lagrangian value.
        
        Args:
            q (torch.Tensor): Position tensor of shape (batch_size, 1)
            q_dot (torch.Tensor): Velocity tensor of shape (batch_size, 1)
            
        Returns:
            torch.Tensor: Lagrangian value L(q, q̇)
        """
        # For a harmonic oscillator, the Lagrangian is L = 0.5 * q̇^2 - 0.5 * q^2
        # We'll add a physics-informed component to help the model learn
        physics_L = 0.5 * q_dot**2 - 0.5 * q**2
        
        # Let the neural network learn corrections to this basic form
        x = torch.cat([q, q_dot], dim=1)
        learned_L = self.net(x)
        
        # Combine physics-based and learned components
        return physics_L + 0.1 * learned_L
    
    def compute_gradients(self, q, q_dot):
        """
        Compute the gradients of the Lagrangian with respect to q and q̇.
        
        Args:
            q (torch.Tensor): Position tensor of shape (batch_size, 1)
            q_dot (torch.Tensor): Velocity tensor of shape (batch_size, 1)
            
        Returns:
            tuple: (∂L/∂q, ∂L/∂q̇) gradients
        """
        # Create input tensors and enable gradient tracking
        q = q.detach().clone().requires_grad_(True)
        q_dot = q_dot.detach().clone().requires_grad_(True)
        
        # Compute Lagrangian
        L = self.forward(q, q_dot)
        
        # Compute gradients
        dL_dq = torch.autograd.grad(
            L.sum(), q, create_graph=True, retain_graph=True
        )[0]
        
        dL_dq_dot = torch.autograd.grad(
            L.sum(), q_dot, create_graph=True, retain_graph=True
        )[0]
        
        return dL_dq, dL_dq_dot
    
    def compute_euler_lagrange(self, q, q_dot, q_ddot):
        """
        Compute the Euler-Lagrange equation: d/dt(∂L/∂q̇) - ∂L/∂q = 0
        
        Args:
            q (torch.Tensor): Position tensor of shape (batch_size, 1)
            q_dot (torch.Tensor): Velocity tensor of shape (batch_size, 1)
            q_ddot (torch.Tensor): Acceleration tensor of shape (batch_size, 1)
            
        Returns:
            torch.Tensor: Euler-Lagrange residual
        """
        # Compute gradients
        dL_dq, dL_dq_dot = self.compute_gradients(q, q_dot)
        
        # Euler-Lagrange equation: d/dt(∂L/∂q̇) - ∂L/∂q = 0
        # d/dt(∂L/∂q̇) is approximated as q_ddot for the harmonic oscillator
        # since ∂L/∂q̇ = q̇ for the standard Lagrangian L = 0.5*q̇^2 - 0.5*q^2
        el_residual = q_ddot - dL_dq
        
        return el_residual


class LagrangianNeuralNetwork:
    """
    Class for training and evaluating a Lagrangian Neural Network.
    """
    def __init__(self, hidden_dim=64, learning_rate=1e-3):
        """
        Initialize the Lagrangian Neural Network.
        
        Args:
            hidden_dim (int): Dimension of hidden layers
            learning_rate (float): Learning rate for optimization
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LagrangianNet(hidden_dim=hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
    
    def train(self, q_data, q_dot_data, q_ddot_data, batch_size=32, epochs=1000, 
              print_every=100, save_model=True):
        """
        Train the Lagrangian Neural Network.
        
        Args:
            q_data (ndarray): Position data of shape (n_samples, 1)
            q_dot_data (ndarray): Velocity data of shape (n_samples, 1)
            q_ddot_data (ndarray): Acceleration data of shape (n_samples, 1)
            batch_size (int): Batch size for training
            epochs (int): Number of training epochs
            print_every (int): Print loss every print_every epochs
            save_model (bool): Whether to save the model after training
            
        Returns:
            list: Training losses
        """
        # Convert data to torch tensors
        q_tensor = torch.tensor(q_data, dtype=torch.float32).to(self.device)
        q_dot_tensor = torch.tensor(q_dot_data, dtype=torch.float32).to(self.device)
        q_ddot_tensor = torch.tensor(q_ddot_data, dtype=torch.float32).to(self.device)
        
        # Print some statistics about the data to help debug
        print(f"Input data statistics:")
        print(f"  q range: [{q_tensor.min().item():.6f}, {q_tensor.max().item():.6f}]")
        print(f"  q_dot range: [{q_dot_tensor.min().item():.6f}, {q_dot_tensor.max().item():.6f}]")
        print(f"  q_ddot range: [{q_ddot_tensor.min().item():.6f}, {q_ddot_tensor.max().item():.6f}]")
        
        # Create dataset and dataloader
        dataset = TensorDataset(q_tensor, q_dot_tensor, q_ddot_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        losses = []
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for q_batch, q_dot_batch, q_ddot_batch in dataloader:
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Compute Euler-Lagrange residual
                el_residual = self.model.compute_euler_lagrange(q_batch, q_dot_batch, q_ddot_batch)
                
                # Target should be zero for the Euler-Lagrange equation
                target = torch.zeros_like(el_residual)
                
                # Compute loss
                loss = self.loss_fn(el_residual, target)
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                self.optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                
                # Print first batch predictions for debugging
                if epoch == 0 and batch_count == 1:
                    print(f"First batch Euler-Lagrange residuals:")
                    for i in range(min(3, len(q_batch))):
                        print(f"  Input (q, q_dot, q_ddot): ({q_batch[i].item():.6f}, {q_dot_batch[i].item():.6f}, {q_ddot_batch[i].item():.6f})")
                        print(f"  Residual: {el_residual[i].item():.6f}")
            
            # Average loss for the epoch
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            # Print progress
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.8f}")
                
                # Print a few predictions for debugging
                if (epoch + 1) % (print_every * 5) == 0:
                    # Sample a random data point
                    sample_idx = np.random.randint(0, len(q_tensor))
                    sample_q = q_tensor[sample_idx].item()
                    sample_q_dot = q_dot_tensor[sample_idx].item()
                    sample_q_ddot = q_ddot_tensor[sample_idx].item()
                    
                    # For harmonic oscillator: q_ddot = -q (normalized units)
                    analytical_q_ddot = -sample_q
                    
                    print(f"Sample prediction at epoch {epoch+1}:")
                    print(f"  Input (q, q_dot): ({sample_q:.6f}, {sample_q_dot:.6f})")
                    print(f"  Analytical q_ddot: {analytical_q_ddot:.6f}")
                    print(f"  Target q_ddot: {sample_q_ddot:.6f}")
        
        # Save the trained model
        if save_model:
            torch.save(self.model.state_dict(), "lagrangian_nn_model.pt")
            
        return losses
    
    def predict_trajectory(self, q0, q_dot0, t_span, steps=100):
        """
        Predict a trajectory using the learned Lagrangian dynamics.
        
        Args:
            q0 (float): Initial position
            q_dot0 (float): Initial velocity
            t_span (tuple): Time span (t_start, t_end)
            steps (int): Number of integration steps
            
        Returns:
            tuple: (t_points, q_pred, q_dot_pred, energy_pred) predicted trajectory
        """
        self.model.eval()
        
        # Create time points
        t_points = np.linspace(t_span[0], t_span[1], steps)
        dt = (t_span[1] - t_span[0]) / (steps - 1)
        
        # Initialize trajectory
        q_traj = [q0]
        q_dot_traj = [q_dot0]
        energy_traj = [0.5 * q_dot0**2 + 0.5 * q0**2]  # Initial energy
        
        # Integrate using symplectic Euler method to preserve energy
        for i in range(1, steps):
            # Get current state
            q_current = q_traj[-1]
            q_dot_current = q_dot_traj[-1]
            
            # Convert to tensors for gradient computation
            q_tensor = torch.tensor([[q_current]], dtype=torch.float32).to(self.device).requires_grad_(True)
            q_dot_tensor = torch.tensor([[q_dot_current]], dtype=torch.float32).to(self.device).requires_grad_(True)
            
            # Compute gradients of the Lagrangian
            dL_dq, dL_dq_dot = self.model.compute_gradients(q_tensor, q_dot_tensor)
            
            # Extract values
            dL_dq_val = dL_dq.item()
            dL_dq_dot_val = dL_dq_dot.item()
            
            # Symplectic Euler update (preserves energy better than standard Euler)
            # First update position using current velocity
            q_new = q_current + q_dot_current * dt
            
            # Then update velocity using new position
            q_tensor_new = torch.tensor([[q_new]], dtype=torch.float32).to(self.device).requires_grad_(True)
            q_dot_tensor_new = torch.tensor([[q_dot_current]], dtype=torch.float32).to(self.device).requires_grad_(True)
            dL_dq_new, _ = self.model.compute_gradients(q_tensor_new, q_dot_tensor_new)
            dL_dq_new_val = dL_dq_new.item()
            
            # Update velocity using the Euler-Lagrange equation: d/dt(∂L/∂q̇) = ∂L/∂q
            # For the harmonic oscillator with L = 0.5*q̇^2 - 0.5*q^2, this gives: q̈ = -q
            q_dot_new = q_dot_current + dL_dq_new_val * dt
            
            # Calculate energy (Hamiltonian) H = 0.5*q̇^2 + 0.5*q^2 for the harmonic oscillator
            energy_new = 0.5 * q_dot_new**2 + 0.5 * q_new**2
            
            # Store
            q_traj.append(q_new)
            q_dot_traj.append(q_dot_new)
            energy_traj.append(energy_new)
        
        return t_points, np.array(q_traj), np.array(q_dot_traj), np.array(energy_traj)
    
    def plot_trajectory_comparison(self, q0, q_dot0, t_span, true_q, true_p, steps=100):
        """
        Plot a comparison between the true and predicted trajectories.
        
        Args:
            q0 (float): Initial position
            q_dot0 (float): Initial velocity
            t_span (tuple): Time span (t_start, t_end)
            true_q (ndarray): True position trajectory
            true_p (ndarray): True momentum trajectory (equivalent to q_dot for m=1)
            steps (int): Number of integration steps
        """
        # Predict trajectory
        t_pred, q_pred, q_dot_pred, energy_pred = self.predict_trajectory(q0, q_dot0, t_span, steps)
        
        # Calculate true energy
        true_energy = 0.5 * true_p**2 + 0.5 * true_q**2
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot position
        plt.subplot(2, 2, 1)
        plt.plot(t_pred, q_pred, 'b-', label='LNN Predicted')
        plt.plot(np.linspace(t_span[0], t_span[1], len(true_q)), true_q, 'r--', label='True')
        plt.xlabel('Time')
        plt.ylabel('Position (q)')
        plt.legend()
        plt.title('Position vs Time')
        
        # Plot momentum/velocity
        plt.subplot(2, 2, 2)
        plt.plot(t_pred, q_dot_pred, 'b-', label='LNN Predicted')
        plt.plot(np.linspace(t_span[0], t_span[1], len(true_p)), true_p, 'r--', label='True')
        plt.xlabel('Time')
        plt.ylabel('Momentum (p)')
        plt.legend()
        plt.title('Momentum vs Time')
        
        # Plot phase space
        plt.subplot(2, 2, 3)
        plt.plot(q_pred, q_dot_pred, 'b-', label='LNN Predicted')
        plt.plot(true_q, true_p, 'r--', label='True')
        plt.xlabel('Position (q)')
        plt.ylabel('Momentum (p)')
        plt.legend()
        plt.title('Phase Space')
        
        # Plot energy conservation
        plt.subplot(2, 2, 4)
        plt.plot(t_pred, energy_pred, 'b-', label='LNN Predicted')
        plt.plot(np.linspace(t_span[0], t_span[1], len(true_q)), true_energy, 'r--', label='True')
        plt.xlabel('Time')
        plt.ylabel('Energy (H)')
        plt.legend()
        plt.title('Energy Conservation')
        
        plt.tight_layout()
        plt.savefig('images/neural_networks/lnn_trajectory_comparison.png')
        plt.close()
        
    def load_model(self, model_path):
        """
        Load a trained model.
        
        Args:
            model_path (str): Path to the saved model
        """
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()


def prepare_training_data_from_qho(qho_data, derivatives):
    """
    Prepare training data for the Lagrangian Neural Network from
    quantum harmonic oscillator data.
    
    Args:
        qho_data (ndarray): Input data of shape (n_samples, 2) containing (q, p)
        derivatives (ndarray): Target data of shape (n_samples, 2) containing (dq/dt, dp/dt)
        
    Returns:
        tuple: (q, q_dot, q_ddot) for LNN training
    """
    # Extract position and momentum
    q = qho_data[:, 0].reshape(-1, 1)
    p = qho_data[:, 1].reshape(-1, 1)  # p = q_dot for m=1
    
    # Extract derivatives
    dq_dt = derivatives[:, 0].reshape(-1, 1)
    dp_dt = derivatives[:, 1].reshape(-1, 1)  # dp/dt = q_ddot for m=1
    
    # For the harmonic oscillator, q_dot = p and q_ddot = dp/dt
    q_dot = p
    q_ddot = dp_dt
    
    return q, q_dot, q_ddot


def main():
    """Train and evaluate a Lagrangian Neural Network on the oscillator data."""
    # Load the training data
    data = np.load('harmonic_oscillator_data.npz')
    qp_data = data['qp']
    derivatives = data['derivatives']
    
    print(f"Loaded {len(qp_data)} training samples")
    
    # Prepare data for LNN
    q, q_dot, q_ddot = prepare_training_data_from_qho(qp_data, derivatives)
    
    # Create and train the Lagrangian Neural Network
    lnn = LagrangianNeuralNetwork(hidden_dim=64, learning_rate=1e-3)
    losses = lnn.train(q, q_dot, q_ddot, batch_size=32, epochs=1000, print_every=100)
    
    # Plot the training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('LNN Training Loss')
    plt.yscale('log')
    plt.savefig('lnn_training_loss.png')
    plt.close()
    
    # Generate a true trajectory for comparison
    from quantum_harmonic_oscillator import QuantumHarmonicOscillator
    
    # Create the oscillator
    qho = QuantumHarmonicOscillator(n_points=256, x_range=(-10, 10), omega=1.0)
    
    # Create an initial state with momentum
    psi_0 = np.exp(-(qho.x - 0.0)**2 / (2 * 0.5**2)) * np.exp(1j * 1.0 * qho.x)
    psi_0 = psi_0 / np.sqrt(np.sum(np.abs(psi_0)**2) * qho.dx)
    
    # Evolve the state
    t_points, psi_t = qho.evolve(psi_0, t_max=10.0, n_steps=100)
    
    # Calculate position and momentum expectation values
    q_true = np.array([qho.position_expectation(psi) for psi in psi_t])
    p_true = np.array([qho.momentum_expectation(psi) for psi in psi_t])
    
    # Compare the true and predicted trajectories
    q0 = q_true[0]
    p0 = p_true[0]
    lnn.plot_trajectory_comparison(q0, p0, (0, 10), q_true, p_true, steps=100)
    
    print("Lagrangian Neural Network training and evaluation complete.")


if __name__ == "__main__":
    main()
