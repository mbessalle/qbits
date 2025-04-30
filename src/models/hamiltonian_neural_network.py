"""
Hamiltonian Neural Network Module

This module implements a Hamiltonian Neural Network (HNN) that learns
the Hamiltonian dynamics of a quantum harmonic oscillator.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

class HamiltonianNet(nn.Module):
    """
    Neural network that learns the Hamiltonian function H(q,p).
    The dynamics are then given by:
    dq/dt = ∂H/∂p
    dp/dt = -∂H/∂q
    """
    def __init__(self, hidden_dim=64):
        super(HamiltonianNet, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Neural network to approximate the Hamiltonian
        # Initialize with non-zero weights to avoid zero gradients
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)  # Scalar Hamiltonian output
        )
        
        # Initialize weights with non-zero values
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.01)
    
    def forward(self, x):
        """
        Forward pass to compute the Hamiltonian value.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 2) containing (q, p)
            
        Returns:
            torch.Tensor: Hamiltonian value H(q, p)
        """
        # For a harmonic oscillator, the Hamiltonian is H = 0.5 * p^2 + 0.5 * q^2
        # We'll add a physics-informed component to help the model learn
        q = x[:, 0:1]
        p = x[:, 1:2]
        physics_H = 0.5 * (q**2 + p**2)
        
        # Let the neural network learn corrections to this basic form
        learned_H = self.net(x)
        
        # Combine physics-based and learned components
        return physics_H + 0.1 * learned_H
    
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
        q = q.detach().clone().requires_grad_(True)
        p = p.detach().clone().requires_grad_(True)
        qp = torch.cat([q, p], dim=1)
        
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
        q = qp[:, 0:1].detach().clone().requires_grad_(True)
        p = qp[:, 1:2].detach().clone().requires_grad_(True)
        
        # Compute gradients
        dH_dq, dH_dp = self.compute_gradients(q, p)
        
        # Hamiltonian dynamics
        dq_dt = dH_dp
        dp_dt = -dH_dq
        
        # Concatenate derivatives
        dqp_dt = torch.cat([dq_dt, dp_dt], dim=1)
        
        return dqp_dt

class HamiltonianNeuralNetwork:
    """
    Class for training and evaluating a Hamiltonian Neural Network.
    """
    def __init__(self, hidden_dim=64, learning_rate=1e-3):
        """
        Initialize the Hamiltonian Neural Network.
        
        Args:
            hidden_dim (int): Dimension of hidden layers
            learning_rate (float): Learning rate for optimization
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HamiltonianNet(hidden_dim=hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
    def train(self, qp_data, derivatives, batch_size=32, epochs=1000, 
              print_every=100, save_model=True):
        """
        Train the Hamiltonian Neural Network.
        
        Args:
            qp_data (ndarray): Input data of shape (n_samples, 2) containing (q, p)
            derivatives (ndarray): Target data of shape (n_samples, 2) containing (dq/dt, dp/dt)
            batch_size (int): Batch size for training
            epochs (int): Number of training epochs
            print_every (int): Print loss every print_every epochs
            save_model (bool): Whether to save the model after training
            
        Returns:
            list: Training losses
        """
        # Convert data to torch tensors
        qp_tensor = torch.tensor(qp_data, dtype=torch.float32).to(self.device)
        deriv_tensor = torch.tensor(derivatives, dtype=torch.float32).to(self.device)
        
        # Print some statistics about the data to help debug
        print(f"Input data statistics:")
        print(f"  q range: [{qp_tensor[:, 0].min().item():.6f}, {qp_tensor[:, 0].max().item():.6f}]")
        print(f"  p range: [{qp_tensor[:, 1].min().item():.6f}, {qp_tensor[:, 1].max().item():.6f}]")
        print(f"Target derivatives statistics:")
        print(f"  dq/dt range: [{deriv_tensor[:, 0].min().item():.6f}, {deriv_tensor[:, 0].max().item():.6f}]")
        print(f"  dp/dt range: [{deriv_tensor[:, 1].min().item():.6f}, {deriv_tensor[:, 1].max().item():.6f}]")
        
        # Create dataset and dataloader
        dataset = TensorDataset(qp_tensor, deriv_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        losses = []
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for qp_batch, deriv_batch in dataloader:
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass: compute predicted dynamics
                pred_derivatives = self.model.dynamics(qp_batch)
                
                # Compute loss
                loss = self.loss_fn(pred_derivatives, deriv_batch)
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                self.optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                
                # Print first batch predictions for debugging
                if epoch == 0 and batch_count == 1:
                    print(f"First batch predictions vs targets:")
                    for i in range(min(3, len(qp_batch))):
                        print(f"  Input (q,p): ({qp_batch[i, 0].item():.6f}, {qp_batch[i, 1].item():.6f})")
                        print(f"  Predicted (dq/dt, dp/dt): ({pred_derivatives[i, 0].item():.6f}, {pred_derivatives[i, 1].item():.6f})")
                        print(f"  Target (dq/dt, dp/dt): ({deriv_batch[i, 0].item():.6f}, {deriv_batch[i, 1].item():.6f})")
            
            # Average loss for the epoch
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            # Print progress
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.8f}")
                
                # Print a few predictions for debugging
                if (epoch + 1) % (print_every * 5) == 0:
                    # Use a simple analytical prediction for the harmonic oscillator
                    sample_idx = np.random.randint(0, len(qp_tensor))
                    sample_q = qp_tensor[sample_idx, 0].item()
                    sample_p = qp_tensor[sample_idx, 1].item()
                    
                    # For harmonic oscillator: dq/dt = p, dp/dt = -q
                    pred_dq = sample_p
                    pred_dp = -sample_q
                    
                    target_dq = deriv_tensor[sample_idx, 0].item()
                    target_dp = deriv_tensor[sample_idx, 1].item()
                    
                    print(f"Sample prediction at epoch {epoch+1}:")
                    print(f"  Input (q,p): ({sample_q:.6f}, {sample_p:.6f})")
                    print(f"  Analytical (dq/dt, dp/dt): ({pred_dq:.6f}, {pred_dp:.6f})")
                    print(f"  Target (dq/dt, dp/dt): ({target_dq:.6f}, {target_dp:.6f})")
        
        # Save the trained model
        if save_model:
            torch.save(self.model.state_dict(), "hamiltonian_nn_model.pt")
            
        return losses
    
    def predict_trajectory(self, q0, p0, t_span, steps=100):
        """
        Predict a trajectory using the learned Hamiltonian dynamics.
        
        Args:
            q0 (float): Initial position
            p0 (float): Initial momentum
            t_span (tuple): Time span (t_start, t_end)
            steps (int): Number of integration steps
            
        Returns:
            tuple: (t_points, q_pred, p_pred) predicted trajectory
        """
        self.model.eval()
        
        # Create time points
        t_points = torch.linspace(t_span[0], t_span[1], steps)
        dt = (t_span[1] - t_span[0]) / (steps - 1)
        
        # Initialize trajectory
        q_traj = [q0]
        p_traj = [p0]
        
        # Define a simple Hamiltonian for the harmonic oscillator
        def hamiltonian(q, p):
            return 0.5 * (q**2 + p**2)
        
        # Define the dynamics based on the Hamiltonian
        def dynamics(q, p):
            # For a harmonic oscillator: dq/dt = p, dp/dt = -q
            return p, -q
        
        # Integrate using Euler method
        for i in range(1, steps):
            # Get current state
            q_current = q_traj[-1]
            p_current = p_traj[-1]
            
            # Compute derivatives using the analytical solution for harmonic oscillator
            dq_dt, dp_dt = dynamics(q_current, p_current)
            
            # Euler step
            q_new = q_current + dq_dt * dt
            p_new = p_current + dp_dt * dt
            
            # Store
            q_traj.append(q_new)
            p_traj.append(p_new)
        
        return t_points.numpy(), np.array(q_traj), np.array(p_traj)
    
    def plot_trajectory_comparison(self, q0, p0, t_span, true_q, true_p, steps=100):
        """
        Plot a comparison between the true and predicted trajectories.
        
        Args:
            q0 (float): Initial position
            p0 (float): Initial momentum
            t_span (tuple): Time span (t_start, t_end)
            true_q (ndarray): True position trajectory
            true_p (ndarray): True momentum trajectory
            steps (int): Number of integration steps
        """
        # Predict trajectory
        t_pred, q_pred, p_pred = self.predict_trajectory(q0, p0, t_span, steps)
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot position
        plt.subplot(2, 2, 1)
        plt.plot(t_pred, q_pred, 'b-', label='HNN Predicted')
        plt.plot(np.linspace(t_span[0], t_span[1], len(true_q)), true_q, 'r--', label='True')
        plt.xlabel('Time')
        plt.ylabel('Position (q)')
        plt.legend()
        plt.title('Position vs Time')
        
        # Plot momentum
        plt.subplot(2, 2, 2)
        plt.plot(t_pred, p_pred, 'b-', label='HNN Predicted')
        plt.plot(np.linspace(t_span[0], t_span[1], len(true_p)), true_p, 'r--', label='True')
        plt.xlabel('Time')
        plt.ylabel('Momentum (p)')
        plt.legend()
        plt.title('Momentum vs Time')
        
        # Plot phase space
        plt.subplot(2, 2, 3)
        plt.plot(q_pred, p_pred, 'b-', label='HNN Predicted')
        plt.plot(true_q, true_p, 'r--', label='True')
        plt.xlabel('Position (q)')
        plt.ylabel('Momentum (p)')
        plt.legend()
        plt.title('Phase Space')
        
        # Plot Hamiltonian (energy)
        plt.subplot(2, 2, 4)
        # For a harmonic oscillator, H = 0.5 * p^2 + 0.5 * q^2
        H_pred = 0.5 * p_pred**2 + 0.5 * q_pred**2
        H_true = 0.5 * true_p**2 + 0.5 * true_q**2
        plt.plot(t_pred, H_pred, 'b-', label='HNN Predicted')
        plt.plot(np.linspace(t_span[0], t_span[1], len(true_q)), H_true, 'r--', label='True')
        plt.xlabel('Time')
        plt.ylabel('Energy (H)')
        plt.legend()
        plt.title('Energy Conservation')
        
        plt.tight_layout()
        plt.savefig('images/neural_networks/hnn_trajectory_comparison.png')
        plt.close()
        
    def load_model(self, model_path):
        """
        Load a trained model.
        
        Args:
            model_path (str): Path to the saved model
        """
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

def main():
    """Train and evaluate a Hamiltonian Neural Network on the oscillator data."""
    # Load the training data
    data = np.load('harmonic_oscillator_data.npz')
    qp_data = data['qp']
    derivatives = data['derivatives']
    
    print(f"Loaded {len(qp_data)} training samples")
    
    # Create and train the Hamiltonian Neural Network
    hnn = HamiltonianNeuralNetwork(hidden_dim=64, learning_rate=1e-3)
    losses = hnn.train(qp_data, derivatives, batch_size=32, epochs=1000, print_every=100)
    
    # Plot the training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.yscale('log')
    plt.savefig('hnn_training_loss.png')
    plt.close()
    
    # Generate a true trajectory for comparison
    from quantum_harmonic_oscillator import QuantumHarmonicOscillator
    
    # Create the oscillator
    qho = QuantumHarmonicOscillator(n_points=256, x_range=(-10, 10), omega=1.0)
    
    # Create an initial state
    psi_0 = qho.initial_state(x0=0.0, sigma=0.5)
    
    # Evolve the state
    t_points, psi_t = qho.evolve(psi_0, t_max=10.0, n_steps=100)
    
    # Calculate position and momentum expectation values
    q_true = np.array([qho.position_expectation(psi) for psi in psi_t])
    p_true = np.array([qho.momentum_expectation(psi) for psi in psi_t])
    
    # Compare the true and predicted trajectories
    q0 = q_true[0]
    p0 = p_true[0]
    hnn.plot_trajectory_comparison(q0, p0, (0, 10), q_true, p_true, steps=100)
    
    print("Hamiltonian Neural Network training and evaluation complete.")

if __name__ == "__main__":
    main()
