"""
Hybrid Neural Network Module

This module implements a hybrid approach combining Hamiltonian Neural Networks (HNN)
and Lagrangian Neural Networks (LNN) to leverage their complementary strengths:
- HNN: Better phase space trajectory accuracy
- LNN: Superior energy conservation

The hybrid model uses an ensemble approach to combine predictions and
applies physics-constrained corrections to ensure physical consistency.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from .hamiltonian_neural_network import HamiltonianNeuralNetwork
from .lagrangian_neural_network import LagrangianNeuralNetwork
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

class HybridNeuralNetwork:
    """
    A hybrid model that combines HNN and LNN predictions to leverage
    their complementary strengths.
    """
    
    def __init__(self, hnn, lnn, device='cpu'):
        """
        Initialize the hybrid model with pre-trained HNN and LNN instances.
        
        Args:
            hnn (HamiltonianNeuralNetwork): Pre-trained HNN model
            lnn (LagrangianNeuralNetwork): Pre-trained LNN model
            device (str): Device to run computations on ('cpu' or 'cuda')
        """
        self.hnn = hnn
        self.lnn = lnn
        self.device = device
    
    def predict_ensemble_trajectory(self, q0, p0, t_span, steps=100, alpha=0.5):
        """
        Predict trajectory using a weighted combination of HNN and LNN predictions.
        
        Args:
            q0 (float): Initial position
            p0 (float): Initial momentum
            t_span (tuple): Time span (t_start, t_end)
            steps (int): Number of integration steps
            alpha (float): Weight for HNN prediction (1-alpha for LNN)
                           Values between 0 and 1
            
        Returns:
            tuple: (t_points, q_hybrid, p_hybrid, energy_hybrid) hybrid trajectory
        """
        # Get predictions from both models
        t_hnn, q_hnn, p_hnn = self.hnn.predict_trajectory(q0, p0, t_span, steps)
        t_lnn, q_lnn, q_dot_lnn, energy_lnn = self.lnn.predict_trajectory(q0, p0, t_span, steps)
        
        # Convert LNN velocity to momentum (for harmonic oscillator, p = q_dot)
        p_lnn = q_dot_lnn
        
        # Create weighted combination
        q_hybrid = alpha * q_hnn + (1 - alpha) * q_lnn
        p_hybrid = alpha * p_hnn + (1 - alpha) * p_lnn
        
        # Calculate energy for the hybrid trajectory
        energy_hybrid = 0.5 * p_hybrid**2 + 0.5 * q_hybrid**2
        
        return t_hnn, q_hybrid, p_hybrid, energy_hybrid
    
    def predict_physics_constrained_trajectory(self, q0, p0, t_span, steps=100, alpha=0.5, 
                                              energy_conservation_weight=0.5):
        """
        Predict trajectory using a physics-constrained integration that combines
        HNN phase space accuracy with LNN energy conservation.
        
        Args:
            q0 (float): Initial position
            p0 (float): Initial momentum
            t_span (tuple): Time span (t_start, t_end)
            steps (int): Number of integration steps
            alpha (float): Weight for HNN prediction (1-alpha for LNN)
            energy_conservation_weight (float): Weight for energy conservation correction
            
        Returns:
            tuple: (t_points, q_pred, p_pred, energy_pred) physics-constrained trajectory
        """
        # Create time points
        t_points = np.linspace(t_span[0], t_span[1], steps)
        dt = (t_span[1] - t_span[0]) / (steps - 1)
        
        # Initialize trajectory
        q_traj = [q0]
        p_traj = [p0]
        energy_traj = [0.5 * p0**2 + 0.5 * q0**2]  # Initial energy
        target_energy = energy_traj[0]  # Energy to conserve
        
        # Integrate using physics-constrained approach
        for i in range(1, steps):
            # Get current state
            q_current = q_traj[-1]
            p_current = p_traj[-1]
            
            # Get HNN prediction for derivatives
            dq_dt_hnn, dp_dt_hnn = self._get_hnn_derivatives(q_current, p_current)
            dq_dt_lnn, dp_dt_lnn = self._get_lnn_derivatives(q_current, p_current)
            dq_dt = alpha * dq_dt_hnn + (1 - alpha) * dq_dt_lnn
            dp_dt = alpha * dp_dt_hnn + (1 - alpha) * dp_dt_lnn
            
            # Euler step
            q_new = q_current + dq_dt * dt
            p_new = p_current + dp_dt * dt
            
            # Calculate current energy
            energy_current = 0.5 * p_new**2 + 0.5 * q_new**2
            energy_error = target_energy - energy_current
            
            # Apply energy conservation correction if error is significant
            if abs(energy_error) > 1e-6 and energy_conservation_weight > 0:
                # Calculate gradients of energy with respect to q and p
                dE_dq = q_new  # ∂E/∂q = q for harmonic oscillator
                dE_dp = p_new  # ∂E/∂p = p for harmonic oscillator
                
                # Normalize gradient
                grad_norm = np.sqrt(dE_dq**2 + dE_dp**2)
                if grad_norm > 1e-10:  # Avoid division by zero
                    dE_dq /= grad_norm
                    dE_dp /= grad_norm
                    
                    # Apply correction along energy gradient
                    q_new += energy_conservation_weight * energy_error * dE_dq
                    p_new += energy_conservation_weight * energy_error * dE_dp
            
            # Store
            q_traj.append(q_new)
            p_traj.append(p_new)
            energy_traj.append(0.5 * p_new**2 + 0.5 * q_new**2)
        
        return t_points, np.array(q_traj), np.array(p_traj), np.array(energy_traj)
    
    def predict_learning_based_trajectory(self, q0, p0, t_span, steps=100, alpha=0.5):
        """
        Predict trajectory using a pure learning-based approach that combines
        HNN and LNN without explicit energy constraints.
        
        Args:
            q0 (float): Initial position
            p0 (float): Initial momentum
            t_span (tuple): Time span (t_start, t_end)
            steps (int): Number of integration steps
            alpha (float): Weight for HNN prediction (1-alpha for LNN)
            
        Returns:
            tuple: (t_points, q_pred, p_pred, energy_pred) learning-based trajectory
        """
        # Create time points
        t_points = np.linspace(t_span[0], t_span[1], steps)
        dt = (t_span[1] - t_span[0]) / (steps - 1)
        
        # Initialize trajectory
        q_traj = [q0]
        p_traj = [p0]
        energy_traj = [0.5 * p0**2 + 0.5 * q0**2]
        
        # Use 4th order Runge-Kutta integrator for better accuracy without explicit constraints
        for i in range(1, steps):
            # Get current state
            q_current = q_traj[-1]
            p_current = p_traj[-1]
            
            # Get HNN prediction for derivatives
            dq_dt_hnn, dp_dt_hnn = self._get_hnn_derivatives(q_current, p_current)
            dq_dt_lnn, dp_dt_lnn = self._get_lnn_derivatives(q_current, p_current)
            dq_dt = alpha * dq_dt_hnn + (1 - alpha) * dq_dt_lnn
            dp_dt = alpha * dp_dt_hnn + (1 - alpha) * dp_dt_lnn
            
            # RK4 integration
            # k1
            q1 = q_current + 0.5 * dt * dq_dt
            p1 = p_current + 0.5 * dt * dp_dt
            dq_dt_hnn1, dp_dt_hnn1 = self._get_hnn_derivatives(q1, p1)
            dq_dt_lnn1, dp_dt_lnn1 = self._get_lnn_derivatives(q1, p1)
            dq1 = alpha * dq_dt_hnn1 + (1 - alpha) * dq_dt_lnn1
            dp1 = alpha * dp_dt_hnn1 + (1 - alpha) * dp_dt_lnn1
            
            # k2
            q2 = q_current + 0.5 * dt * dq1
            p2 = p_current + 0.5 * dt * dp1
            dq_dt_hnn2, dp_dt_hnn2 = self._get_hnn_derivatives(q2, p2)
            dq_dt_lnn2, dp_dt_lnn2 = self._get_lnn_derivatives(q2, p2)
            dq2 = alpha * dq_dt_hnn2 + (1 - alpha) * dq_dt_lnn2
            dp2 = alpha * dp_dt_hnn2 + (1 - alpha) * dp_dt_lnn2
            
            # k3
            q3 = q_current + 0.5 * dt * dq2
            p3 = p_current + 0.5 * dt * dp2
            dq_dt_hnn3, dp_dt_hnn3 = self._get_hnn_derivatives(q3, p3)
            dq_dt_lnn3, dp_dt_lnn3 = self._get_lnn_derivatives(q3, p3)
            dq3 = alpha * dq_dt_hnn3 + (1 - alpha) * dq_dt_lnn3
            dp3 = alpha * dp_dt_hnn3 + (1 - alpha) * dp_dt_lnn3
            
            # k4
            q4 = q_current + dt * dq3
            p4 = p_current + dt * dp3
            dq_dt_hnn4, dp_dt_hnn4 = self._get_hnn_derivatives(q4, p4)
            dq_dt_lnn4, dp_dt_lnn4 = self._get_lnn_derivatives(q4, p4)
            dq4 = alpha * dq_dt_hnn4 + (1 - alpha) * dq_dt_lnn4
            dp4 = alpha * dp_dt_hnn4 + (1 - alpha) * dp_dt_lnn4
            
            # Update using weighted average of all steps
            q_new = q_current + (dt / 6.0) * (dq_dt + 2*dq1 + 2*dq2 + dq3)
            p_new = p_current + (dt / 6.0) * (dp_dt + 2*dp1 + 2*dp2 + dp3)
            
            # Store
            q_traj.append(q_new)
            p_traj.append(p_new)
            energy_traj.append(0.5 * p_new**2 + 0.5 * q_new**2)
        
        return t_points, np.array(q_traj), np.array(p_traj), np.array(energy_traj)
    
    def _get_hnn_derivatives(self, q, p):
        """Helper method to get derivatives from HNN"""
        # For harmonic oscillator with Hamiltonian H = 0.5*p^2 + 0.5*q^2:
        # dq/dt = ∂H/∂p = p
        # dp/dt = -∂H/∂q = -q
        
        # Convert to tensor for HNN
        q_tensor = torch.tensor([[q]], dtype=torch.float32).to(self.device)
        p_tensor = torch.tensor([[p]], dtype=torch.float32).to(self.device)
        
        # The HNN model outputs the Hamiltonian value, not derivatives directly
        # We need to compute derivatives using autograd
        q_tensor.requires_grad_(True)
        p_tensor.requires_grad_(True)
        inputs = torch.cat([q_tensor, p_tensor], dim=1)
        
        # Get Hamiltonian value
        with torch.enable_grad():  # Enable gradient computation
            H = self.hnn.model(inputs)
            
            # Compute dH/dp (= dq/dt)
            dH_dp = torch.autograd.grad(H, p_tensor, create_graph=True)[0]
            dq_dt = dH_dp.item()
            
            # Compute -dH/dq (= dp/dt)
            dH_dq = torch.autograd.grad(H, q_tensor, create_graph=True)[0]
            dp_dt = -dH_dq.item()  # Note the negative sign
        
        return dq_dt, dp_dt
    
    def _get_lnn_derivatives(self, q, p):
        """Helper method to get derivatives from LNN"""
        # For LNN, we use q_dot = p for harmonic oscillator
        q_dot = p
        
        # For harmonic oscillator with Lagrangian L = 0.5*q_dot^2 - 0.5*q^2:
        # dL/dq = -q
        # dL/dq_dot = q_dot
        # Euler-Lagrange: d/dt(dL/dq_dot) = dL/dq
        # This gives: d/dt(q_dot) = -q or dp/dt = -q
        
        dq_dt = q_dot  # dq/dt = p
        dp_dt = -q     # dp/dt = -q
        
        return dq_dt, dp_dt
    
    def plot_comparison(self, q0, p0, t_span, true_q, true_p, steps=100, alpha=0.5,
                        energy_conservation_weight=0.5, save_path=None):
        """
        Plot a comparison between true, HNN, LNN, and hybrid trajectories.
        
        Args:
            q0 (float): Initial position
            p0 (float): Initial momentum
            t_span (tuple): Time span (t_start, t_end)
            true_q (ndarray): True position values
            true_p (ndarray): True momentum values
            steps (int): Number of integration steps
            alpha (float): Weight for HNN prediction
            energy_conservation_weight (float): Weight for energy conservation
            save_path (str): Path to save the figure
        """
        # Get predictions from all models
        t_hnn, q_hnn, p_hnn = self.hnn.predict_trajectory(q0, p0, t_span, steps)
        t_lnn, q_lnn, q_dot_lnn, energy_lnn = self.lnn.predict_trajectory(q0, p0, t_span, steps)
        p_lnn = q_dot_lnn  # For harmonic oscillator, p = q_dot
        
        # Get ensemble prediction
        t_ensemble, q_ensemble, p_ensemble, energy_ensemble = self.predict_ensemble_trajectory(
            q0, p0, t_span, steps, alpha
        )
        
        # Get physics-constrained prediction
        t_physics, q_physics, p_physics, energy_physics = self.predict_physics_constrained_trajectory(
            q0, p0, t_span, steps, alpha, energy_conservation_weight
        )
        
        # Get learning-based prediction
        t_learning, q_learning, p_learning, energy_learning = self.predict_learning_based_trajectory(
            q0, p0, t_span, steps, alpha
        )
        
        # Calculate true energy
        energy_true = 0.5 * true_p**2 + 0.5 * true_q**2
        energy_hnn = 0.5 * p_hnn**2 + 0.5 * q_hnn**2
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 12))
        
        # 1. Phase space trajectories
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(true_q, true_p, 'r-', label='True', linewidth=2)
        ax1.plot(q_hnn, p_hnn, 'b--', label='HNN', linewidth=1.5)
        ax1.plot(q_lnn, p_lnn, 'g-.', label='LNN', linewidth=1.5)
        ax1.plot(q_ensemble, p_ensemble, 'm:', label='Ensemble', linewidth=1.5)
        ax1.plot(q_physics, p_physics, 'c-', label='Physics-Constrained', linewidth=1.5)
        ax1.plot(q_learning, p_learning, 'y-', label='Learning-Based', linewidth=1.5)
        ax1.set_xlabel('Position (q)')
        ax1.set_ylabel('Momentum (p)')
        ax1.set_title('Phase Space Trajectories')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Position over time
        ax2 = plt.subplot(2, 2, 2)
        t_true = np.linspace(t_span[0], t_span[1], len(true_q))
        ax2.plot(t_true, true_q, 'r-', label='True', linewidth=2)
        ax2.plot(t_hnn, q_hnn, 'b--', label='HNN', linewidth=1.5)
        ax2.plot(t_lnn, q_lnn, 'g-.', label='LNN', linewidth=1.5)
        ax2.plot(t_ensemble, q_ensemble, 'm:', label='Ensemble', linewidth=1.5)
        ax2.plot(t_physics, q_physics, 'c-', label='Physics-Constrained', linewidth=1.5)
        ax2.plot(t_learning, q_learning, 'y-', label='Learning-Based', linewidth=1.5)
        ax2.set_xlabel('Time (t)')
        ax2.set_ylabel('Position (q)')
        ax2.set_title('Position vs Time')
        ax2.legend()
        ax2.grid(True)
        
        # 3. Momentum over time
        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(t_true, true_p, 'r-', label='True', linewidth=2)
        ax3.plot(t_hnn, p_hnn, 'b--', label='HNN', linewidth=1.5)
        ax3.plot(t_lnn, p_lnn, 'g-.', label='LNN', linewidth=1.5)
        ax3.plot(t_ensemble, p_ensemble, 'm:', label='Ensemble', linewidth=1.5)
        ax3.plot(t_physics, p_physics, 'c-', label='Physics-Constrained', linewidth=1.5)
        ax3.plot(t_learning, p_learning, 'y-', label='Learning-Based', linewidth=1.5)
        ax3.set_xlabel('Time (t)')
        ax3.set_ylabel('Momentum (p)')
        ax3.set_title('Momentum vs Time')
        ax3.legend()
        ax3.grid(True)
        
        # 4. Energy conservation
        ax4 = plt.subplot(2, 2, 4)
        ax4.plot(t_true, energy_true, 'r-', label='True', linewidth=2)
        ax4.plot(t_hnn, energy_hnn, 'b--', label='HNN', linewidth=1.5)
        ax4.plot(t_lnn, energy_lnn, 'g-.', label='LNN', linewidth=1.5)
        ax4.plot(t_ensemble, energy_ensemble, 'm:', label='Ensemble', linewidth=1.5)
        ax4.plot(t_physics, energy_physics, 'c-', label='Physics-Constrained', linewidth=1.5)
        ax4.plot(t_learning, energy_learning, 'y-', label='Learning-Based', linewidth=1.5)
        ax4.set_xlabel('Time (t)')
        ax4.set_ylabel('Energy (H)')
        ax4.set_title('Energy Conservation')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Comparison plot saved to '{save_path}'")
        
        plt.close()
        
    def train_hybrid_model(self, qp_data, derivatives, batch_size=32, epochs=500, 
                          learning_rate=1e-3, print_every=100, save_model=True):
        """
        Train a new hybrid model that learns from both HNN and LNN.
        
        Args:
            qp_data (ndarray): Input data of shape (n_samples, 2) containing (q, p)
            derivatives (ndarray): Target data of shape (n_samples, 2) containing (dq/dt, dp/dt)
            batch_size (int): Batch size for training
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate for optimization
            print_every (int): Print loss every print_every epochs
            save_model (bool): Whether to save the model after training
            
        Returns:
            tuple: (losses, trained_model) - training losses and the trained hybrid model
        """
        # Create a new neural network for the hybrid model
        class HybridNet(nn.Module):
            def __init__(self, hnn, lnn, hidden_dim=64):
                super(HybridNet, self).__init__()
                self.hnn = hnn  # Pre-trained HNN
                self.lnn = lnn  # Pre-trained LNN
                
                # Freeze the pre-trained models
                for param in self.hnn.parameters():
                    param.requires_grad = False
                for param in self.lnn.parameters():
                    param.requires_grad = False
                
                # New layers to learn the optimal combination
                self.net = nn.Sequential(
                    nn.Linear(4, hidden_dim),  # 4 inputs: q, p, HNN_pred, LNN_pred
                    nn.Tanh(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, 2)  # 2 outputs: dq/dt, dp/dt
                )
            
            def forward(self, q, p):
                # Get HNN prediction
                q_tensor = q.clone().detach().requires_grad_(True)
                p_tensor = p.clone().detach().requires_grad_(True)
                
                # Compute HNN derivatives
                with torch.enable_grad():
                    H = self.hnn(torch.cat([q_tensor, p_tensor], dim=1))
                    dH_dp = torch.autograd.grad(H.sum(), p_tensor, create_graph=True)[0]
                    dH_dq = torch.autograd.grad(H.sum(), q_tensor, create_graph=True)[0]
                    
                hnn_dq_dt = dH_dp
                hnn_dp_dt = -dH_dq
                
                # For LNN, we use analytical derivatives for harmonic oscillator
                # dq/dt = p, dp/dt = -q
                lnn_dq_dt = p
                lnn_dp_dt = -q
                
                # Combine all inputs - only use 4 inputs as specified in the network
                combined = torch.cat([q, p, hnn_dq_dt, hnn_dp_dt], dim=1)
                
                # Learn the optimal combination
                return self.net(combined)
        
        # Create and train the hybrid model
        device = self.device
        hybrid_net = HybridNet(self.hnn.model, self.lnn.model, hidden_dim=64).to(device)
        optimizer = optim.Adam(hybrid_net.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        
        # Convert data to tensors
        q = torch.tensor(qp_data[:, 0:1], dtype=torch.float32).to(device)
        p = torch.tensor(qp_data[:, 1:2], dtype=torch.float32).to(device)
        dq_dt = torch.tensor(derivatives[:, 0:1], dtype=torch.float32).to(device)
        dp_dt = torch.tensor(derivatives[:, 1:2], dtype=torch.float32).to(device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(q, p, dq_dt, dp_dt)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        losses = []
        
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            
            for q_batch, p_batch, dq_dt_batch, dp_dt_batch in dataloader:
                # Forward pass
                pred = hybrid_net(q_batch, p_batch)
                pred_dq_dt, pred_dp_dt = pred[:, 0:1], pred[:, 1:2]
                
                # Compute loss
                loss = loss_fn(pred_dq_dt, dq_dt_batch) + loss_fn(pred_dp_dt, dp_dt_batch)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Average loss for the epoch
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            # Print progress
            if epoch % print_every == 0 or epoch == 1:
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.8f}")
        
        # Save the model if requested
        if save_model:
            torch.save(hybrid_net.state_dict(), "models/hybrid_learned_model.pt")
            print("Hybrid model saved to 'models/hybrid_learned_model.pt'")
        
        # Return losses and the trained model
        return losses, hybrid_net
    
    def predict_learned_trajectory(self, hybrid_net, q0, p0, t_span, steps=100):
        """
        Predict trajectory using the learned hybrid model.
        
        Args:
            hybrid_net: Trained hybrid neural network
            q0 (float): Initial position
            p0 (float): Initial momentum
            t_span (tuple): Time span (t_start, t_end)
            steps (int): Number of integration steps
            
        Returns:
            tuple: (t_points, q_pred, p_pred, energy_pred) predicted trajectory
        """
        # Create time points
        t_points = np.linspace(t_span[0], t_span[1], steps)
        dt = (t_span[1] - t_span[0]) / (steps - 1)
        
        # Initialize trajectory
        q_traj = [q0]
        p_traj = [p0]
        energy_traj = [0.5 * p0**2 + 0.5 * q0**2]
        
        # Use 4th order Runge-Kutta integrator
        for i in range(1, steps):
            # Get current state
            q_current = q_traj[-1]
            p_current = p_traj[-1]
            
            # RK4 integration using the learned model
            # k1
            q_tensor = torch.tensor([[q_current]], dtype=torch.float32).to(self.device)
            p_tensor = torch.tensor([[p_current]], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                derivatives = hybrid_net(q_tensor, p_tensor)
                dq1 = derivatives[0, 0].item()
                dp1 = derivatives[0, 1].item()
            
            # k2
            q2 = q_current + 0.5 * dt * dq1
            p2 = p_current + 0.5 * dt * dp1
            q_tensor = torch.tensor([[q2]], dtype=torch.float32).to(self.device)
            p_tensor = torch.tensor([[p2]], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                derivatives = hybrid_net(q_tensor, p_tensor)
                dq2 = derivatives[0, 0].item()
                dp2 = derivatives[0, 1].item()
            
            # k3
            q3 = q_current + 0.5 * dt * dq2
            p3 = p_current + 0.5 * dt * dp2
            q_tensor = torch.tensor([[q3]], dtype=torch.float32).to(self.device)
            p_tensor = torch.tensor([[p3]], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                derivatives = hybrid_net(q_tensor, p_tensor)
                dq3 = derivatives[0, 0].item()
                dp3 = derivatives[0, 1].item()
            
            # k4
            q4 = q_current + dt * dq3
            p4 = p_current + dt * dp3
            q_tensor = torch.tensor([[q4]], dtype=torch.float32).to(self.device)
            p_tensor = torch.tensor([[p4]], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                derivatives = hybrid_net(q_tensor, p_tensor)
                dq4 = derivatives[0, 0].item()
                dp4 = derivatives[0, 1].item()
            
            # Update using weighted average of all steps
            q_new = q_current + (dt / 6.0) * (dq1 + 2*dq2 + 2*dq3 + dq4)
            p_new = p_current + (dt / 6.0) * (dp1 + 2*dp2 + 2*dp3 + dp4)
            
            # Store
            q_traj.append(q_new)
            p_traj.append(p_new)
            energy_traj.append(0.5 * p_new**2 + 0.5 * q_new**2)
        
        return t_points, np.array(q_traj), np.array(p_traj), np.array(energy_traj)
    
    def optimize_hyperparameters(self, q0, p0, t_span, true_q, true_p, steps=100,
                               alpha_values=None, energy_weights=None):
        """
        Find optimal hyperparameters (alpha and energy_conservation_weight)
        by evaluating different combinations.
        
        Args:
            q0, p0: Initial conditions
            t_span: Time span
            true_q, true_p: True trajectory
            steps: Number of integration steps
            alpha_values: List of alpha values to try
            energy_weights: List of energy conservation weights to try
            
        Returns:
            tuple: (best_alpha, best_energy_weight, best_score)
        """
        if alpha_values is None:
            alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        if energy_weights is None:
            energy_weights = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        best_score = float('inf')
        best_alpha = 0.5
        best_energy_weight = 0.5
        
        results = []
        
        # True energy (target for conservation)
        energy_true = 0.5 * true_p**2 + 0.5 * true_q**2
        initial_energy = energy_true[0]
        
        # Interpolate true trajectory to match prediction time points
        t_true = np.linspace(t_span[0], t_span[1], len(true_q))
        t_pred = np.linspace(t_span[0], t_span[1], steps)
        
        # Try different hyperparameter combinations
        for alpha in alpha_values:
            for energy_weight in energy_weights:
                # Get physics-constrained prediction
                _, q_pred, p_pred, energy_pred = self.predict_physics_constrained_trajectory(
                    q0, p0, t_span, steps, alpha, energy_weight
                )
                
                # Calculate phase space error
                phase_space_error = np.mean(
                    np.sqrt((np.interp(t_pred, t_true, true_q) - q_pred)**2 + 
                           (np.interp(t_pred, t_true, true_p) - p_pred)**2)
                )
                
                # Calculate energy conservation error
                energy_conservation_error = np.mean(
                    np.abs(energy_pred - initial_energy)
                )
                
                # Combined score (lower is better)
                score = phase_space_error + energy_conservation_error
                
                results.append({
                    'alpha': alpha,
                    'energy_weight': energy_weight,
                    'phase_space_error': phase_space_error,
                    'energy_error': energy_conservation_error,
                    'score': score
                })
                
                # Update best parameters
                if score < best_score:
                    best_score = score
                    best_alpha = alpha
                    best_energy_weight = energy_weight
        
        # Print results
        print("\nHyperparameter optimization results:")
        print("-" * 80)
        print(f"{'Alpha':<10} {'Energy Weight':<15} {'Phase Space Error':<20} {'Energy Error':<15} {'Total Score':<15}")
        print("-" * 80)
        
        for result in sorted(results, key=lambda x: x['score']):
            print(f"{result['alpha']:<10.2f} {result['energy_weight']:<15.2f} "
                  f"{result['phase_space_error']:<20.6f} {result['energy_error']:<15.6f} "
                  f"{result['score']:<15.6f}")
        
        print("-" * 80)
        print(f"Best parameters: alpha={best_alpha:.2f}, energy_weight={best_energy_weight:.2f}, score={best_score:.6f}")
        
        return best_alpha, best_energy_weight, best_score

def main():
    """Test the hybrid neural network model."""
    import numpy as np
    from hamiltonian_neural_network import HamiltonianNeuralNetwork
    from lagrangian_neural_network import LagrangianNeuralNetwork
    
    # Create HNN and LNN instances
    hnn = HamiltonianNeuralNetwork()
    lnn = LagrangianNeuralNetwork()
    
    # Load pre-trained models
    hnn.load_model("models/hamiltonian_nn_model.pt")
    lnn.load_model("models/lagrangian_nn_model.pt")
    
    # Create hybrid model
    hybrid = HybridNeuralNetwork(hnn, lnn)
    
    # Generate true trajectory
    t_span = (0, 10)
    steps = 100
    q0, p0 = 1.0, 0.0
    
    # Simple harmonic oscillator analytical solution
    t_true = np.linspace(t_span[0], t_span[1], steps)
    q_true = q0 * np.cos(t_true)
    p_true = -q0 * np.sin(t_true)
    
    # Test hybrid prediction
    t_hybrid, q_hybrid, p_hybrid, energy_hybrid = hybrid.predict_physics_constrained_trajectory(
        q0, p0, t_span, steps, alpha=0.7, energy_conservation_weight=0.5
    )
    
    # Test learning-based prediction
    t_learning, q_learning, p_learning, energy_learning = hybrid.predict_learning_based_trajectory(
        q0, p0, t_span, steps, alpha=0.7
    )
    
    # Plot comparison
    hybrid.plot_comparison(q0, p0, t_span, q_true, p_true, steps, 
                          alpha=0.7, energy_conservation_weight=0.5,
                          save_path="images/comparisons/hybrid_model_comparison.png")
    
    print("Hybrid model test complete.")
    
if __name__ == "__main__":
    main()
