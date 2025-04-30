"""
Measurement Module

This module implements quantum measurement operations and observable predictions
for the quantum harmonic oscillator system.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.special import hermite
from scipy.linalg import expm

class MeasurementOperator:
    """
    Class for performing measurements on quantum states.
    Implements basis transformations and observable calculations.
    """
    def __init__(self, n_points=128, x_range=(-5, 5)):
        """
        Initialize the measurement operator.
        
        Args:
            n_points (int): Number of spatial grid points
            x_range (tuple): Range of x values (min, max)
        """
        self.n_points = n_points
        self.x_range = x_range
        self.x = np.linspace(x_range[0], x_range[1], n_points)
        self.dx = (x_range[1] - x_range[0]) / (n_points - 1)
        
        # For momentum space
        self.k = 2 * np.pi * np.fft.fftfreq(n_points, self.dx)
        
        # Precompute basis states (harmonic oscillator eigenstates)
        self._compute_basis_states(max_n=10)
    
    def _compute_basis_states(self, max_n=10):
        """
        Compute the harmonic oscillator eigenstates.
        
        Args:
            max_n (int): Maximum quantum number
        """
        self.basis_states = []
        
        # Normalization constant
        norm_const = 1.0 / np.sqrt(np.pi)
        
        # Import math for factorial
        import math
        
        for n in range(max_n):
            # Hermite polynomial
            H_n = hermite(n)(self.x)
            
            # Harmonic oscillator eigenstate
            psi_n = norm_const * H_n * np.exp(-self.x**2 / 2) / np.sqrt(2**n * math.factorial(n))
            
            # Normalize
            psi_n = psi_n / np.sqrt(np.sum(np.abs(psi_n)**2) * self.dx)
            
            self.basis_states.append(psi_n)
    
    def measure_in_position_space(self, psi):
        """
        Measure the wavefunction in position space.
        
        Args:
            psi (ndarray): Wavefunction
            
        Returns:
            ndarray: Probability density in position space
        """
        return np.abs(psi)**2
    
    def measure_in_momentum_space(self, psi):
        """
        Measure the wavefunction in momentum space.
        
        Args:
            psi (ndarray): Wavefunction in position space
            
        Returns:
            ndarray: Probability density in momentum space
        """
        # FFT to transform to momentum space
        psi_k = np.fft.fft(psi) * self.dx / np.sqrt(2 * np.pi)
        
        # Probability density
        prob_density = np.abs(psi_k)**2
        
        return prob_density
    
    def measure_in_energy_basis(self, psi, max_n=10):
        """
        Measure the wavefunction in the energy eigenbasis.
        
        Args:
            psi (ndarray): Wavefunction in position space
            max_n (int): Maximum quantum number to consider
            
        Returns:
            ndarray: Probability distribution over energy eigenstates
        """
        # Compute overlaps with energy eigenstates
        overlaps = np.zeros(max_n, dtype=complex)
        
        for n in range(max_n):
            # Inner product with basis state
            overlaps[n] = np.sum(np.conj(self.basis_states[n]) * psi) * self.dx
        
        # Probability distribution
        prob_distribution = np.abs(overlaps)**2
        
        return prob_distribution
    
    def measure_observable(self, psi, observable_type='position'):
        """
        Measure the expectation value of an observable.
        
        Args:
            psi (ndarray): Wavefunction
            observable_type (str): Type of observable ('position', 'momentum', 'energy')
            
        Returns:
            float: Expectation value of the observable
        """
        if observable_type == 'position':
            # Position operator X
            prob = self.measure_in_position_space(psi)
            return np.sum(self.x * prob * self.dx)
        
        elif observable_type == 'momentum':
            # Momentum operator P
            prob_k = self.measure_in_momentum_space(psi)
            dk = self.k[1] - self.k[0]
            return np.sum(self.k * prob_k * dk)
        
        elif observable_type == 'energy':
            # Energy operator H = P^2/2 + X^2/2
            # Kinetic energy
            psi_k = np.fft.fft(psi) * self.dx / np.sqrt(2 * np.pi)
            T = 0.5 * np.sum(np.abs(psi_k)**2 * self.k**2) * (self.k[1] - self.k[0])
            
            # Potential energy
            V = 0.5 * np.sum(np.abs(psi)**2 * self.x**2) * self.dx
            
            return T + V
        
        elif observable_type == 'number':
            # Number operator N
            prob_n = self.measure_in_energy_basis(psi)
            return np.sum(np.arange(len(prob_n)) * prob_n)
        
        else:
            raise ValueError(f"Unknown observable type: {observable_type}")
    
    def pauli_x_expectation(self, psi):
        """
        Calculate the expectation value of the Pauli X operator.
        For the harmonic oscillator, this is related to the position operator.
        
        Args:
            psi (ndarray): Wavefunction
            
        Returns:
            float: Expectation value of Pauli X
        """
        # For simplicity, we map the position expectation to [-1, 1]
        x_expect = self.measure_observable(psi, 'position')
        x_max = np.max(np.abs(self.x))
        return x_expect / x_max
    
    def pauli_z_expectation(self, psi):
        """
        Calculate the expectation value of the Pauli Z operator.
        For the harmonic oscillator, this is related to the parity operator.
        
        Args:
            psi (ndarray): Wavefunction
            
        Returns:
            float: Expectation value of Pauli Z
        """
        # Parity operator: P|x⟩ = |-x⟩
        # Expectation value: ⟨ψ|P|ψ⟩ = ∫ ψ*(x)ψ(-x) dx
        
        # Interpolate ψ(-x)
        psi_minus_x = np.interp(-self.x, self.x, psi)
        
        # Calculate ⟨ψ|P|ψ⟩
        parity = np.sum(np.conj(psi) * psi_minus_x) * self.dx
        
        return np.real(parity)  # Should be real, but ensure it


class ObservablePredictor(nn.Module):
    """
    Neural network that predicts quantum observables from latent space variables.
    """
    def __init__(self, input_dim=2, hidden_dim=64):
        """
        Initialize the observable predictor.
        
        Args:
            input_dim (int): Dimension of input (latent space)
            hidden_dim (int): Dimension of hidden layers
        """
        super(ObservablePredictor, self).__init__()
        
        # Neural network for predicting observables
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4)  # Output: [⟨X⟩, ⟨Z⟩, ⟨E⟩, ⟨N⟩]
        )
    
    def forward(self, x):
        """
        Forward pass to predict observables.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Predicted observables
        """
        return self.net(x)


class QuantumMeasurementSystem:
    """
    Complete system for quantum measurements and observable predictions.
    """
    def __init__(self, n_points=128, x_range=(-5, 5), hidden_dim=64, learning_rate=1e-3):
        """
        Initialize the quantum measurement system.
        
        Args:
            n_points (int): Number of spatial grid points
            x_range (tuple): Range of x values (min, max)
            hidden_dim (int): Dimension of hidden layers
            learning_rate (float): Learning rate for optimization
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.measurement_op = MeasurementOperator(n_points=n_points, x_range=x_range)
        self.predictor = ObservablePredictor(input_dim=2, hidden_dim=hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
    
    def generate_training_data(self, wavefunctions):
        """
        Generate training data for the observable predictor.
        
        Args:
            wavefunctions (ndarray): Array of wavefunctions with shape (n_samples, n_points)
            
        Returns:
            tuple: (latent_vars, observables) for training
        """
        n_samples = len(wavefunctions)
        
        # Extract latent variables (position and momentum expectations)
        q = np.array([self.measurement_op.measure_observable(psi, 'position') 
                      for psi in wavefunctions])
        p = np.array([self.measurement_op.measure_observable(psi, 'momentum') 
                      for psi in wavefunctions])
        
        # Calculate observables
        x_expect = np.array([self.measurement_op.pauli_x_expectation(psi) 
                             for psi in wavefunctions])
        z_expect = np.array([self.measurement_op.pauli_z_expectation(psi) 
                             for psi in wavefunctions])
        e_expect = np.array([self.measurement_op.measure_observable(psi, 'energy') 
                             for psi in wavefunctions])
        n_expect = np.array([self.measurement_op.measure_observable(psi, 'number') 
                             for psi in wavefunctions])
        
        # Combine latent variables and observables
        latent_vars = np.stack([q, p], axis=1)
        observables = np.stack([x_expect, z_expect, e_expect, n_expect], axis=1)
        
        return latent_vars, observables
    
    def train(self, latent_vars, observables, batch_size=32, epochs=500, print_every=50):
        """
        Train the observable predictor.
        
        Args:
            latent_vars (ndarray): Latent variables (q, p) with shape (n_samples, 2)
            observables (ndarray): Target observables with shape (n_samples, 4)
            batch_size (int): Batch size for training
            epochs (int): Number of training epochs
            print_every (int): Print loss every print_every epochs
            
        Returns:
            list: Training losses
        """
        # Convert data to torch tensors
        latent_tensor = torch.tensor(latent_vars, dtype=torch.float32).to(self.device)
        obs_tensor = torch.tensor(observables, dtype=torch.float32).to(self.device)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(latent_tensor, obs_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        losses = []
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for latent_batch, obs_batch in dataloader:
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                pred_obs = self.predictor(latent_batch)
                
                # Compute loss
                loss = self.loss_fn(pred_obs, obs_batch)
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # Average loss for the epoch
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            # Print progress
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # Save the trained model
        torch.save(self.predictor.state_dict(), "observable_predictor_model.pt")
            
        return losses
    
    def predict_observables(self, q, p):
        """
        Predict observables from latent variables.
        
        Args:
            q (float or ndarray): Position(s)
            p (float or ndarray): Momentum(s)
            
        Returns:
            ndarray: Predicted observables [⟨X⟩, ⟨Z⟩, ⟨E⟩, ⟨N⟩]
        """
        self.predictor.eval()
        
        # Convert to tensor
        if isinstance(q, (int, float)) and isinstance(p, (int, float)):
            qp = torch.tensor([[q, p]], dtype=torch.float32).to(self.device)
        else:
            qp = torch.tensor(np.stack([q, p], axis=1), dtype=torch.float32).to(self.device)
        
        # Predict
        with torch.no_grad():
            pred_obs = self.predictor(qp)
        
        return pred_obs.cpu().numpy()
    
    def plot_observable_comparison(self, true_q, true_p, true_obs, pred_obs=None):
        """
        Plot a comparison between true and predicted observables.
        
        Args:
            true_q (ndarray): True position trajectory
            true_p (ndarray): True momentum trajectory
            true_obs (ndarray): True observables with shape (n_steps, 4)
            pred_obs (ndarray, optional): Predicted observables with shape (n_steps, 4)
        """
        if pred_obs is None:
            pred_obs = self.predict_observables(true_q, true_p)
        
        # Create time points
        t = np.linspace(0, 10, len(true_q))
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Observable names and labels
        obs_names = ['⟨X⟩', '⟨Z⟩', '⟨E⟩', '⟨N⟩']
        
        # Plot each observable
        for i, name in enumerate(obs_names):
            plt.subplot(2, 2, i+1)
            plt.plot(t, true_obs[:, i], 'r--', label='True')
            plt.plot(t, pred_obs[:, i], 'b-', label='Predicted')
            plt.xlabel('Time')
            plt.ylabel(name)
            plt.legend()
            plt.title(f'{name} vs Time')
        
        plt.tight_layout()
        plt.savefig('observable_comparison.png')
        plt.close()
    
    def load_model(self, model_path):
        """
        Load a trained model.
        
        Args:
            model_path (str): Path to the saved model
        """
        self.predictor.load_state_dict(torch.load(model_path, map_location=self.device))
        self.predictor.eval()


def main():
    """Test the quantum measurement system."""
    # Load the quantum harmonic oscillator module
    from quantum_harmonic_oscillator import QuantumHarmonicOscillator
    
    # Create the oscillator
    qho = QuantumHarmonicOscillator(n_points=256, x_range=(-10, 10), omega=1.0)
    
    # Generate different initial states and evolve them
    n_states = 5
    all_wavefunctions = []
    
    for i in range(n_states):
        # Vary the initial state parameters
        x0 = np.random.uniform(-2, 2)
        sigma = np.random.uniform(0.3, 1.0)
        
        # Create and evolve the initial state
        psi_0 = qho.initial_state(x0=x0, sigma=sigma)
        t_points, psi_t = qho.evolve(psi_0, t_max=10.0, n_steps=50)
        
        # Add to collection
        all_wavefunctions.append(psi_t)
    
    # Flatten the collection of wavefunctions
    all_wavefunctions = np.vstack(all_wavefunctions)
    print(f"Generated {len(all_wavefunctions)} wavefunctions for training")
    
    # Create and train the quantum measurement system
    qms = QuantumMeasurementSystem(n_points=256, x_range=(-10, 10))
    
    # Generate training data
    latent_vars, observables = qms.generate_training_data(all_wavefunctions)
    print(f"Generated training data: {latent_vars.shape}, {observables.shape}")
    
    # Train the observable predictor
    losses = qms.train(latent_vars, observables, batch_size=32, epochs=500, print_every=50)
    
    # Plot the training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.yscale('log')
    plt.savefig('observable_predictor_training_loss.png')
    plt.close()
    
    # Test on a new trajectory
    psi_0 = qho.initial_state(x0=1.0, sigma=0.5)
    t_points, psi_t = qho.evolve(psi_0, t_max=10.0, n_steps=100)
    
    # Calculate true observables
    true_q = np.array([qho.position_expectation(psi) for psi in psi_t])
    true_p = np.array([qho.momentum_expectation(psi) for psi in psi_t])
    
    # Calculate true observables using the measurement operator
    measurement_op = MeasurementOperator(n_points=256, x_range=(-10, 10))
    true_x_expect = np.array([measurement_op.pauli_x_expectation(psi) for psi in psi_t])
    true_z_expect = np.array([measurement_op.pauli_z_expectation(psi) for psi in psi_t])
    true_e_expect = np.array([measurement_op.measure_observable(psi, 'energy') for psi in psi_t])
    true_n_expect = np.array([measurement_op.measure_observable(psi, 'number') for psi in psi_t])
    
    true_obs = np.stack([true_x_expect, true_z_expect, true_e_expect, true_n_expect], axis=1)
    
    # Predict observables
    pred_obs = qms.predict_observables(true_q, true_p)
    
    # Plot comparison
    qms.plot_observable_comparison(true_q, true_p, true_obs, pred_obs)
    
    print("Quantum measurement system training and evaluation complete.")

if __name__ == "__main__":
    main()
