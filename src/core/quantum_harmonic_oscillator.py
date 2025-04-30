"""
Quantum Harmonic Oscillator Simulation Module

This module implements a 1D quantum harmonic oscillator simulation
using numerical integration with NumPy and SciPy.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class QuantumHarmonicOscillator:
    def __init__(self, n_points=128, x_range=(-5, 5), omega=1.0):
        """
        Initialize the quantum harmonic oscillator simulation.
        
        Args:
            n_points (int): Number of spatial grid points
            x_range (tuple): Range of x values (min, max)
            omega (float): Angular frequency of the oscillator
        """
        self.n_points = n_points
        self.x_range = x_range
        self.omega = omega
        self.x = np.linspace(x_range[0], x_range[1], n_points)
        self.dx = (x_range[1] - x_range[0]) / (n_points - 1)
        
        # For momentum space
        self.k = 2 * np.pi * np.fft.fftfreq(n_points, self.dx)
    
    def _hamiltonian_operator(self, t, psi):
        """
        Hamiltonian operator for the quantum harmonic oscillator.
        H = p^2/2m + mω^2x^2/2 (with m=1)
        
        Args:
            t (float): Time (not used, but required by solve_ivp)
            psi (ndarray): Wavefunction at time t
            
        Returns:
            ndarray: Time derivative of psi according to Schrödinger equation
        """
        # Kinetic energy term (second derivative)
        kinetic = -0.5 * np.gradient(np.gradient(psi, self.dx), self.dx)
        
        # Potential energy term
        potential = 0.5 * self.omega**2 * self.x**2 * psi
        
        # Full Hamiltonian
        H_psi = kinetic + potential
        
        # Schrödinger equation: i*ħ*∂ψ/∂t = Hψ (with ħ=1)
        return -1j * H_psi
    
    def initial_state(self, x0=0, sigma=0.5):
        """
        Create an initial Gaussian wavepacket.
        
        Args:
            x0 (float): Center of the wavepacket
            sigma (float): Width of the wavepacket
            
        Returns:
            ndarray: Normalized initial wavefunction
        """
        psi_0 = np.exp(-(self.x - x0)**2 / (2 * sigma**2))
        # Normalize
        psi_0 = psi_0 / np.sqrt(np.sum(np.abs(psi_0)**2) * self.dx)
        return psi_0
    
    def evolve(self, psi_0, t_max=10.0, n_steps=100):
        """
        Evolve the wavefunction using the Schrödinger equation.
        
        Args:
            psi_0 (ndarray): Initial wavefunction
            t_max (float): Maximum evolution time
            n_steps (int): Number of time steps
            
        Returns:
            tuple: (t_points, psi_t) where psi_t has shape (n_steps, n_points)
        """
        t_points = np.linspace(0, t_max, n_steps)
        
        # Solve the Schrödinger equation
        sol = solve_ivp(
            self._hamiltonian_operator,
            [0, t_max],
            psi_0,
            t_eval=t_points,
            method='RK45',
            rtol=1e-6,
            atol=1e-8
        )
        
        # Transpose to get shape (n_steps, n_points)
        psi_t = sol.y.T
        
        return t_points, psi_t
    
    def position_expectation(self, psi):
        """Calculate the expectation value of position."""
        prob = np.abs(psi)**2
        prob = prob / np.sum(prob * self.dx)  # Ensure normalization
        return np.sum(self.x * prob * self.dx)
    
    def momentum_wavefunction(self, psi):
        """
        Transform the position wavefunction to momentum space using FFT.
        
        Args:
            psi (ndarray): Wavefunction in position space
            
        Returns:
            ndarray: Wavefunction in momentum space
        """
        # FFT to transform to momentum space
        psi_k = np.fft.fft(psi) * self.dx / np.sqrt(2 * np.pi)
        return psi_k
    
    def momentum_expectation(self, psi):
        """Calculate the expectation value of momentum."""
        psi_k = self.momentum_wavefunction(psi)
        prob_k = np.abs(psi_k)**2
        dk = self.k[1] - self.k[0]
        prob_k = prob_k / np.sum(prob_k * dk)  # Ensure normalization
        return np.sum(self.k * prob_k * dk)
    
    def energy_expectation(self, psi):
        """Calculate the expectation value of energy."""
        # Kinetic energy
        psi_k = self.momentum_wavefunction(psi)
        T = 0.5 * np.sum(np.abs(psi_k)**2 * self.k**2) * (self.k[1] - self.k[0])
        
        # Potential energy
        V = 0.5 * self.omega**2 * np.sum(np.abs(psi)**2 * self.x**2) * self.dx
        
        return T + V
    
    def generate_training_data(self, n_initial_states=10, t_max=10.0, n_steps=100):
        """
        Generate training data for the Hamiltonian Neural Network.
        
        Args:
            n_initial_states (int): Number of different initial states
            t_max (float): Maximum evolution time
            n_steps (int): Number of time steps
            
        Returns:
            tuple: (q, p, dq_dt, dp_dt) training data
        """
        q_data = []
        p_data = []
        dq_dt_data = []
        dp_dt_data = []
        
        for i in range(n_initial_states):
            # Vary the initial state parameters
            x0 = np.random.uniform(-2, 2)
            sigma = np.random.uniform(0.3, 1.0)
            
            # Add momentum to the initial state by introducing a phase factor
            # This creates a moving wave packet
            k0 = np.random.uniform(-2, 2)  # Initial momentum
            
            # Create initial state with momentum
            psi_0 = np.exp(-(self.x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * self.x)
            psi_0 = psi_0 / np.sqrt(np.sum(np.abs(psi_0)**2) * self.dx)  # Normalize
            
            # Evolve the state
            t_points, psi_t = self.evolve(psi_0, t_max=t_max, n_steps=n_steps)
            
            # Calculate position and momentum expectation values
            q = np.array([self.position_expectation(psi) for psi in psi_t])
            
            # Calculate momentum expectation values
            p = np.array([self.momentum_expectation(psi) for psi in psi_t])
            
            # Calculate derivatives (using finite differences)
            dt = t_points[1] - t_points[0]
            dq_dt = np.gradient(q, dt)
            dp_dt = np.gradient(p, dt)
            
            # Append to data arrays
            q_data.append(q)
            p_data.append(p)
            dq_dt_data.append(dq_dt)
            dp_dt_data.append(dp_dt)
        
        # Combine all data
        q_data = np.concatenate(q_data)
        p_data = np.concatenate(p_data)
        dq_dt_data = np.concatenate(dq_dt_data)
        dp_dt_data = np.concatenate(dp_dt_data)
        
        # Stack q and p for input to the neural network
        qp_data = np.stack([q_data, p_data], axis=1)
        derivatives = np.stack([dq_dt_data, dp_dt_data], axis=1)
        
        return qp_data, derivatives
    
    def plot_evolution(self, t_points, psi_t):
        """
        Plot the evolution of the wavefunction.
        
        Args:
            t_points (ndarray): Time points
            psi_t (ndarray): Evolved wavefunctions with shape (n_steps, n_points)
        """
        plt.figure(figsize=(12, 8))
        
        # Plot initial, middle, and final wavefunctions
        plt.subplot(2, 2, 1)
        plt.plot(self.x, np.abs(psi_t[0])**2, label='Initial')
        plt.plot(self.x, np.abs(psi_t[len(t_points)//2])**2, label='Middle')
        plt.plot(self.x, np.abs(psi_t[-1])**2, label='Final')
        plt.xlabel('Position (x)')
        plt.ylabel('Probability density')
        plt.legend()
        plt.title('Wavefunction Evolution')
        
        # Plot position expectation value over time
        plt.subplot(2, 2, 2)
        x_expect = np.array([self.position_expectation(psi) for psi in psi_t])
        plt.plot(t_points, x_expect)
        plt.xlabel('Time')
        plt.ylabel('⟨x⟩')
        plt.title('Position Expectation Value')
        
        # Plot momentum expectation value over time
        plt.subplot(2, 2, 3)
        p_expect = np.array([self.momentum_expectation(psi) for psi in psi_t])
        plt.plot(t_points, p_expect)
        plt.xlabel('Time')
        plt.ylabel('⟨p⟩')
        plt.title('Momentum Expectation Value')
        
        # Plot energy expectation value over time
        plt.subplot(2, 2, 4)
        e_expect = np.array([self.energy_expectation(psi) for psi in psi_t])
        plt.plot(t_points, e_expect)
        plt.xlabel('Time')
        plt.ylabel('⟨E⟩')
        plt.title('Energy Expectation Value')
        
        plt.tight_layout()
        plt.savefig('quantum_harmonic_oscillator_evolution.png')
        plt.close()

def main():
    """Run a demonstration of the quantum harmonic oscillator simulation."""
    # Create the oscillator
    qho = QuantumHarmonicOscillator(n_points=256, x_range=(-10, 10), omega=1.0)
    
    # Create an initial state
    psi_0 = qho.initial_state(x0=-2.0, sigma=0.5)
    
    # Evolve the state
    t_points, psi_t = qho.evolve(psi_0, t_max=10.0, n_steps=100)
    
    # Plot the evolution
    qho.plot_evolution(t_points, psi_t)
    
    # Generate training data for the Hamiltonian Neural Network
    qp_data, derivatives = qho.generate_training_data(n_initial_states=5)
    
    print(f"Generated {len(qp_data)} training samples")
    print(f"Input shape (q,p): {qp_data.shape}")
    print(f"Output shape (dq/dt, dp/dt): {derivatives.shape}")
    
    # Save the training data
    np.savez('harmonic_oscillator_data.npz', 
             qp=qp_data, 
             derivatives=derivatives)

if __name__ == "__main__":
    main()
