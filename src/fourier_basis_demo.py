"""
Fourier Basis and Unitary Transformations Demo

This script demonstrates the application of Fourier and unitary transformations
as basis changes in quantum systems and neural networks. It also compares
the downstream performance (trajectory accuracy, energy conservation) of
a Unitary HNN and a standard HNN.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# Assuming these modules exist in the same directory or are installed
# Add error handling for imports
try:
    from quantum_harmonic_oscillator import QuantumHarmonicOscillator
    from unitary_transforms import (
        FourierTransformLayer,
        UnitaryLayer,
        MeasurementBasisChange,
        FourierRecurrentUnit
    )
    # Ensure the standard HNN class is also imported correctly
    from hamiltonian_neural_network import HamiltonianNeuralNetwork
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure quantum_harmonic_oscillator.py, unitary_transforms.py, and hamiltonian_neural_network.py are accessible.")
    exit() # Exit if essential modules are missing

# Import solve_ivp for better integration
from scipy.integrate import solve_ivp
import os # To check for model file existence
import traceback # For detailed error reporting

# --- Class Definitions (UnitaryHamiltonianNet, FourierBasisMeasurement) ---
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
        # Create input tensor. Ensure it requires grad if the original input did.
        qp = torch.cat([q, p], dim=1)
        
        # Set requires_grad=True if not already set
        if not qp.requires_grad:
            qp = qp.detach().requires_grad_(True)

        # Compute Hamiltonian
        H = self.forward(qp)

        # Compute gradients using autograd
        dH = None # Initialize dH
        try:
            # Now we can safely compute gradients
            dH = torch.autograd.grad(
                H.sum(), qp, create_graph=True, retain_graph=True
            )[0]
        except RuntimeError as e:
            print(f"RuntimeError during autograd.grad: {e}")
            # Return zero gradients or re-raise, depending on desired behavior
            dH = torch.zeros_like(qp) # Return zeros if grad fails

        # If dH is still None (e.g., due to an unexpected issue), create zeros
        if dH is None:
            print("Error: dH calculation failed unexpectedly. Returning zero gradients.")
            dH = torch.zeros_like(qp)

        # Extract gradients
        dH_dq = dH[:, 0:1]
        dH_dp = dH[:, 1:2]

        return dH_dq, dH_dp

    def dynamics(self, qp):
        """
        Compute the Hamiltonian dynamics (dq/dt, dp/dt).

        Args:
            qp (torch.Tensor): Input tensor of shape (batch_size, 2) containing (q, p).
                               MUST require gradients if used during training for backward().

        Returns:
            torch.Tensor: Time derivatives (dq/dt, dp/dt)
        """
        # Detach q and p for gradient computation input if necessary,
        # but ensure the original qp requires grad.
        # It's cleaner to ensure qp passed in requires grad.
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
             try:
                 layer.project_to_unitary()
             except Exception as e:
                 print(f"Error projecting unitary layer: {e}")


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
            n_bases (int): Number of measurement bases (excluding position and momentum)
        """
        self.n_points = n_points
        self.x_range = x_range
        self.x = np.linspace(x_range[0], x_range[1], n_points)
        self.dx = (x_range[1] - x_range[0]) / (n_points - 1) if n_points > 1 else 0

        # For momentum space
        self.k = 2 * np.pi * np.fft.fftfreq(n_points, self.dx) if self.dx != 0 else np.zeros(n_points)

        # Create measurement basis change module
        # n_bases here refers to the number of *learnable* unitary bases
        try:
            # Need to import MeasurementBasisChange from unitary_transforms
            from unitary_transforms import MeasurementBasisChange
            self.mbc = MeasurementBasisChange(n_points, n_bases=n_bases, complex_input=True)
            self.n_learnable_bases = n_bases
        except Exception as e:
            print(f"Error initializing MeasurementBasisChange: {e}")
            self.mbc = None
            self.n_learnable_bases = 0


    def wavefunction_to_tensor(self, psi):
        """
        Convert a complex wavefunction to a tensor representation [real; imag].

        Args:
            psi (ndarray): Complex wavefunction of shape (n_points,)

        Returns:
            torch.Tensor: Tensor representation of shape (1, 2 * n_points)
        """
        # Extract real and imaginary parts
        psi_real = np.real(psi)
        psi_imag = np.imag(psi)

        # Combine into a single array [real_part, imag_part]
        # Expected shape by MeasurementBasisChange seems to be concatenated
        psi_combined = np.concatenate([psi_real, psi_imag])

        # Convert to tensor and add batch dimension
        return torch.tensor(psi_combined, dtype=torch.float32).unsqueeze(0)

    def measure_in_multiple_bases(self, psi):
        """
        Calculate probability distributions in position, momentum, and learned unitary bases.

        Args:
            psi (ndarray): Complex wavefunction

        Returns:
            dict: Dictionary mapping basis name to probability distribution (ndarray)
        """
        if self.mbc is None:
            print("MeasurementBasisChange module not initialized. Cannot measure.")
            return {'position': np.abs(psi)**2} # Return at least position

        # Convert to tensor representation expected by MeasurementBasisChange
        # Assumes psi is complex of shape (n_points,)
        psi_tensor = self.wavefunction_to_tensor(psi)

        # Measure in different bases
        results = {}

        # Position basis (original) - calculated directly
        results['position'] = np.abs(psi)**2

        # Momentum basis (using Fourier transform layer in MBC)
        try:
            # Ensure output shape matches n_points
            momentum_prob = self.mbc.measure_in_fourier_basis(psi_tensor).squeeze().detach().numpy()
            results['momentum'] = momentum_prob[:self.n_points] # Take the first n_points components
        except Exception as e:
            print(f"Error measuring in Fourier basis: {e}")
            results['momentum'] = np.zeros(self.n_points) # Placeholder

        # Other learnable unitary bases
        for i in range(self.n_learnable_bases):
            basis_name = f'basis_{i}'
            try:
                # Ensure output shape matches n_points
                prob = self.mbc.measure_in_basis(psi_tensor, i).squeeze().detach().numpy()
                results[basis_name] = prob[:self.n_points] # Take the first n_points components
            except Exception as e:
                print(f"Error measuring in basis {i}: {e}")
                results[basis_name] = np.zeros(self.n_points) # Placeholder


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

        # Number of subplots required
        n_plots = len(measurements)
        if n_plots == 0:
             print("No measurement results to visualize.")
             return

        # Create figure
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots), sharex=False)
        # Ensure axes is always an array, even if n_plots is 1
        if n_plots == 1:
            axes = [axes]

        # Plot each measurement
        for i, (basis_name, prob) in enumerate(measurements.items()):
            ax = axes[i]

            if basis_name == 'position':
                domain = self.x
                xlabel = 'Position (x)'
                # Normalize probability density for visualization
                norm_factor = (np.sum(prob) * self.dx) if self.dx > 0 and np.sum(prob) > 1e-9 else 1.0
                norm_prob = prob / norm_factor
                ax.plot(domain, norm_prob)
            elif basis_name == 'momentum':
                # Use fftshift for better visualization of momentum space
                domain = np.fft.fftshift(self.k)
                prob_shifted = np.fft.fftshift(prob)
                dk = abs(self.k[1] - self.k[0]) if len(self.k) > 1 else 0
                norm_factor = (np.sum(prob_shifted) * dk) if dk > 0 and np.sum(prob_shifted) > 1e-9 else 1.0
                norm_prob = prob_shifted / norm_factor
                ax.plot(domain, norm_prob)
                xlabel = 'Momentum (k)'
            else:
                # For other bases, use indices as the domain
                domain = np.arange(len(prob))
                xlabel = 'Basis Index'
                # Normalize discrete probability distribution
                norm_factor = np.sum(prob) if np.sum(prob) > 1e-9 else 1.0
                norm_prob = prob / norm_factor
                ax.plot(domain, norm_prob)

            ax.set_xlabel(xlabel)
            ax.set_ylabel('Probability / Density')
            ax.set_title(f'{basis_name.capitalize()} Basis')
            ax.grid(True)
            # Set appropriate x-limits if needed
            if basis_name != 'position' and len(domain)>0:
                 ax.set_xlim(min(domain), max(domain))


        plt.tight_layout()
        plt.suptitle(title, y=1.02, fontsize=16)
        # Ensure directory exists or handle saving appropriately
        plot_filename = f'{title.replace(" ", "_").replace("=", "").replace(".","").lower()}_basis_measurements.png'
        try:
            plt.savefig(plot_filename)
            #print(f"Saved basis plot: {plot_filename}") # Optional: uncomment for verbose output
        except Exception as e:
            print(f"Error saving plot '{plot_filename}': {e}")
        plt.close(fig) # Close the figure to free memory


# --- Training Function (train_unitary_hamiltonian_net) ---
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
    print(f"Starting Unitary HNN training for {epochs} epochs...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train() # Set model to training mode

        for qp_batch, deriv_batch in dataloader:
            # Zero gradients
            optimizer.zero_grad()

            # --- Ensure qp_batch requires gradients ---
            qp_batch.requires_grad_(True) # <--- Keep this line

            # Forward pass to get derivatives
            try:
                pred_derivatives = model.dynamics(qp_batch)
                # Compute loss
                loss = loss_fn(pred_derivatives, deriv_batch)
                # Backward pass
                loss.backward() # This should now work correctly
                # Update parameters
                optimizer.step()
                # Accumulate loss
                epoch_loss += loss.item()
            except Exception as e:
                print(f"\nError during Unitary HNN training step (Epoch {epoch+1}): {e}")
                print(f"qp_batch shape: {qp_batch.shape}, requires_grad: {qp_batch.requires_grad}")
                # Optionally skip batch or raise error
                continue # Skip this batch


        # Project to unitary group periodically
        if (epoch + 1) % project_every == 0:
            model.project_unitaries()

        # Average loss for the epoch
        avg_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0
        losses.append(avg_loss)

        # Print progress
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    # Final projection after training
    model.project_unitaries()
    print("Unitary HNN training finished.")
    model.eval() # Set model to evaluation mode after training

    # Save the model
    model_path = "unitary_hamiltonian_nn_model.pt"
    try:
      torch.save(model.state_dict(), model_path)
      print(f"Unitary HNN model saved to '{model_path}'")
    except Exception as e:
      print(f"Error saving Unitary HNN model: {e}")


    return model, losses


# --- Animation Function (animate_basis_changes) ---
# <<< Keep this function exactly as it was >>>
def animate_basis_changes(qho, t_points, psi_t, n_frames=100, filename='basis_changes_animation.gif'):
    """
    Create an animation of wavefunction evolution in different bases.

    Args:
        qho (QuantumHarmonicOscillator): Quantum harmonic oscillator instance
        t_points (ndarray): Time points
        psi_t (ndarray): Evolved wavefunctions with shape (n_steps, n_points)
        n_frames (int): Number of frames in the animation
        filename (str): Output filename for the animation GIF
    """
    print(f"Creating animation with {n_frames} frames...")
    # Create Fourier basis measurement system
    # Let's use 2 learnable bases for the animation (n_bases=2)
    fbm = FourierBasisMeasurement(
        n_points=qho.n_points,
        x_range=qho.x_range,
        n_bases=2 # Number of learnable unitary bases to show
    )
    if fbm.mbc is None: # Check if initialization failed
        print("Skipping animation due to MeasurementBasisChange initialization error.")
        return

    # Select frames to animate
    if t_points is None or len(t_points) <= 1:
        print("Warning: Not enough time points for animation.")
        return
    frame_indices = np.linspace(0, len(t_points)-1, n_frames, dtype=int)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten() # axes will be [pos_ax, mom_ax, basis0_ax, basis1_ax]

    # Initialize plots
    lines = []
    # Ensure domains are valid even if n_points=1
    pos_domain = qho.x if qho.n_points > 1 else [0]
    mom_domain = np.fft.fftshift(fbm.k) if qho.n_points > 1 else [0]
    idx_domain = np.arange(qho.n_points) if qho.n_points > 0 else [0]
    plot_domains = [pos_domain, mom_domain, idx_domain, idx_domain]
    plot_xlabels = ['Position (x)', 'Momentum (k)', 'Basis 0 Index', 'Basis 1 Index']
    plot_titles = ['Position Basis', 'Momentum Basis', 'Basis 0', 'Basis 1']

    for i, ax in enumerate(axes):
        line, = ax.plot([], [], lw=2) # Added line width
        lines.append(line)
        ax.set_ylim(0, None) # Auto-adjust y-axis upper limit initially
        try: # Handle potential errors with empty domains
             ax.set_xlim(min(plot_domains[i]), max(plot_domains[i]))
        except ValueError:
             ax.set_xlim(0, 1) # Default limits if domain is empty/invalid
        ax.set_xlabel(plot_xlabels[i])
        ax.set_ylabel('Probability / Density')
        ax.set_title(plot_titles[i])
        ax.grid(True)

    # Add time annotation
    time_text = fig.text(0.5, 0.96, '', ha='center', fontsize=12) # Adjusted position

    # Store max y values for stable limits
    max_y_values = [0.0] * len(axes)

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
        if psi_t is None or idx >= len(psi_t): # Check if psi_t is valid
             print(f"Warning: Invalid index {idx} for psi_t in animation frame {frame}")
             return lines + [time_text] # Return existing lines
        psi = psi_t[idx]
        t = t_points[idx]

        # Measure in different bases
        measurements = fbm.measure_in_multiple_bases(psi)

        # Data to plot
        plot_data = [
            measurements.get('position', np.array([])), # Use .get for safety
            np.fft.fftshift(measurements.get('momentum', np.array([]))), # Shift momentum for plotting
            measurements.get('basis_0', np.array([])),
            measurements.get('basis_1', np.array([]))
        ]

        # Normalize and update plots
        for i, line in enumerate(lines):
            domain = plot_domains[i]
            data = plot_data[i]

            if len(data) == 0 or len(domain) != len(data): # Skip if data is empty or mismatched
                line.set_data([], [])
                continue

            # Normalize based on basis type
            if i == 0: # Position (density)
                norm_factor = (np.sum(data) * fbm.dx) if fbm.dx > 0 and np.sum(data) > 1e-9 else 1.0
            elif i == 1: # Momentum (density)
                 dk = abs(fbm.k[1] - fbm.k[0]) if len(fbm.k) > 1 else 0
                 norm_factor = (np.sum(data) * dk) if dk > 0 and np.sum(data) > 1e-9 else 1.0
            else: # Discrete bases
                 norm_factor = np.sum(data) if np.sum(data) > 1e-9 else 1.0

            # Avoid division by zero or near-zero
            if norm_factor < 1e-12: norm_factor = 1.0
            norm_data = data / norm_factor
            line.set_data(domain, norm_data)

            # Update max y value if needed
            current_max = np.max(norm_data) if len(norm_data) > 0 else 0
            if current_max > max_y_values[i]:
                 max_y_values[i] = current_max
                 axes[i].set_ylim(0, max_y_values[i] * 1.1 + 1e-9) # Set ylim with padding, avoid zero range

        # Update time text
        time_text.set_text(f'Time: {t:.2f}')

        return lines + [time_text]

    # Create animation
    try:
        anim = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True, interval=50) # Added interval
        # Save animation
        anim.save(filename, writer='pillow', fps=15) # Adjusted fps
        print(f"Animation saved to '{filename}'")
    except Exception as e:
        print(f"Error creating or saving animation: {e}")
        traceback.print_exc() # Print detailed traceback for animation errors


    plt.close(fig) # Close the figure


# --- Helper Functions for Downstream Comparison ---
# <<< Keep these functions exactly as they were >>>
def simulate_hnn_dynamics_scipy(nn_model, qp_init, t_span, n_steps, device):
    """
    Simulate HNN dynamics using scipy.integrate.solve_ivp.

    Args:
        nn_model (nn.Module): Trained HNN model (Standard or Unitary).
        qp_init (ndarray): Initial state [q0, p0].
        t_span (tuple): Time interval (t_start, t_end).
        n_steps (int): Number of time steps to evaluate.
        device (torch.device): Device ('cpu' or 'cuda').

    Returns:
        tuple: (t_eval, trajectory) or (None, None) if solver fails.
               trajectory has shape (n_steps, 2).
    """
    if nn_model is None:
        print("DEBUG: Simulation skipped: Model is None.") # DEBUG
        return None, None

    model_name = type(nn_model).__name__
    print(f"DEBUG: Attempting simulation for {model_name}...") # DEBUG
    nn_model.eval() # Ensure model is in evaluation mode

    # Define the derivative function for the ODE solver
    def ode_func(t, qp_numpy):
        # Convert numpy array to torch tensor
        qp_tensor = torch.tensor([qp_numpy], dtype=torch.float32, requires_grad=True).to(device)
        try:
            # Remove the torch.no_grad() context since we need gradients
            # Get derivatives from the HNN model's dynamics method
            dqp_dt_tensor = nn_model.dynamics(qp_tensor)
            # Check for NaN/Inf values
            if torch.isnan(dqp_dt_tensor).any() or torch.isinf(dqp_dt_tensor).any():
                print(f"!!! DEBUG: NaN/Inf detected in {model_name} dynamics at t={t:.2f} !!! Input: {qp_numpy}") # DEBUG
                return np.zeros_like(qp_numpy)
            # Convert back to numpy array and explicitly take real part to avoid complex warning
            result = dqp_dt_tensor.cpu().detach().numpy().flatten()  # Add detach() here
            if np.iscomplexobj(result):
                print(f"DEBUG: Complex values detected in {model_name} dynamics at t={t:.2f}, taking real part.")
                result = np.real(result)
            return result
        except Exception as e:
            print(f"!!! DEBUG: Error during {model_name} dynamics calculation at t={t:.2f}: {e} !!! Input: {qp_numpy}") # DEBUG
            traceback.print_exc() # Print full traceback
            return np.zeros_like(qp_numpy)

    # Time points for evaluation
    t_eval = np.linspace(t_span[0], t_span[1], n_steps)

    # Solve the ODE with robust settings
    sol = None # Initialize sol
    try:
        print(f"DEBUG: Calling solve_ivp for {model_name}...") # DEBUG
        sol = solve_ivp(
            ode_func,
            t_span,
            qp_init,
            t_eval=t_eval,
            method='RK45', # Explicit Runge-Kutta method
            rtol=1e-6,
            atol=1e-8,
            max_step=0.1 # Limit max step size for potentially stiff problems
        )
        print(f"DEBUG: solve_ivp call finished for {model_name}.") # DEBUG

        if sol.success:
            print(f"DEBUG: {model_name} simulation successful. Trajectory shape: {sol.y.T.shape}") # DEBUG
            return sol.t, sol.y.T # Transpose to get shape (n_steps, 2)
        else:
            print(f"!!! DEBUG: Warning: ODE solver failed for {model_name}. Status: {sol.status}, Message: {sol.message} !!!") # DEBUG
            # Attempt to return partial results if available
            if hasattr(sol, 't') and hasattr(sol, 'y') and len(sol.t) > 0:
                 print(f"DEBUG: Returning partial results ({len(sol.t)} steps) for {model_name}.") # DEBUG
                 # Pad the results to the expected length with NaNs or last value? Padding with NaN is safer.
                 full_traj = np.full((n_steps, 2), np.nan)
                 valid_steps = len(sol.t)
                 # Find indices in t_eval corresponding to sol.t (approximate matching)
                 eval_indices = np.searchsorted(t_eval, sol.t, side='left')
                 # Ensure indices are within bounds
                 eval_indices = np.clip(eval_indices, 0, n_steps - 1)
                 # Fill valid steps
                 full_traj[eval_indices] = sol.y.T[:valid_steps] # Fill based on matching time points

                 # Check if the first step was successful
                 if valid_steps > 0:
                     return t_eval, full_traj # Return padded trajectory
                 else:
                     return None, None
            else:
                 return None, None
    except Exception as e:
        print(f"!!! DEBUG: Exception during solve_ivp for {model_name}: {e} !!!") # DEBUG
        traceback.print_exc() # Print full traceback
        return None, None


def calculate_learned_energy(nn_model, trajectory, device):
    """
    Calculate energy using the network's learned Hamiltonian H(q,p).

    Args:
        nn_model (nn.Module): Trained HNN model (Standard or Unitary).
        trajectory (ndarray): Simulated trajectory of shape (n_steps, 2).
        device (torch.device): Device ('cpu' or 'cuda').

    Returns:
        ndarray or None: Learned energy values shape (n_steps,) or None if error or invalid input.
    """
    if nn_model is None or trajectory is None or len(trajectory) == 0:
        print("DEBUG: Energy calculation skipped: Model or trajectory is None or empty.") # DEBUG
        return None

    model_name = type(nn_model).__name__
    print(f"DEBUG: Attempting energy calculation for {model_name}...") # DEBUG
    nn_model.eval() # Ensure model is in evaluation mode

    # Filter out NaN values from trajectory before converting to tensor
    valid_indices = ~np.isnan(trajectory).any(axis=1)
    if not np.any(valid_indices):
        print(f"DEBUG: No valid (non-NaN) trajectory points found for {model_name}. Cannot calculate energy.") # DEBUG
        return None

    valid_trajectory = trajectory[valid_indices]
    print(f"DEBUG: Calculating energy for {len(valid_trajectory)} valid points for {model_name}.") # DEBUG

    try:
        # Convert valid trajectory points to tensor
        qp_tensor = torch.tensor(valid_trajectory, dtype=torch.float32).to(device)

        # Get derivatives from the HNN model's dynamics method
        energy_valid = nn_model(qp_tensor)

        # Check for NaN/Inf in energy output
        if torch.isnan(energy_valid).any() or torch.isinf(energy_valid).any():
             print(f"!!! DEBUG: NaN/Inf detected in calculated energy for {model_name} !!!") # DEBUG
             return None

        # Create a full energy array with NaNs where trajectory was NaN
        full_energy = np.full(len(trajectory), np.nan)
        full_energy[valid_indices] = energy_valid.cpu().detach().numpy().flatten()

        print(f"DEBUG: Energy calculation successful for {model_name}. Output shape: {full_energy.shape}") # DEBUG
        return full_energy # Use flatten() for shape (n_steps,)

    except Exception as e:
        print(f"!!! DEBUG: Error calculating learned energy for {model_name}: {e} !!!") # DEBUG
        traceback.print_exc() # Print full traceback
        return None


def plot_dynamics_comparison(time_pts, true_qp, unitary_qp, standard_qp,
                             true_e_analytic, unitary_e_learned, standard_e_learned,
                             filename="hnn_dynamics_comparison.png"):
    """
    Plots comparison of trajectories and energy conservation.

    Args:
        time_pts (ndarray): Time points for the plots.
        true_qp (ndarray): True trajectory shape (n_steps, 2).
        unitary_qp (ndarray or None): Unitary HNN trajectory.
        standard_qp (ndarray or None): Standard HNN trajectory.
        true_e_analytic (ndarray): True analytical energy shape (n_steps,).
        unitary_e_learned (ndarray or None): Learned energy for Unitary HNN.
        standard_e_learned (ndarray or None): Learned energy for Standard HNN.
        filename (str): Output filename for the plot.
    """
    # Ensure time_pts is valid
    if time_pts is None or len(time_pts) == 0:
        print("DEBUG: Cannot generate dynamics plot: Invalid time points.") # DEBUG
        return

    print(f"DEBUG: Generating dynamics comparison plot: {filename}...") # DEBUG
    plt.style.use('seaborn-v0_8-darkgrid') # Use a nice style
    fig, axes = plt.subplots(2, 2, figsize=(14, 12)) # Slightly larger figure
    fig.suptitle("Dynamics and Energy Conservation Comparison", fontsize=18, y=0.98)

    # --- Phase Space ---
    ax = axes[0, 0]
    plot_count = 0
    if true_qp is not None:
        ax.plot(true_qp[:, 0], true_qp[:, 1], 'r-', label='True', linewidth=2.5, alpha=0.8)
        plot_count += 1
    if unitary_qp is not None:
        valid_u = ~np.isnan(unitary_qp).any(axis=1)
        if np.any(valid_u):
            ax.plot(unitary_qp[valid_u, 0], unitary_qp[valid_u, 1], 'b--', label='Unitary HNN', linewidth=1.5)
            plot_count += 1
    if standard_qp is not None:
        valid_s = ~np.isnan(standard_qp).any(axis=1)
        if np.any(valid_s):
            ax.plot(standard_qp[valid_s, 0], standard_qp[valid_s, 1], 'g-.', label='Standard HNN', linewidth=1.5)
            plot_count += 1
    ax.set_xlabel('Position (q)')
    ax.set_ylabel('Momentum (p)')
    ax.set_title('Phase Space Trajectories')
    if plot_count > 0: ax.legend(fontsize='medium')
    ax.grid(True, linestyle=':')
    ax.axis('equal')

    # --- Position vs Time ---
    ax = axes[0, 1]
    plot_count = 0
    if true_qp is not None:
        ax.plot(time_pts, true_qp[:, 0], 'r-', label='True', linewidth=2.5, alpha=0.8)
        plot_count += 1
    if unitary_qp is not None:
        valid_u = ~np.isnan(unitary_qp).any(axis=1)
        if np.any(valid_u):
            ax.plot(time_pts[valid_u], unitary_qp[valid_u, 0], 'b--', label='Unitary HNN', linewidth=1.5)
            plot_count += 1
    if standard_qp is not None:
        valid_s = ~np.isnan(standard_qp).any(axis=1)
        if np.any(valid_s):
            ax.plot(time_pts[valid_s], standard_qp[valid_s, 0], 'g-.', label='Standard HNN', linewidth=1.5)
            plot_count += 1
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Position (q)')
    ax.set_title('Position vs Time')
    if plot_count > 0: ax.legend(fontsize='medium')
    ax.grid(True, linestyle=':')

    # --- Momentum vs Time ---
    ax = axes[1, 0]
    plot_count = 0
    if true_qp is not None:
        ax.plot(time_pts, true_qp[:, 1], 'r-', label='True', linewidth=2.5, alpha=0.8)
        plot_count += 1
    if unitary_qp is not None:
        valid_u = ~np.isnan(unitary_qp).any(axis=1)
        if np.any(valid_u):
            ax.plot(time_pts[valid_u], unitary_qp[valid_u, 1], 'b--', label='Unitary HNN', linewidth=1.5)
            plot_count += 1
    if standard_qp is not None:
        valid_s = ~np.isnan(standard_qp).any(axis=1)
        if np.any(valid_s):
            ax.plot(time_pts[valid_s], standard_qp[valid_s, 1], 'g-.', label='Standard HNN', linewidth=1.5)
            plot_count += 1
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Momentum (p)')
    ax.set_title('Momentum vs Time')
    if plot_count > 0: ax.legend(fontsize='medium')
    ax.grid(True, linestyle=':')

    # --- Energy Conservation ---
    ax = axes[1, 1]
    base_energy = None
    plot_count = 0
    if true_e_analytic is not None and len(true_e_analytic) > 0:
        valid_true_e = ~np.isnan(true_e_analytic)
        if np.any(valid_true_e):
             ax.plot(time_pts[valid_true_e], true_e_analytic[valid_true_e], 'r-', label=f'True Energy ({true_e_analytic[valid_true_e][0]:.3f})', linewidth=2.5, alpha=0.8)
             base_energy = true_e_analytic[valid_true_e][0] # Use first valid true energy as base
             plot_count += 1


    if unitary_e_learned is not None and len(unitary_e_learned) > 0:
        valid_u_e = ~np.isnan(unitary_e_learned)
        if np.any(valid_u_e):
            label_u = f'Unitary HNN Learned E (Start: {unitary_e_learned[valid_u_e][0]:.3f})'
            if base_energy is not None:
                 delta_e_u = np.abs(unitary_e_learned[valid_u_e] - base_energy)
                 label_u += f' Max ΔE: {np.max(delta_e_u):.3f}'
            ax.plot(time_pts[valid_u_e], unitary_e_learned[valid_u_e], 'b--', label=label_u, linewidth=1.5)
            plot_count += 1
        else:
            print("DEBUG: Unitary HNN learned energy contains only NaNs.") #DEBUG


    if standard_e_learned is not None and len(standard_e_learned) > 0:
        valid_s_e = ~np.isnan(standard_e_learned)
        if np.any(valid_s_e):
            label_s = f'Standard HNN Learned E (Start: {standard_e_learned[valid_s_e][0]:.3f})'
            if base_energy is not None:
                delta_e_s = np.abs(standard_e_learned[valid_s_e] - base_energy)
                label_s += f' Max ΔE: {np.max(delta_e_s):.3f}'
            ax.plot(time_pts[valid_s_e], standard_e_learned[valid_s_e], 'g-.', label=label_s, linewidth=1.5)
            plot_count += 1
        else:
            print("DEBUG: Standard HNN learned energy contains only NaNs.") #DEBUG


    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Energy (H)')
    ax.set_title('Energy Conservation (Learned Hamiltonian)')
    if plot_count > 0: ax.legend(fontsize='medium')
    ax.grid(True, linestyle=':')

    # Adjust y-limits for energy plot if necessary to show variations clearly
    all_energies = [e for e in [true_e_analytic, unitary_e_learned, standard_e_learned] if e is not None and len(e) > 0]
    valid_energies = [e[~np.isnan(e)] for e in all_energies if np.any(~np.isnan(e))] # Filter out NaNs before concatenating

    if len(valid_energies) > 0:
      concatenated_energies = np.concatenate(valid_energies)
      if len(concatenated_energies) > 0:
          min_e, max_e = np.min(concatenated_energies), np.max(concatenated_energies)
          if max_e > min_e + 1e-9: # Avoid issues with constant energy
              padding = (max_e - min_e) * 0.1 + 1e-9 # Add small padding
              ax.set_ylim(min_e - padding, max_e + padding)
          else: # Handle case where energy is constant
              ax.set_ylim(min_e - 0.1, max_e + 0.1)


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    try:
        plt.savefig(filename)
        print(f"DEBUG: Dynamics comparison plot saved successfully to '{filename}'") # DEBUG
    except Exception as e:
        print(f"!!! DEBUG: Error saving plot '{filename}': {e} !!!") # DEBUG
        traceback.print_exc()
    plt.close(fig) # Close the figure


# ==========================================
# MODIFIED MAIN FUNCTION
# ==========================================
def main():
    """Main function to demonstrate Fourier and unitary transformations."""
    print("=" * 80)
    print("Fourier and Unitary Transformations Demo")
    print("=" * 80)

    # --- Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    np.random.seed(42) # for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # --- Step 1: Quantum Harmonic Oscillator Setup ---
    print("\n--- Step 1: Quantum Harmonic Oscillator Setup ---")
    try:
        qho = QuantumHarmonicOscillator(n_points=128, x_range=(-6, 6), omega=1.0) # Slightly wider range
    except Exception as e:
        print(f"Error initializing QuantumHarmonicOscillator: {e}")
        return # Cannot proceed without QHO

    # Create an initial state for visualization/FRU
    try:
        psi_0_viz = qho.initial_state(x0=-1.0, sigma=0.5)
        t_points_viz, psi_t_viz = qho.evolve(psi_0_viz, t_max=10.0, n_steps=101) # Ensure >1 step
        if t_points_viz is None or psi_t_viz is None:
             raise ValueError("QHO evolution failed for visualization state.")
        print("QHO initialized and visualization state evolved.")
    except Exception as e:
        print(f"Error creating/evolving visualization state: {e}")
        # Allow script to continue if only visualization fails, but warn user
        t_points_viz, psi_t_viz = None, None


    # --- Step 2 & 3: Basis Change Visualization (Optional - can be slow) ---
    run_basis_viz = False # Set to True to run these steps
    if run_basis_viz and t_points_viz is not None:
        print("\n--- Step 2: Demonstrating Measurement in Different Bases ---")
        try:
            fbm = FourierBasisMeasurement(n_points=qho.n_points, x_range=qho.x_range, n_bases=2) # 2 learnable bases
            if fbm.mbc is not None: # Check if initialized correctly
                for t_idx in [0, 25, 50, 75, 100]:
                    if t_idx < len(psi_t_viz):
                        psi = psi_t_viz[t_idx]
                        fbm.visualize_basis_measurements(
                            psi,
                            title=f"Wavefunction_Bases_t={t_points_viz[t_idx]:.2f}"
                        )
                print("Basis measurements visualization complete.")
            else:
                print("Skipping basis visualization due to FBM init error.")
        except Exception as e:
            print(f"Error during basis visualization: {e}")


        print("\n--- Step 3: Creating Animation of Basis Changes ---")
        try:
            animate_basis_changes(qho, t_points_viz, psi_t_viz, n_frames=50, filename='basis_changes_animation.gif')
        except Exception as e:
            print(f"Error during animation creation: {e}")
    else:
        print("\n--- Steps 2 & 3: Basis Visualization Skipped ---")

    # --- Step 4: Generate Training Data ---
    print("\n--- Step 4: Generating Training Data ---")
    try:
        # Use slightly more data for potentially better training
        qp_data, derivatives = qho.generate_training_data(n_initial_states=25, n_steps=201, t_max=15.0) # More diverse data
        if qp_data is None or derivatives is None:
             raise ValueError("Training data generation failed.")
        print(f"Generated {len(qp_data)} training samples")
    except Exception as e:
        print(f"Error generating training data: {e}")
        return # Cannot proceed without training data

    # --- Training Parameters ---
    epochs = 600 # Increased epochs
    batch_size = 128 # Larger batch size
    hidden_dim = 128 # Increased hidden dimension
    learning_rate = 5e-4 # Adjusted learning rate
    project_every = 5 # Project more frequently
    n_unitary_layers = 3 # Number of unitary layers for UnitaryHNN

    # --- Step 5: Train Unitary HNN ---
    print("\n--- Step 5: Training Unitary HNN ---")
    unitary_hnn_trained, unitary_losses = None, []
    try:
        unitary_hnn_trained, unitary_losses = train_unitary_hamiltonian_net(
            qp_data,
            derivatives,
            epochs=epochs,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            n_unitary_layers=n_unitary_layers,
            learning_rate=learning_rate,
            project_every=project_every
        )
        # Plot the training loss for Unitary HNN
        plt.figure(figsize=(10, 6))
        plt.plot(unitary_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss for Unitary Hamiltonian Network')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig('unitary_hnn_training_loss.png')
        plt.close()
        print("Unitary HNN training complete. Loss plot saved.")
    except Exception as e:
        print(f"!!! Error during Unitary HNN training: {e} !!!")
        traceback.print_exc()
        # unitary_hnn_trained will remain None

    # --- Step 6: Train Standard HNN ---
    print("\n--- Step 6: Training Standard HNN ---")
    standard_hnn_model_path = "hamiltonian_nn_model.pt"
    standard_losses = []
    try:
        standard_hnn = HamiltonianNeuralNetwork(hidden_dim=hidden_dim, learning_rate=learning_rate)
        standard_losses = standard_hnn.train(
            qp_data,
            derivatives,
            batch_size=batch_size,
            epochs=epochs,
            print_every=100, # Print less often for longer training
            save_model=True # Saves to standard_hnn_model_path
        )
        print("Standard HNN training complete.")
    except Exception as e:
        print(f"!!! Error during Standard HNN training: {e} !!!")
        traceback.print_exc()


    # Plot comparison of training losses (if both trainings ran)
    if unitary_losses and standard_losses:
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(unitary_losses, label=f'Unitary HNN (Final: {unitary_losses[-1]:.4f})')
            plt.plot(standard_losses, label=f'Standard HNN (Final: {standard_losses[-1]:.4f})')
            plt.xlabel('Epoch')
            plt.ylabel('Loss (MSE on derivatives)')
            plt.title('Training Loss Comparison')
            plt.yscale('log')
            plt.legend()
            plt.grid(True)
            plt.savefig('hnn_loss_comparison.png')
            plt.close()
            print("Training loss comparison plot saved to 'hnn_loss_comparison.png'")
        except Exception as e:
            print(f"Error generating loss comparison plot: {e}")
    elif unitary_losses:
         print("Standard HNN losses not available, skipping loss comparison plot.")
    elif standard_losses:
         print("Unitary HNN losses not available, skipping loss comparison plot.")
    else:
         print("Neither model produced losses, skipping loss comparison plot.")


    # --- Step 7: Load Models for Evaluation ---
    print("\n--- Step 7: Loading Trained Models for Evaluation ---")
    # Unitary HNN is already in memory (if training succeeded)
    if unitary_hnn_trained:
        unitary_hnn_trained.eval() # Set to evaluation mode
        print("DEBUG: Unitary HNN model available from training.") # DEBUG
    else:
        # Attempt to load if training failed but model file exists
        unitary_model_path = "unitary_hamiltonian_nn_model.pt"
        if os.path.exists(unitary_model_path):
             print(f"DEBUG: Attempting to load Unitary HNN from {unitary_model_path}...") # DEBUG
             try:
                 unitary_hnn_trained = UnitaryHamiltonianNet(input_dim=2, hidden_dim=hidden_dim, n_unitary_layers=n_unitary_layers).to(device)
                 unitary_hnn_trained.load_state_dict(torch.load(unitary_model_path, map_location=device))
                 unitary_hnn_trained.eval()
                 print("DEBUG: Unitary HNN model loaded successfully from file.") # DEBUG
             except Exception as e:
                 print(f"DEBUG: Error loading Unitary HNN from file: {e}") # DEBUG
                 unitary_hnn_trained = None # Ensure it's None
        else:
             print("DEBUG: Unitary HNN training failed and model file not found.") # DEBUG


    # Load Standard HNN
    standard_hnn_trained = None # Initialize as None
    standard_hnn_available = False
    if os.path.exists(standard_hnn_model_path):
        print(f"DEBUG: Attempting to load Standard HNN from {standard_hnn_model_path}...") # DEBUG
        try:
            # Must instantiate with the same parameters used during training
            standard_hnn_trained = HamiltonianNeuralNetwork(hidden_dim=hidden_dim).to(device)
            standard_hnn_trained.load_state_dict(torch.load(standard_hnn_model_path, map_location=device))
            standard_hnn_trained.eval() # Set to evaluation mode
            print("DEBUG: Standard HNN model loaded successfully.") # DEBUG
            standard_hnn_available = True
        except Exception as e:
            print(f"DEBUG: Error loading Standard HNN model from '{standard_hnn_model_path}': {e}") # DEBUG
            standard_hnn_trained = None # Ensure it's None if loading failed
    else:
        print(f"DEBUG: Standard HNN model file '{standard_hnn_model_path}' not found. Cannot load for comparison.") # DEBUG

    # --- ADDED DEBUG PRINT ---
    print(f"DEBUG: Status before Step 8: unitary_hnn_trained is {'VALID' if unitary_hnn_trained else 'None'}, standard_hnn_available is {standard_hnn_available}")

    # --- Step 8: Simulate and Compare Downstream Performance ---
    print("\n--- Step 8: Simulating and Comparing Downstream Performance ---")
    print("DEBUG: Entering Step 8...") # DEBUG

    # --- REMOVED SKIP CONDITION FOR DEBUGGING ---
    # if unitary_hnn_trained is None and not standard_hnn_available:
    #     print("Neither HNN model is available for simulation. Skipping Step 8.")
    # else:
    # Always try to run the simulation part now for debugging

    # Simulation Setup
    qp0_sim = np.array([2.0, 0.0]) # Different initial condition for testing (starts at max q)
    sim_t_max = 30.0 # Longer time duration
    sim_n_steps = 601 # More steps for smoother plots
    sim_t_span = (0, sim_t_max)
    print(f"DEBUG: Simulation parameters: qp0={qp0_sim}, t_max={sim_t_max}, n_steps={sim_n_steps}") # DEBUG

    # Simulate Trajectories using solve_ivp
    print("DEBUG: Simulating Unitary HNN trajectory...") # DEBUG
    sim_t_eval_u, unitary_traj = simulate_hnn_dynamics_scipy(unitary_hnn_trained, qp0_sim, sim_t_span, sim_n_steps, device)
    print(f"DEBUG: Unitary HNN simulation result: t_eval is {'VALID' if sim_t_eval_u is not None else 'None'}, trajectory is {'VALID' if unitary_traj is not None else 'None'}") # DEBUG

    print("DEBUG: Simulating Standard HNN trajectory...") # DEBUG
    sim_t_eval_s, standard_traj = simulate_hnn_dynamics_scipy(standard_hnn_trained, qp0_sim, sim_t_span, sim_n_steps, device)
    print(f"DEBUG: Standard HNN simulation result: t_eval is {'VALID' if sim_t_eval_s is not None else 'None'}, trajectory is {'VALID' if standard_traj is not None else 'None'}") # DEBUG


    # Use the time evaluation points from a successful simulation (prefer unitary if both ok)
    sim_t_eval = sim_t_eval_u if sim_t_eval_u is not None else sim_t_eval_s
    print(f"DEBUG: Using sim_t_eval from {'Unitary' if sim_t_eval_u is not None else 'Standard (or None)'}") # DEBUG

    # Get True Trajectory (Analytical for QHO with omega=1)
    true_traj_sim = None
    true_energy_analytic = None
    if sim_t_eval is not None: # Check if simulation time points are valid
        try:
            print("DEBUG: Calculating true trajectory...") # DEBUG
            true_q_sim = qp0_sim[0] * np.cos(sim_t_eval) + qp0_sim[1] * np.sin(sim_t_eval)
            true_p_sim = -qp0_sim[0] * np.sin(sim_t_eval) + qp0_sim[1] * np.cos(sim_t_eval)
            true_traj_sim = np.stack([true_q_sim, true_p_sim], axis=1)
            # True Energy H = 0.5 * (p^2 + q^2) (assuming omega=1, m=1)
            true_energy_analytic = 0.5 * (true_traj_sim[:, 1]**2 + true_traj_sim[:, 0]**2)
            print("DEBUG: Calculated true analytical trajectory and energy.") # DEBUG
        except Exception as e:
            print(f"!!! DEBUG: Error calculating true trajectory: {e} !!!") # DEBUG
            traceback.print_exc()
    else:
        print("DEBUG: Cannot calculate true trajectory as sim_t_eval is None.") # DEBUG


    # Calculate Learned Energies
    print("DEBUG: Calculating learned energies...") # DEBUG
    unitary_energy_learned = calculate_learned_energy(unitary_hnn_trained, unitary_traj, device)
    print(f"DEBUG: Unitary HNN energy calculated: {'VALID' if unitary_energy_learned is not None else 'None'}") # DEBUG
    standard_energy_learned = calculate_learned_energy(standard_hnn_trained, standard_traj, device)
    print(f"DEBUG: Standard HNN energy calculated: {'VALID' if standard_energy_learned is not None else 'None'}") # DEBUG


    # Plot Comparison (only if true trajectory is available)
    print("DEBUG: Checking conditions for plotting dynamics comparison...") # DEBUG
    if sim_t_eval is not None and true_traj_sim is not None and true_energy_analytic is not None:
        print("DEBUG: Conditions met, calling plot_dynamics_comparison...") # DEBUG
        plot_dynamics_comparison(sim_t_eval, true_traj_sim, unitary_traj, standard_traj,
                                 true_energy_analytic, unitary_energy_learned, standard_energy_learned,
                                 filename="hnn_dynamics_comparison.png")
    else:
         print("DEBUG: Skipping dynamics comparison plot due to simulation failure or invalid time points.") # DEBUG

    # --- Step 9: Demonstrate Fourier Recurrent Unit ---
    print("\n--- Step 9: Demonstrating Fourier Recurrent Unit ---")
    # <<< Keep the FRU part exactly as it was >>>
    if t_points_viz is not None and psi_t_viz is not None:
        try:
            # Use original visualization time series
            q_series = np.array([qho.position_expectation(psi) for psi in psi_t_viz])
            p_series = np.array([qho.momentum_expectation(psi) for psi in psi_t_viz])

            time_series = np.stack([q_series, p_series], axis=1)
            # Add batch dimension and move to device
            time_series_tensor = torch.tensor(time_series, dtype=torch.float32).unsqueeze(0).to(device)

            # Create FRU and move to device
            fru = FourierRecurrentUnit(
                input_dim=2,
                hidden_dim=32, # Slightly larger FRU hidden dim
                output_dim=2,
                n_fourier_features=16 # More features
            ).to(device)

            # Forward pass
            fru.eval() # Set to eval mode
            with torch.no_grad():
              outputs, hidden = fru(time_series_tensor)

            # Convert output to numpy
            outputs_np = outputs.cpu().detach().numpy()[0]

            # Plot comparison
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True) # Share x-axis

            axes[0].plot(t_points_viz, q_series, 'b-', label='True <q>', alpha=0.8)
            axes[0].plot(t_points_viz, outputs_np[:, 0], 'r--', label='FRU Output q')
            axes[0].set_ylabel('Position Expectation')
            axes[0].legend()
            axes[0].grid(True, linestyle=':')
            axes[0].set_title('Position Expectation vs FRU Output')

            axes[1].plot(t_points_viz, p_series, 'b-', label='True <p>', alpha=0.8)
            axes[1].plot(t_points_viz, outputs_np[:, 1], 'r--', label='FRU Output p')
            axes[1].set_xlabel('Time')
            axes[1].set_ylabel('Momentum Expectation')
            axes[1].legend()
            axes[1].grid(True, linestyle=':')
            axes[1].set_title('Momentum Expectation vs FRU Output')

            plt.tight_layout()
            plt.savefig('fourier_recurrent_unit_comparison.png')
            plt.close(fig)

            print("Fourier Recurrent Unit demonstration complete.")
            print("Plot saved to 'fourier_recurrent_unit_comparison.png'")
        except Exception as e:
            print(f"!!! Error during FRU demonstration: {e} !!!")
            traceback.print_exc()
    else:
        print("Skipping FRU demonstration due to missing visualization data.")


    print("\nAll demonstrations finished!")


if __name__ == "__main__":
    # Wrap main execution in a try-except block for overall error catching
    try:
        main()
    except Exception as e:
        print("\n!!! An unexpected error occurred during script execution: !!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        print("\n--- Traceback ---")
        traceback.print_exc()
        print("-----------------") # FIX: Added closing quote here
