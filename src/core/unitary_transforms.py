"""
Unitary Transformations Module

This module implements Fourier and unitary transformations as basis changes
for quantum measurements and neural network architectures.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import expm, dft
import matplotlib.pyplot as plt

class FourierTransformLayer(nn.Module):
    """
    Neural network layer that applies a Fourier transform as a basis change.
    Can be learned or fixed.
    """
    def __init__(self, dim, learnable=False):
        """
        Initialize the Fourier transform layer.
        
        Args:
            dim (int): Dimension of the input/output
            learnable (bool): Whether the Fourier matrix is learnable
        """
        super(FourierTransformLayer, self).__init__()
        
        # Initialize with the DFT matrix
        dft_matrix = dft(dim)
        fourier_real = np.real(dft_matrix)
        fourier_imag = np.imag(dft_matrix)
        
        # Combine real and imaginary parts for a real-valued network
        # [real, -imag; imag, real] block form for complex multiplication
        fourier_matrix = np.block([
            [fourier_real, -fourier_imag],
            [fourier_imag, fourier_real]
        ]) / np.sqrt(dim)
        
        # Create parameter or buffer
        if learnable:
            self.weight = nn.Parameter(torch.tensor(
                fourier_matrix, dtype=torch.float32))
        else:
            self.register_buffer('weight', torch.tensor(
                fourier_matrix, dtype=torch.float32))
    
    def forward(self, x):
        """
        Apply the Fourier transform.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim)
            
        Returns:
            torch.Tensor: Transformed tensor
        """
        # For a complex input represented as [real; imag], we need to pad
        batch_size = x.shape[0]
        dim = x.shape[1]
        
        # If input is real-valued, pad with zeros for imaginary part
        if x.shape[1] == self.weight.shape[0] // 2:
            x_padded = torch.cat([x, torch.zeros_like(x)], dim=1)
        else:
            x_padded = x
            
        # Apply the transform
        output = F.linear(x_padded, self.weight)
        
        return output


class UnitaryLayer(nn.Module):
    """
    Neural network layer that applies a unitary transformation as a basis change.
    """
    def __init__(self, dim, complex_input=False, init='identity'):
        """
        Initialize the unitary layer.
        
        Args:
            dim (int): Dimension of the input/output
            complex_input (bool): Whether the input is complex (represented as [real; imag])
            init (str): Initialization method ('identity', 'random', 'fourier')
        """
        super(UnitaryLayer, self).__init__()
        
        self.dim = dim
        self.complex_input = complex_input
        
        # For complex inputs, we need a complex-valued matrix
        if complex_input:
            # Initialize parameters for a complex matrix
            if init == 'identity':
                real_part = np.eye(dim)
                imag_part = np.zeros((dim, dim))
            elif init == 'random':
                # Random initialization that's close to unitary
                X = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
                # QR decomposition gives a unitary matrix Q
                Q, _ = np.linalg.qr(X)
                real_part = np.real(Q)
                imag_part = np.imag(Q)
            elif init == 'fourier':
                dft_matrix = dft(dim) / np.sqrt(dim)
                real_part = np.real(dft_matrix)
                imag_part = np.imag(dft_matrix)
            
            # Create parameters for real and imaginary parts
            self.weight_real = nn.Parameter(torch.tensor(real_part, dtype=torch.float32))
            self.weight_imag = nn.Parameter(torch.tensor(imag_part, dtype=torch.float32))
        else:
            # For real inputs, we use a real-valued orthogonal matrix
            if init == 'identity':
                matrix = np.eye(dim)
            elif init == 'random':
                # Random orthogonal matrix
                X = np.random.randn(dim, dim)
                # QR decomposition gives an orthogonal matrix Q
                Q, _ = np.linalg.qr(X)
                matrix = Q
            elif init == 'fourier':
                # Use the real part of the DFT matrix (not strictly orthogonal)
                # We'll project it to the orthogonal group
                dft_matrix = dft(dim) / np.sqrt(dim)
                X = np.real(dft_matrix)
                U, _, Vh = np.linalg.svd(X)
                matrix = U @ Vh  # Orthogonal approximation
            
            self.weight = nn.Parameter(torch.tensor(matrix, dtype=torch.float32))
    
    def get_unitary_matrix(self):
        """
        Get the current unitary matrix.
        
        Returns:
            torch.Tensor: The unitary matrix
        """
        if self.complex_input:
            # Construct complex matrix from real and imaginary parts
            return torch.complex(self.weight_real, self.weight_imag)
        else:
            return self.weight
    
    def project_to_unitary(self):
        """
        Project the weights to the unitary/orthogonal group.
        This should be called during training to maintain the unitary constraint.
        """
        with torch.no_grad():
            if self.complex_input:
                # Construct the complex matrix
                W_complex = torch.complex(self.weight_real, self.weight_imag)
                
                # SVD decomposition
                U, _, Vh = torch.linalg.svd(W_complex)
                
                # Reconstruct unitary matrix
                W_unitary = U @ Vh
                
                # Update parameters
                self.weight_real.copy_(W_unitary.real)
                self.weight_imag.copy_(W_unitary.imag)
            else:
                # SVD decomposition
                U, _, Vh = torch.linalg.svd(self.weight)
                
                # Reconstruct orthogonal matrix
                W_orthogonal = U @ Vh
                
                # Update parameter
                self.weight.copy_(W_orthogonal)
    
    def forward(self, x):
        """
        Apply the unitary transformation.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Transformed tensor
        """
        if self.complex_input:
            # Split input into real and imaginary parts
            real_part = x[:, :self.dim]
            imag_part = x[:, self.dim:]
            
            # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            output_real = F.linear(real_part, self.weight_real) - F.linear(imag_part, self.weight_imag)
            output_imag = F.linear(real_part, self.weight_imag) + F.linear(imag_part, self.weight_real)
            
            # Concatenate real and imaginary parts
            output = torch.cat([output_real, output_imag], dim=1)
        else:
            # Simple linear transformation with orthogonal matrix
            output = F.linear(x, self.weight)
        
        return output


class MeasurementBasisChange(nn.Module):
    """
    Neural network module that implements measurement basis changes
    using unitary transformations.
    """
    def __init__(self, dim, n_bases=3, complex_input=True):
        """
        Initialize the measurement basis change module.
        
        Args:
            dim (int): Dimension of the input/output
            n_bases (int): Number of different measurement bases
            complex_input (bool): Whether the input is complex
        """
        super(MeasurementBasisChange, self).__init__()
        
        self.dim = dim
        self.n_bases = n_bases
        self.complex_input = complex_input
        
        # Create a set of unitary transformations for different bases
        self.basis_transforms = nn.ModuleList([
            UnitaryLayer(dim, complex_input=complex_input, 
                        init='random' if i > 0 else 'identity')
            for i in range(n_bases)
        ])
        
        # Initialize the Fourier transform layer
        self.fourier_layer = FourierTransformLayer(dim, learnable=False)
    
    def forward(self, x, basis_idx=0):
        """
        Apply the measurement basis change.
        
        Args:
            x (torch.Tensor): Input tensor
            basis_idx (int): Index of the measurement basis to use
            
        Returns:
            torch.Tensor: Transformed tensor in the selected basis
        """
        # Apply the selected basis transformation
        basis_idx = basis_idx % self.n_bases
        return self.basis_transforms[basis_idx](x)
    
    def measure_in_basis(self, x, basis_idx=0):
        """
        Measure the input in the selected basis.
        
        Args:
            x (torch.Tensor): Input tensor (wavefunction)
            basis_idx (int): Index of the measurement basis
            
        Returns:
            torch.Tensor: Probability distribution in the selected basis
        """
        # Apply basis change
        x_transformed = self.forward(x, basis_idx)
        
        if self.complex_input:
            # Calculate probability distribution
            real_part = x_transformed[:, :self.dim]
            imag_part = x_transformed[:, self.dim:]
            prob = real_part**2 + imag_part**2
        else:
            # For real inputs, just square
            prob = x_transformed**2
            
        # Normalize
        prob = prob / torch.sum(prob, dim=1, keepdim=True)
        
        return prob
    
    def measure_in_fourier_basis(self, x):
        """
        Measure the input in the Fourier basis.
        
        Args:
            x (torch.Tensor): Input tensor (wavefunction)
            
        Returns:
            torch.Tensor: Probability distribution in the Fourier basis
        """
        # Apply Fourier transform
        x_transformed = self.fourier_layer(x)
        
        if self.complex_input:
            # Calculate probability distribution
            real_part = x_transformed[:, :self.dim]
            imag_part = x_transformed[:, self.dim:]
            prob = real_part**2 + imag_part**2
        else:
            # For real inputs, we need to reshape
            # The Fourier transform doubles the dimension for real inputs
            half_dim = x_transformed.shape[1] // 2
            real_part = x_transformed[:, :half_dim]
            imag_part = x_transformed[:, half_dim:]
            prob = real_part**2 + imag_part**2
            
        # Normalize
        prob = prob / torch.sum(prob, dim=1, keepdim=True)
        
        return prob


class FourierRecurrentUnit(nn.Module):
    """
    Fourier Recurrent Unit (FRU) that uses Fourier basis functions
    to summarize hidden states over time.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_fourier_features=8):
        """
        Initialize the Fourier Recurrent Unit.
        
        Args:
            input_dim (int): Dimension of the input
            hidden_dim (int): Dimension of the hidden state
            output_dim (int): Dimension of the output
            n_fourier_features (int): Number of Fourier features
        """
        super(FourierRecurrentUnit, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_fourier_features = n_fourier_features
        
        # Input-to-hidden transformation
        self.input_to_hidden = nn.Linear(input_dim, hidden_dim)
        
        # Hidden-to-hidden transformation
        self.hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim)
        
        # Fourier feature layer
        self.fourier_layer = nn.Linear(hidden_dim, n_fourier_features * 2)
        
        # Output layer
        self.output_layer = nn.Linear(n_fourier_features * 2, output_dim)
        
        # Activation functions
        self.hidden_activation = nn.Tanh()
        
    def forward(self, x, hidden=None):
        """
        Forward pass of the FRU.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            hidden (torch.Tensor): Initial hidden state
            
        Returns:
            tuple: (output, final_hidden)
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        outputs = []
        
        # Process each time step
        for t in range(seq_len):
            # Input at current time step
            x_t = x[:, t, :]
            
            # Update hidden state
            hidden_input = self.input_to_hidden(x_t) + self.hidden_to_hidden(hidden)
            hidden = self.hidden_activation(hidden_input)
            
            # Apply Fourier feature transformation
            fourier_features = self.fourier_layer(hidden)
            
            # Split into sine and cosine components
            sin_features = fourier_features[:, :self.n_fourier_features]
            cos_features = fourier_features[:, self.n_fourier_features:]
            
            # Apply sine and cosine activations
            sin_features = torch.sin(sin_features)
            cos_features = torch.cos(cos_features)
            
            # Combine features
            combined_features = torch.cat([sin_features, cos_features], dim=1)
            
            # Generate output
            output = self.output_layer(combined_features)
            outputs.append(output)
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=1)
        
        return outputs, hidden


def visualize_basis_changes(input_dim=16, n_bases=4):
    """
    Visualize different measurement bases for a simple quantum state.
    
    Args:
        input_dim (int): Dimension of the quantum state
        n_bases (int): Number of measurement bases to visualize
    """
    # Create a simple quantum state (Gaussian wavepacket)
    x = np.linspace(-5, 5, input_dim)
    psi = np.exp(-(x - 1.0)**2 / 2.0)
    psi = psi / np.sqrt(np.sum(np.abs(psi)**2))
    
    # Convert to complex representation
    psi_complex = np.zeros(input_dim * 2)
    psi_complex[:input_dim] = psi  # Real part
    
    # Convert to tensor
    psi_tensor = torch.tensor(psi_complex, dtype=torch.float32).unsqueeze(0)
    
    # Create measurement basis change module
    mbc = MeasurementBasisChange(input_dim, n_bases=n_bases, complex_input=True)
    
    # Measure in different bases
    probs = []
    for i in range(n_bases):
        prob = mbc.measure_in_basis(psi_tensor, i).squeeze().detach().numpy()
        probs.append(prob)
    
    # Measure in Fourier basis
    prob_fourier = mbc.measure_in_fourier_basis(psi_tensor).squeeze().detach().numpy()
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    # Plot original state
    plt.subplot(n_bases + 2, 1, 1)
    plt.plot(x, psi**2)
    plt.title('Original State (Position Basis)')
    plt.xlabel('Position')
    plt.ylabel('Probability')
    
    # Plot Fourier basis
    plt.subplot(n_bases + 2, 1, 2)
    plt.plot(np.arange(input_dim), prob_fourier[:input_dim])
    plt.title('Fourier Basis (Momentum Basis)')
    plt.xlabel('Momentum')
    plt.ylabel('Probability')
    
    # Plot other bases
    for i in range(n_bases):
        plt.subplot(n_bases + 2, 1, i + 3)
        plt.plot(np.arange(input_dim), probs[i][:input_dim])
        plt.title(f'Measurement Basis {i}')
        plt.xlabel('Basis Index')
        plt.ylabel('Probability')
    
    plt.tight_layout()
    plt.savefig('measurement_bases_visualization.png')
    plt.close()
    
    return mbc


def main():
    """Test the unitary transformations module."""
    # Visualize basis changes
    mbc = visualize_basis_changes(input_dim=32, n_bases=4)
    print("Basis changes visualization saved to 'measurement_bases_visualization.png'")
    
    # Test Fourier Recurrent Unit
    batch_size = 10
    seq_len = 20
    input_dim = 5
    hidden_dim = 16
    output_dim = 3
    
    # Create random input
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Create FRU
    fru = FourierRecurrentUnit(input_dim, hidden_dim, output_dim)
    
    # Forward pass
    outputs, hidden = fru(x)
    
    print(f"FRU input shape: {x.shape}")
    print(f"FRU output shape: {outputs.shape}")
    print(f"FRU hidden shape: {hidden.shape}")
    
    # Test unitary layer with projection
    dim = 10
    ul = UnitaryLayer(dim, complex_input=False)
    
    # Check orthogonality before projection
    W = ul.get_unitary_matrix()
    ortho_error = torch.norm(W.T @ W - torch.eye(dim))
    print(f"Orthogonality error before projection: {ortho_error.item():.6f}")
    
    # Apply projection
    ul.project_to_unitary()
    
    # Check orthogonality after projection
    W = ul.get_unitary_matrix()
    ortho_error = torch.norm(W.T @ W - torch.eye(dim))
    print(f"Orthogonality error after projection: {ortho_error.item():.6f}")


if __name__ == "__main__":
    main()
