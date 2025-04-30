"""
Quantum Harmonic Oscillator Visualization

This script creates detailed visualizations of the quantum harmonic oscillator
dynamics, showing the time evolution of position, momentum, and energy.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from quantum_harmonic_oscillator import QuantumHarmonicOscillator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def plot_detailed_evolution(qho, t_points, psi_t, save_path='quantum_detailed_evolution.png'):
    """
    Create a detailed visualization of the quantum harmonic oscillator evolution.
    
    Args:
        qho: QuantumHarmonicOscillator instance
        t_points: Time points
        psi_t: Evolved wavefunctions
        save_path: Path to save the figure
    """
    # Calculate position and momentum expectation values
    q_expect = np.array([qho.position_expectation(psi) for psi in psi_t])
    p_expect = np.array([qho.momentum_expectation(psi) for psi in psi_t])
    
    # Calculate energy expectation values
    energy_expect = np.array([qho.energy_expectation(psi) for psi in psi_t])
    
    # Calculate position and momentum variances
    q_var = np.array([np.sum(np.abs(psi)**2 * (qho.x - q)**2) * qho.dx 
                      for psi, q in zip(psi_t, q_expect)])
    
    # Calculate momentum variances using the momentum wavefunction
    p_var = []
    for psi, p in zip(psi_t, p_expect):
        psi_k = qho.momentum_wavefunction(psi)
        dk = qho.k[1] - qho.k[0]
        p_variance = np.sum(np.abs(psi_k)**2 * (qho.k - p)**2) * dk
        p_var.append(p_variance)
    p_var = np.array(p_var)
    
    # Calculate uncertainty product
    uncertainty_product = np.sqrt(q_var * np.array(p_var))
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 3)
    
    # 1. Wavefunction evolution (heatmap)
    ax1 = fig.add_subplot(gs[0, :])
    im = ax1.imshow(np.abs(psi_t)**2, aspect='auto', 
                   extent=[qho.x[0], qho.x[-1], 0, t_points[-1]],
                   cmap='viridis', interpolation='nearest')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Time')
    ax1.set_title('Wavefunction Evolution (|ψ|²)')
    plt.colorbar(im, ax=ax1, label='Probability Density')
    
    # 2. Position expectation value over time
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t_points, q_expect, 'r-')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Position Expectation ⟨x⟩')
    ax2.set_title('Position vs Time')
    ax2.grid(True)
    
    # 3. Momentum expectation value over time
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t_points, p_expect, 'b-')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Momentum Expectation ⟨p⟩')
    ax3.set_title('Momentum vs Time')
    ax3.grid(True)
    
    # 4. Energy expectation value over time
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(t_points, energy_expect, 'g-')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Energy Expectation ⟨E⟩')
    ax4.set_title('Energy vs Time')
    ax4.grid(True)
    
    # 5. Phase space trajectory
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(q_expect, p_expect, 'k-')
    ax5.scatter(q_expect[0], p_expect[0], color='red', s=50, label='Start')
    ax5.scatter(q_expect[-1], p_expect[-1], color='blue', s=50, label='End')
    ax5.set_xlabel('Position ⟨x⟩')
    ax5.set_ylabel('Momentum ⟨p⟩')
    ax5.set_title('Phase Space Trajectory')
    ax5.grid(True)
    ax5.legend()
    
    # 6. Uncertainty product over time
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(t_points, uncertainty_product, 'm-')
    ax6.axhline(y=0.5, color='k', linestyle='--', label='HUP Minimum (ħ/2)')
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Δx·Δp')
    ax6.set_title('Uncertainty Product vs Time')
    ax6.grid(True)
    ax6.legend()
    
    # 7. Wavefunction at different times
    ax7 = fig.add_subplot(gs[2, 2])
    times_to_plot = [0, len(t_points)//4, len(t_points)//2, 3*len(t_points)//4, -1]
    for i, t_idx in enumerate(times_to_plot):
        ax7.plot(qho.x, np.abs(psi_t[t_idx])**2, 
                label=f't = {t_points[t_idx]:.1f}')
    ax7.set_xlabel('Position')
    ax7.set_ylabel('Probability Density |ψ|²')
    ax7.set_title('Wavefunction at Different Times')
    ax7.grid(True)
    ax7.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return fig


def create_3d_wavefunction_animation(qho, t_points, psi_t, save_path='wavefunction_animation.gif'):
    """
    Create a 3D animation of the wavefunction evolution.
    
    Args:
        qho: QuantumHarmonicOscillator instance
        t_points: Time points
        psi_t: Evolved wavefunctions
        save_path: Path to save the animation
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Prepare the plot
    line, = ax.plot([], [], [], 'r-', lw=2)
    point, = ax.plot([], [], [], 'bo', ms=6)
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
    
    # Set up the axes
    ax.set_xlim(qho.x_range)
    ax.set_ylim(0, 1.5*np.max(np.abs(psi_t)**2))
    ax.set_zlim(0, t_points[-1])
    
    ax.set_xlabel('Position')
    ax.set_ylabel('Probability Density |ψ|²')
    ax.set_zlabel('Time')
    ax.set_title('Quantum Harmonic Oscillator Wavefunction Evolution')
    
    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        time_text.set_text('')
        return line, point, time_text
    
    def animate(i):
        # Plot the wavefunction at time t_points[i]
        x = qho.x
        y = np.abs(psi_t[i])**2
        z = t_points[i] * np.ones_like(x)
        
        line.set_data(x, y)
        line.set_3d_properties(z)
        
        # Add a point at the expectation value
        x_expect = qho.position_expectation(psi_t[i])
        y_expect = np.abs(psi_t[i][np.abs(qho.x - x_expect).argmin()])**2
        
        point.set_data([x_expect], [y_expect])
        point.set_3d_properties([t_points[i]])
        
        time_text.set_text(f'Time: {t_points[i]:.2f}')
        
        return line, point, time_text
    
    # Create animation
    ani = FuncAnimation(fig, animate, frames=len(t_points),
                         init_func=init, blit=True, interval=50)
    
    # Save animation
    ani.save(save_path, writer='pillow', fps=15)
    plt.close()
    
    return ani


def plot_energy_components(qho, t_points, psi_t, save_path='energy_components.png'):
    """
    Plot the kinetic, potential, and total energy components over time.
    
    Args:
        qho: QuantumHarmonicOscillator instance
        t_points: Time points
        psi_t: Evolved wavefunctions
        save_path: Path to save the figure
    """
    # Calculate energy components
    kinetic_energy = []
    potential_energy = []
    total_energy = []
    
    for psi in psi_t:
        # Kinetic energy using momentum space
        psi_k = qho.momentum_wavefunction(psi)
        T = 0.5 * np.sum(np.abs(psi_k)**2 * qho.k**2) * (qho.k[1] - qho.k[0])
        
        # Potential energy
        V = 0.5 * qho.omega**2 * np.sum(np.abs(psi)**2 * qho.x**2) * qho.dx
        
        kinetic_energy.append(T)
        potential_energy.append(V)
        total_energy.append(T + V)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot energy components
    ax1.plot(t_points, kinetic_energy, 'r-', label='Kinetic Energy')
    ax1.plot(t_points, potential_energy, 'g-', label='Potential Energy')
    ax1.plot(t_points, total_energy, 'b-', label='Total Energy')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Energy')
    ax1.set_title('Energy Components vs Time')
    ax1.grid(True)
    ax1.legend()
    
    # Plot energy ratio
    ax2.plot(t_points, np.array(kinetic_energy) / np.array(potential_energy), 'k-')
    ax2.axhline(y=1.0, color='r', linestyle='--', label='Equal KE and PE')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Kinetic Energy / Potential Energy')
    ax2.set_title('Virial Theorem Ratio')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return fig


def create_phase_space_density_plot(qho, t_points, psi_t, save_path='phase_space_density.png'):
    """
    Create a phase space density plot using the Wigner function.
    
    Args:
        qho: QuantumHarmonicOscillator instance
        t_points: Time points
        psi_t: Evolved wavefunctions
        save_path: Path to save the figure
    """
    # We'll compute the Wigner function for a few selected times
    selected_times = [0, len(t_points)//4, len(t_points)//2, 3*len(t_points)//4, -1]
    
    # Create a grid for phase space
    x_grid = qho.x
    p_grid = qho.k
    
    # Create figure
    fig, axes = plt.subplots(1, len(selected_times), figsize=(20, 4))
    
    for i, t_idx in enumerate(selected_times):
        # Compute Wigner function (simplified version)
        wigner = np.zeros((len(x_grid), len(p_grid)))
        
        for x_idx, x in enumerate(x_grid):
            for p_idx, p in enumerate(p_grid):
                # Simplified Wigner function calculation
                # For a proper calculation, we would need to compute the full integral
                psi_x = psi_t[t_idx]
                wigner[x_idx, p_idx] = np.abs(psi_x[x_idx])**2 * np.exp(-p**2)
        
        # Plot the Wigner function
        im = axes[i].imshow(wigner, extent=[p_grid[0], p_grid[-1], x_grid[0], x_grid[-1]], 
                           origin='lower', aspect='auto', cmap='viridis')
        
        # Add expectation value point
        q_expect = qho.position_expectation(psi_t[t_idx])
        p_expect = qho.momentum_expectation(psi_t[t_idx])
        axes[i].plot(p_expect, q_expect, 'ro', ms=5)
        
        axes[i].set_title(f't = {t_points[t_idx]:.1f}')
        axes[i].set_xlabel('Momentum')
        axes[i].set_ylabel('Position')
    
    plt.colorbar(im, ax=axes[-1], label='Phase Space Density')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return fig


def main():
    """Run the visualization script."""
    print("Creating detailed quantum harmonic oscillator visualizations...")
    
    # Create the quantum harmonic oscillator
    qho = QuantumHarmonicOscillator(n_points=256, x_range=(-10, 10), omega=1.0)
    
    # Create different initial states
    
    # 1. Ground state
    psi_ground = qho.initial_state(x0=0.0, sigma=1.0/np.sqrt(2))
    
    # 2. Coherent state (displaced ground state)
    psi_coherent = qho.initial_state(x0=2.0, sigma=1.0/np.sqrt(2))
    
    # 3. Superposition of ground and first excited state
    psi_0 = qho.initial_state(x0=0.0, sigma=1.0/np.sqrt(2))
    psi_1 = qho.initial_state(x0=0.0, sigma=1.0/np.sqrt(2)) * qho.x
    psi_1 = psi_1 / np.sqrt(np.sum(np.abs(psi_1)**2) * qho.dx)
    psi_superposition = (psi_0 + psi_1) / np.sqrt(2)
    psi_superposition = psi_superposition / np.sqrt(np.sum(np.abs(psi_superposition)**2) * qho.dx)
    
    # 4. Coherent state with momentum
    k0 = 2.0  # Initial momentum
    psi_moving = np.exp(-(qho.x - 0.0)**2 / (2 * (1.0/np.sqrt(2))**2)) * np.exp(1j * k0 * qho.x)
    psi_moving = psi_moving / np.sqrt(np.sum(np.abs(psi_moving)**2) * qho.dx)
    
    # Evolve each state
    t_max = 10.0
    n_steps = 100
    t_points = np.linspace(0, t_max, n_steps)
    
    # Evolve the states
    _, psi_ground_t = qho.evolve(psi_ground, t_max=t_max, n_steps=n_steps)
    _, psi_coherent_t = qho.evolve(psi_coherent, t_max=t_max, n_steps=n_steps)
    _, psi_superposition_t = qho.evolve(psi_superposition, t_max=t_max, n_steps=n_steps)
    _, psi_moving_t = qho.evolve(psi_moving, t_max=t_max, n_steps=n_steps)
    
    # Create visualizations for each state
    plot_detailed_evolution(qho, t_points, psi_ground_t, 'ground_state_evolution.png')
    plot_detailed_evolution(qho, t_points, psi_coherent_t, 'coherent_state_evolution.png')
    plot_detailed_evolution(qho, t_points, psi_superposition_t, 'superposition_evolution.png')
    plot_detailed_evolution(qho, t_points, psi_moving_t, 'moving_state_evolution.png')
    
    # Create energy component plots
    plot_energy_components(qho, t_points, psi_ground_t, 'ground_state_energy.png')
    plot_energy_components(qho, t_points, psi_coherent_t, 'coherent_state_energy.png')
    plot_energy_components(qho, t_points, psi_moving_t, 'moving_state_energy.png')
    
    # Create phase space density plots
    create_phase_space_density_plot(qho, t_points, psi_coherent_t, 'coherent_state_phase_space.png')
    create_phase_space_density_plot(qho, t_points, psi_moving_t, 'moving_state_phase_space.png')
    
    # Create 3D animation for the moving state (this can take some time)
    # Uncomment if you want to create the animation
    # create_3d_wavefunction_animation(qho, t_points, psi_moving_t, 'moving_state_animation.gif')
    
    print("Visualizations complete!")
    print("Generated files:")
    print("  - ground_state_evolution.png")
    print("  - coherent_state_evolution.png")
    print("  - superposition_evolution.png")
    print("  - moving_state_evolution.png")
    print("  - ground_state_energy.png")
    print("  - coherent_state_energy.png")
    print("  - moving_state_energy.png")
    print("  - coherent_state_phase_space.png")
    print("  - moving_state_phase_space.png")


if __name__ == "__main__":
    main()
