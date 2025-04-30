"""
Quantum Harmonic Oscillator Simulation and Neural Network Models

This script demonstrates the complete workflow:
1. Simulating a 1D quantum harmonic oscillator
2. Training a Hamiltonian Neural Network to emulate its evolution
3. Training a Lagrangian Neural Network for improved energy conservation
4. Predicting observables using the trained models
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import os

from core.quantum_harmonic_oscillator import QuantumHarmonicOscillator
from models.hamiltonian_neural_network import HamiltonianNeuralNetwork
from models.lagrangian_neural_network import LagrangianNeuralNetwork, prepare_training_data_from_qho
from core.measurement_module import QuantumMeasurementSystem

# Create directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('images/neural_networks', exist_ok=True)
os.makedirs('images/visualizations', exist_ok=True)
os.makedirs('images/comparisons', exist_ok=True)

def main():
    print("=" * 80)
    print("Quantum Harmonic Oscillator Simulation and Neural Network Models")
    print("=" * 80)
    
    # Step 1: Simulate the quantum harmonic oscillator
    print("\nStep 1: Simulating the quantum harmonic oscillator...")
    qho = QuantumHarmonicOscillator(n_points=256, x_range=(-10, 10), omega=1.0)
    
    # Create an initial state
    psi_0 = qho.initial_state(x0=-2.0, sigma=0.5)
    
    # Evolve the state
    t_points, psi_t = qho.evolve(psi_0, t_max=10.0, n_steps=100)
    
    # Plot the evolution
    qho.plot_evolution(t_points, psi_t)
    print("Quantum evolution complete. Visualization saved to 'images/visualizations/quantum_harmonic_oscillator_evolution.png'")
    
    # Step 2: Generate training data for the Neural Networks
    print("\nStep 2: Generating training data for the Neural Networks...")
    qp_data, derivatives = qho.generate_training_data(n_initial_states=10, n_steps=100)
    
    print(f"Generated {len(qp_data)} training samples")
    print(f"Input shape (q,p): {qp_data.shape}")
    print(f"Output shape (dq/dt, dp/dt): {derivatives.shape}")
    
    # Save the training data
    np.savez('data/harmonic_oscillator_data.npz', 
             qp=qp_data, 
             derivatives=derivatives)
    
    # Step 3: Train the Hamiltonian Neural Network
    print("\nStep 3: Training the Hamiltonian Neural Network...")
    hnn = HamiltonianNeuralNetwork(hidden_dim=64, learning_rate=1e-3)
    
    # Train the model
    start_time = time.time()
    hnn_losses = hnn.train(qp_data, derivatives, batch_size=32, epochs=1000, print_every=100)
    training_time = time.time() - start_time
    
    print(f"HNN training complete in {training_time:.2f} seconds")
    
    # Plot the training loss
    plt.figure(figsize=(10, 6))
    plt.plot(hnn_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('HNN Training Loss')
    plt.yscale('log')
    plt.savefig('images/neural_networks/hnn_training_loss.png')
    plt.close()
    
    # Step 4: Train the Lagrangian Neural Network for better energy conservation
    print("\nStep 4: Training the Lagrangian Neural Network for improved energy conservation...")
    
    # Prepare data for LNN
    q, q_dot, q_ddot = prepare_training_data_from_qho(qp_data, derivatives)
    
    # Create and train the Lagrangian Neural Network
    lnn = LagrangianNeuralNetwork(hidden_dim=64, learning_rate=1e-3)
    
    # Train the model
    start_time = time.time()
    lnn_losses = lnn.train(q, q_dot, q_ddot, batch_size=32, epochs=1000, print_every=100)
    training_time = time.time() - start_time
    
    print(f"LNN training complete in {training_time:.2f} seconds")
    
    # Plot the training loss
    plt.figure(figsize=(10, 6))
    plt.plot(lnn_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('LNN Training Loss')
    plt.yscale('log')
    plt.savefig('images/neural_networks/lnn_training_loss.png')
    plt.close()
    
    # Step 5: Evaluate the Neural Networks
    print("\nStep 5: Evaluating the Neural Networks...")
    
    # Create a new initial state
    psi_0_test = qho.initial_state(x0=0.0, sigma=0.5)
    
    # Evolve the state
    t_points_test, psi_t_test = qho.evolve(psi_0_test, t_max=10.0, n_steps=100)
    
    # Calculate position and momentum expectation values
    q_true = np.array([qho.position_expectation(psi) for psi in psi_t_test])
    p_true = np.array([qho.momentum_expectation(psi) for psi in psi_t_test])
    
    # Compare the true and predicted trajectories for HNN
    q0 = q_true[0]
    p0 = p_true[0]
    hnn.plot_trajectory_comparison(q0, p0, (0, 10), q_true, p_true, steps=100)
    print("HNN evaluation complete. Comparison plot saved to 'images/neural_networks/hnn_trajectory_comparison.png'")
    
    # Compare the true and predicted trajectories for LNN
    lnn.plot_trajectory_comparison(q0, p0, (0, 10), q_true, p_true, steps=100)
    print("LNN evaluation complete. Comparison plot saved to 'images/neural_networks/lnn_trajectory_comparison.png'")
    
    # Step 6: Compare energy conservation between HNN and LNN
    print("\nStep 6: Comparing energy conservation between HNN and LNN...")
    
    # Predict trajectories
    t_hnn, q_hnn, p_hnn = hnn.predict_trajectory(q0, p0, (0, 10), steps=100)
    t_lnn, q_lnn, q_dot_lnn, energy_lnn = lnn.predict_trajectory(q0, p0, (0, 10), steps=100)
    
    # Calculate energies
    energy_true = 0.5 * p_true**2 + 0.5 * q_true**2
    energy_hnn = 0.5 * p_hnn**2 + 0.5 * q_hnn**2
    
    # Plot energy comparison
    plt.figure(figsize=(10, 6))
    plt.plot(t_hnn, energy_hnn, 'b-', label='HNN Predicted')
    plt.plot(t_lnn, energy_lnn, 'g-', label='LNN Predicted')
    plt.plot(np.linspace(0, 10, len(energy_true)), energy_true, 'r--', label='True')
    plt.xlabel('Time')
    plt.ylabel('Energy (H)')
    plt.legend()
    plt.title('Energy Conservation Comparison')
    plt.savefig('images/comparisons/energy_conservation_comparison.png')
    plt.close()
    
    print("Energy conservation comparison complete. Plot saved to 'images/comparisons/energy_conservation_comparison.png'")
    
    # Step 7: Train the Observable Predictor
    print("\nStep 7: Training the Observable Predictor...")
    
    # Generate different initial states and evolve them for training the measurement system
    n_states = 5
    all_wavefunctions = []
    
    for i in range(n_states):
        # Vary the initial state parameters
        x0 = np.random.uniform(-2, 2)
        sigma = np.random.uniform(0.3, 1.0)
        
        # Create and evolve the initial state
        psi_0_i = qho.initial_state(x0=x0, sigma=sigma)
        t_points_i, psi_t_i = qho.evolve(psi_0_i, t_max=10.0, n_steps=50)
        
        # Add to collection
        all_wavefunctions.append(psi_t_i)
    
    # Flatten the collection of wavefunctions
    all_wavefunctions = np.vstack(all_wavefunctions)
    print(f"Generated {len(all_wavefunctions)} wavefunctions for training the measurement system")
    
    # Create and train the quantum measurement system
    qms = QuantumMeasurementSystem(n_points=256, x_range=(-10, 10))
    
    # Generate training data
    latent_vars, observables = qms.generate_training_data(all_wavefunctions)
    print(f"Generated training data: {latent_vars.shape}, {observables.shape}")
    
    # Train the observable predictor
    start_time = time.time()
    losses = qms.train(latent_vars, observables, batch_size=32, epochs=500, print_every=50)
    training_time = time.time() - start_time
    
    print(f"Observable predictor training complete in {training_time:.2f} seconds")
    
    # Plot the training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Observable Predictor Training Loss')
    plt.yscale('log')
    plt.savefig('images/neural_networks/observable_predictor_training_loss.png')
    plt.close()
    
    # Save models
    torch.save(hnn.model.state_dict(), 'models/hamiltonian_nn_model.pt')
    torch.save(lnn.model.state_dict(), 'models/lagrangian_nn_model.pt')
    torch.save(qms.predictor.state_dict(), 'models/observable_predictor_model.pt')
    
    print("Models saved to the 'models' directory")
    
    print("\nAll steps completed successfully!")

if __name__ == "__main__":
    main()
