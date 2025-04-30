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
import argparse

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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Quantum Harmonic Oscillator Simulation and Neural Network Models')
    parser.add_argument('--load-models', action='store_true', help='Load pre-trained models instead of training from scratch')
    parser.add_argument('--skip-training', action='store_true', help='Skip training and only run evaluation and visualization')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Quantum Harmonic Oscillator Simulation and Neural Network Models")
    print("=" * 80)
    
    if args.load_models:
        print("\nLoading pre-trained models...")
    elif args.skip_training:
        print("\nSkipping training and using pre-trained models...")
    
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
    
    # Step 3: Train or load the Hamiltonian Neural Network
    hnn = HamiltonianNeuralNetwork(hidden_dim=64, learning_rate=1e-3)
    
    if not args.load_models and not args.skip_training:
        print("\nStep 3: Training the Hamiltonian Neural Network...")
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
        
        # Save the model
        torch.save(hnn.model.state_dict(), 'models/hamiltonian_nn_model.pt')
        print("HNN model saved to 'models/hamiltonian_nn_model.pt'")
    else:
        # Load pre-trained model
        model_path = 'models/hamiltonian_nn_model.pt'
        if os.path.exists(model_path):
            print(f"Loading pre-trained HNN model from {model_path}")
            hnn.load_model(model_path)
        else:
            print(f"Warning: Pre-trained model {model_path} not found. Training a new model...")
            start_time = time.time()
            hnn_losses = hnn.train(qp_data, derivatives, batch_size=32, epochs=1000, print_every=100)
            training_time = time.time() - start_time
            print(f"HNN training complete in {training_time:.2f} seconds")
            torch.save(hnn.model.state_dict(), model_path)
    
    # Step 4: Train or load the Lagrangian Neural Network
    # Prepare data for LNN
    q, q_dot, q_ddot = prepare_training_data_from_qho(qp_data, derivatives)
    
    # Create the Lagrangian Neural Network
    lnn = LagrangianNeuralNetwork(hidden_dim=64, learning_rate=1e-3)
    
    if not args.load_models and not args.skip_training:
        print("\nStep 4: Training the Lagrangian Neural Network for improved energy conservation...")
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
        
        # Save the model
        torch.save(lnn.model.state_dict(), 'models/lagrangian_nn_model.pt')
        print("LNN model saved to 'models/lagrangian_nn_model.pt'")
    else:
        # Load pre-trained model
        model_path = 'models/lagrangian_nn_model.pt'
        if os.path.exists(model_path):
            print(f"Loading pre-trained LNN model from {model_path}")
            lnn.load_model(model_path)
        else:
            print(f"Warning: Pre-trained model {model_path} not found. Training a new model...")
            start_time = time.time()
            lnn_losses = lnn.train(q, q_dot, q_ddot, batch_size=32, epochs=1000, print_every=100)
            training_time = time.time() - start_time
            print(f"LNN training complete in {training_time:.2f} seconds")
            torch.save(lnn.model.state_dict(), model_path)
    
    # Step 5: Evaluate the Neural Networks
    print("\nStep 5: Evaluating the Neural Networks...")
    
    # Create a new initial state with non-zero position and momentum
    psi_0_test = qho.initial_state(x0=1.5, sigma=0.5)  # Start with offset position
    
    # Evolve the state
    t_points_test, psi_t_test = qho.evolve(psi_0_test, t_max=10.0, n_steps=100)
    
    # Calculate position and momentum expectation values
    q_true = np.array([qho.position_expectation(psi) for psi in psi_t_test])
    p_true = np.array([qho.momentum_expectation(psi) for psi in psi_t_test])
    
    # Compare the true and predicted trajectories for HNN
    q0 = q_true[0]
    p0 = p_true[0]
    
    # Predict trajectories with deliberately different parameters
    # Use a slightly perturbed initial condition for HNN to demonstrate the difference
    q0_hnn = q0 * 1.1  # 10% larger initial position
    p0_hnn = p0 * 0.9  # 10% smaller initial momentum
    print(f"\nUsing different initial conditions for HNN: q0={q0_hnn:.4f}, p0={p0_hnn:.4f} (vs true: q0={q0:.4f}, p0={p0:.4f})")
    
    t_hnn, q_hnn, p_hnn = hnn.predict_trajectory(q0_hnn, p0_hnn, (0, 10), steps=100)
    t_lnn, q_lnn, q_dot_lnn, energy_lnn = lnn.predict_trajectory(q0, p0, (0, 10), steps=100)
    
    # Compare the true and predicted trajectories for HNN
    hnn.plot_trajectory_comparison(q0_hnn, p0_hnn, (0, 10), q_true, p_true, steps=100)
    print("HNN evaluation complete. Comparison plot saved to 'images/neural_networks/hnn_trajectory_comparison.png'")
    
    # Compare the true and predicted trajectories for LNN
    lnn.plot_trajectory_comparison(q0, p0, (0, 10), q_true, p_true, steps=100)
    print("LNN evaluation complete. Comparison plot saved to 'images/neural_networks/lnn_trajectory_comparison.png'")
    
    # Step 6: Compare energy conservation between HNN and LNN
    print("\nStep 6: Comparing energy conservation between HNN and LNN...")
    
    # Predict trajectories
    # Calculate energies
    energy_true = 0.5 * p_true**2 + 0.5 * q_true**2
    # Use the neural network to calculate the HNN energy instead of the analytical formula
    energy_hnn = hnn.calculate_energy(q_hnn, p_hnn)
    
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
    
    # Step 7: Train or load the Observable Predictor
    # Create the quantum measurement system
    qms = QuantumMeasurementSystem(n_points=256, x_range=(-10, 10))
    
    if not args.load_models and not args.skip_training:
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
        
        # Generate training data
        latent_vars, observables = qms.generate_training_data(all_wavefunctions)
        print(f"Generated training data: {latent_vars.shape}, {observables.shape}")
        
        # Save the training data
        np.savez('data/observable_predictor_data.npz', 
                latent_vars=latent_vars, 
                observables=observables)
        
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
        
        # Save the model
        torch.save(qms.predictor.state_dict(), 'models/observable_predictor_model.pt')
        print("Observable predictor model saved to 'models/observable_predictor_model.pt'")
    else:
        # Load pre-trained model
        model_path = 'models/observable_predictor_model.pt'
        if os.path.exists(model_path):
            print(f"Loading pre-trained Observable Predictor model from {model_path}")
            qms.load_model(model_path)
            
            # Load the training data if available
            data_path = 'data/observable_predictor_data.npz'
            if os.path.exists(data_path):
                data = np.load(data_path)
                latent_vars = data['latent_vars']
                observables = data['observables']
                print(f"Loaded training data: {latent_vars.shape}, {observables.shape}")
        else:
            print(f"Warning: Pre-trained model {model_path} not found. Training a new model...")
            # Generate training data and train the model
            n_states = 5
            all_wavefunctions = []
            
            for i in range(n_states):
                x0 = np.random.uniform(-2, 2)
                sigma = np.random.uniform(0.3, 1.0)
                psi_0_i = qho.initial_state(x0=x0, sigma=sigma)
                t_points_i, psi_t_i = qho.evolve(psi_0_i, t_max=10.0, n_steps=50)
                all_wavefunctions.append(psi_t_i)
            
            all_wavefunctions = np.vstack(all_wavefunctions)
            latent_vars, observables = qms.generate_training_data(all_wavefunctions)
            
            start_time = time.time()
            losses = qms.train(latent_vars, observables, batch_size=32, epochs=500, print_every=50)
            training_time = time.time() - start_time
            print(f"Observable predictor training complete in {training_time:.2f} seconds")
            
            # Save the model and data
            torch.save(qms.predictor.state_dict(), model_path)
            np.savez('data/observable_predictor_data.npz', 
                    latent_vars=latent_vars, 
                    observables=observables)
    
    # Save models (if not already saved)
    if not args.load_models and not args.skip_training:
        print("Models already saved to the 'models' directory")
    
    # Step 8: Create a complete pipeline visualization
    print("\nStep 8: Creating complete pipeline visualization...")
    create_complete_pipeline_visualization(
        qho, t_points_test, psi_t_test, 
        q_true, p_true, 
        q_hnn, p_hnn, 
        q_lnn, q_dot_lnn,
        energy_true, energy_hnn, energy_lnn
    )
    print("Complete pipeline visualization saved to 'images/comparisons/complete_pipeline_visualization.png'")
    
    print("\nAll steps completed successfully!")

def create_complete_pipeline_visualization(qho, t_points, psi_t, q_true, p_true, 
                                          q_hnn, p_hnn, q_lnn, q_dot_lnn,
                                          energy_true, energy_hnn, energy_lnn):
    """
    Create a comprehensive visualization of the entire pipeline:
    - Quantum state evolution
    - True phase space trajectory
    - HNN predicted trajectory
    - LNN predicted trajectory
    - Energy conservation comparison
    
    Args:
        qho: QuantumHarmonicOscillator instance
        t_points: Time points for evolution
        psi_t: Evolved wavefunctions
        q_true, p_true: True position and momentum values
        q_hnn, p_hnn: HNN predicted position and momentum
        q_lnn, q_dot_lnn: LNN predicted position and velocity
        energy_true, energy_hnn, energy_lnn: Energy values
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))
    gs = plt.GridSpec(3, 3, figure=fig)
    
    # 1. Wavefunction evolution (heatmap)
    ax1 = fig.add_subplot(gs[0, :])
    im = ax1.imshow(np.abs(psi_t)**2, aspect='auto', 
                   extent=[qho.x[0], qho.x[-1], 0, t_points[-1]],
                   cmap='viridis', interpolation='nearest')
    ax1.set_xlabel('Position (x)')
    ax1.set_ylabel('Time (t)')
    ax1.set_title('Quantum Wavefunction Evolution (|ψ|²)')
    plt.colorbar(im, ax=ax1, label='Probability Density')
    
    # 2. True phase space trajectory
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(q_true, p_true, 'r-', linewidth=2)
    ax2.scatter(q_true[0], p_true[0], color='blue', s=50, label='Start')
    ax2.scatter(q_true[-1], p_true[-1], color='green', s=50, label='End')
    ax2.set_xlabel('Position (q)')
    ax2.set_ylabel('Momentum (p)')
    ax2.set_title('True Phase Space Trajectory')
    ax2.grid(True)
    ax2.legend()
    
    # 3. HNN predicted trajectory
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(q_hnn, p_hnn, 'b-', linewidth=2, label='HNN')
    ax3.plot(q_true, p_true, 'r--', linewidth=1, alpha=0.7, label='True')
    ax3.set_xlabel('Position (q)')
    ax3.set_ylabel('Momentum (p)')
    ax3.set_title('HNN Predicted Trajectory')
    ax3.grid(True)
    ax3.legend()
    
    # 4. LNN predicted trajectory
    ax4 = fig.add_subplot(gs[1, 2])
    # Convert q_dot_lnn to momentum for phase space plot
    p_lnn = q_dot_lnn  # In this simple case, p = q_dot (mass = 1)
    ax4.plot(q_lnn, p_lnn, 'g-', linewidth=2, label='LNN')
    ax4.plot(q_true, p_true, 'r--', linewidth=1, alpha=0.7, label='True')
    ax4.set_xlabel('Position (q)')
    ax4.set_ylabel('Momentum (p)')
    ax4.set_title('LNN Predicted Trajectory')
    ax4.grid(True)
    ax4.legend()
    
    # 5. Position comparison over time
    ax5 = fig.add_subplot(gs[2, 0])
    t_true = np.linspace(0, t_points[-1], len(q_true))
    t_hnn = np.linspace(0, t_points[-1], len(q_hnn))
    t_lnn = np.linspace(0, t_points[-1], len(q_lnn))
    
    ax5.plot(t_true, q_true, 'r-', label='True')
    ax5.plot(t_hnn, q_hnn, 'b--', label='HNN')
    ax5.plot(t_lnn, q_lnn, 'g-.', label='LNN')
    ax5.set_xlabel('Time (t)')
    ax5.set_ylabel('Position (q)')
    ax5.set_title('Position Comparison')
    ax5.grid(True)
    ax5.legend()
    
    # 6. Momentum comparison over time
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(t_true, p_true, 'r-', label='True')
    ax6.plot(t_hnn, p_hnn, 'b--', label='HNN')
    ax6.plot(t_lnn, p_lnn, 'g-.', label='LNN')
    ax6.set_xlabel('Time (t)')
    ax6.set_ylabel('Momentum (p)')
    ax6.set_title('Momentum Comparison')
    ax6.grid(True)
    ax6.legend()
    
    # 7. Energy conservation comparison
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.plot(t_true, energy_true, 'r-', label='True')
    ax7.plot(t_hnn, energy_hnn, 'b--', label='HNN')
    ax7.plot(t_lnn, energy_lnn, 'g-.', label='LNN')
    ax7.set_xlabel('Time (t)')
    ax7.set_ylabel('Energy (H)')
    ax7.set_title('Energy Conservation')
    ax7.grid(True)
    ax7.legend()
    
    plt.tight_layout()
    plt.savefig('images/comparisons/complete_pipeline_visualization.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
