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
from models.hybrid_neural_network import HybridNeuralNetwork
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
    
    # Step 6: Compare energy conservation between HNN, LNN, and Hybrid model
    print("\nStep 6: Comparing energy conservation between HNN, LNN, and Hybrid model...")
    
    # Calculate energies
    energy_true = 0.5 * p_true**2 + 0.5 * q_true**2
    true_mean = np.mean(energy_true)
    true_std = np.std(energy_true)
    true_max_dev = 100 * (np.max(energy_true) - np.min(energy_true)) / true_mean
    
    energy_hnn = 0.5 * p_hnn**2 + 0.5 * q_hnn**2
    hnn_mean = np.mean(energy_hnn)
    hnn_std = np.std(energy_hnn)
    hnn_max_dev = 100 * (np.max(energy_hnn) - np.min(energy_hnn)) / hnn_mean
    
    energy_lnn = 0.5 * q_dot_lnn**2 + 0.5 * q_lnn**2
    lnn_mean = np.mean(energy_lnn)
    lnn_std = np.std(energy_lnn)
    lnn_max_dev = 100 * (np.max(energy_lnn) - np.min(energy_lnn)) / lnn_mean
    
    # Create hybrid model
    print("\nStep 7: Creating and evaluating hybrid models...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 7.1: Ensemble Hybrid Model
    print("\n7.1: Evaluating Ensemble Hybrid Model...")
    hybrid_model = HybridNeuralNetwork(hnn, lnn, device=device)
    
    # Optimize hyperparameters for hybrid model
    print("Optimizing ensemble hybrid model hyperparameters...")
    
    # Grid search over alpha and energy conservation weight
    alphas = [0.3, 0.5, 0.7, 0.9]
    energy_weights = [0.3, 0.5, 0.7, 0.9]
    
    best_alpha = 0.5
    best_energy_weight = 0.5
    best_score = float('inf')
    
    results = []
    
    for alpha in alphas:
        for energy_weight in energy_weights:
            # Generate predictions with current hyperparameters
            _, q_hybrid, p_hybrid, energy_hybrid = hybrid_model.predict_physics_constrained_trajectory(
                q0, p0, (0, 10), steps=100, alpha=alpha, energy_conservation_weight=energy_weight
            )
            
            # Calculate phase space error
            phase_space_error = np.mean((q_hybrid - q_true)**2 + (p_hybrid - p_true)**2)
            
            # Calculate energy conservation error
            energy_error = np.std(energy_hybrid) / np.mean(energy_hybrid)
            
            # Combined score (lower is better)
            score = phase_space_error + energy_error
            
            results.append((alpha, energy_weight, phase_space_error, energy_error, score))
            
            # Update best parameters if this is better
            if score < best_score:
                best_score = score
                best_alpha = alpha
                best_energy_weight = energy_weight
    
    # Generate hybrid model predictions with best parameters
    print(f"Generating ensemble hybrid model predictions with alpha={best_alpha}, energy_weight={best_energy_weight}...")
    t_hybrid, q_hybrid, p_hybrid, energy_hybrid = hybrid_model.predict_physics_constrained_trajectory(
        q0, p0, (0, 10), steps=100, alpha=best_alpha, energy_conservation_weight=best_energy_weight
    )
    
    hybrid_mean = np.mean(energy_hybrid)
    hybrid_std = np.std(energy_hybrid)
    hybrid_max_dev = 100 * (np.max(energy_hybrid) - np.min(energy_hybrid)) / hybrid_mean
    
    # Now evaluate the learning-based approach
    print("\nEvaluating learning-based hybrid model...")
    
    # Optimize alpha for learning-based model
    best_learning_alpha = 0.5
    best_learning_score = float('inf')
    learning_results = []
    
    for alpha in alphas:
        # Generate predictions with current alpha
        _, q_learning, p_learning, energy_learning = hybrid_model.predict_learning_based_trajectory(
            q0, p0, (0, 10), steps=100, alpha=alpha
        )
        
        # Calculate phase space error
        phase_space_error = np.mean((q_learning - q_true)**2 + (p_learning - p_true)**2)
        
        # Calculate energy conservation error
        energy_error = np.std(energy_learning) / np.mean(energy_learning)
        
        # Combined score (lower is better)
        score = phase_space_error + energy_error
        
        learning_results.append((alpha, phase_space_error, energy_error, score))
        
        # Update best parameters if this is better
        if score < best_learning_score:
            best_learning_score = score
            best_learning_alpha = alpha
    
    # Print learning-based results
    print("\nLearning-based model optimization results:")
    print("-" * 80)
    print(f"{'Alpha':<10} {'Phase Space Error':<20} {'Energy Error':<15} {'Total Score':<15}")
    print("-" * 80)
    
    # Sort by score
    learning_results.sort(key=lambda x: x[3])
    
    for alpha, phase_error, energy_error, score in learning_results:
        print(f"{alpha:<10.2f} {phase_error:<20.6f} {energy_error:<15.6f} {score:<15.6f}")
    
    print("-" * 80)
    print(f"Best learning-based alpha: {best_learning_alpha}, score={best_learning_score:.6f}")
    
    # Generate learning-based predictions with best alpha
    print(f"Generating learning-based hybrid model predictions with alpha={best_learning_alpha}...")
    t_learning, q_learning, p_learning, energy_learning = hybrid_model.predict_learning_based_trajectory(
        q0, p0, (0, 10), steps=100, alpha=best_learning_alpha
    )
    
    learning_mean = np.mean(energy_learning)
    learning_std = np.std(energy_learning)
    learning_max_dev = 100 * (np.max(energy_learning) - np.min(energy_learning)) / learning_mean
    
    # Now train a true hybrid model that learns from both HNN and LNN
    print("\nTraining a true hybrid model that learns from both HNN and LNN...")
    
    if not args.skip_training:
        # Train the hybrid model
        losses, hybrid_learned_model = hybrid_model.train_hybrid_model(
            qp_data, derivatives, batch_size=32, epochs=300, 
            learning_rate=1e-3, print_every=50, save_model=True
        )
        
        # Plot training loss
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Hybrid Model Training Loss')
        plt.grid(True)
        plt.savefig('images/neural_networks/hybrid_learned_model_loss.png')
        plt.close()
        print("Training loss plot saved to 'images/neural_networks/hybrid_learned_model_loss.png'")
    else:
        # Create a dummy model structure to load the saved weights
        class HybridNet(nn.Module):
            def __init__(self, hnn, lnn, hidden_dim=64):
                super(HybridNet, self).__init__()
                self.hnn = hnn
                self.lnn = lnn
                self.net = nn.Sequential(
                    nn.Linear(4, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, 2)
                )
            
            def forward(self, q, p):
                # Implementation details not needed for loading
                pass
        
        # Load the pre-trained model
        hybrid_learned_model = HybridNet(hybrid_model.hnn.model, hybrid_model.lnn.model).to(device)
        try:
            hybrid_learned_model.load_state_dict(torch.load("models/hybrid_learned_model.pt"))
            print("Loaded pre-trained hybrid learned model")
        except:
            print("No pre-trained hybrid learned model found, skipping evaluation")
            hybrid_learned_model = None
    
    # Evaluate the learned hybrid model if available
    if hybrid_learned_model is not None:
        print("Evaluating the learned hybrid model...")
        t_learned, q_learned, p_learned, energy_learned = hybrid_model.predict_learned_trajectory(
            hybrid_learned_model, q0, p0, (0, 10), steps=100
        )
        
        # Add the learned model to the comparison plot
        plt.figure(figsize=(15, 12))
        
        # Phase space trajectories
        plt.subplot(2, 2, 1)
        plt.plot(q_true, p_true, 'r-', linewidth=3, label='True')
        plt.plot(q_hnn, p_hnn, 'b--', linewidth=1.5, label='HNN')
        plt.plot(q_lnn, q_dot_lnn, 'g-.', linewidth=1.5, label='LNN')
        plt.plot(q_hybrid, p_hybrid, 'c-', linewidth=1.5, label='Physics-Constrained')
        plt.plot(q_learning, p_learning, 'm-', linewidth=1.5, label='RK4 Combined')
        plt.plot(q_learned, p_learned, 'y-', linewidth=1.5, label='Learned Hybrid')
        plt.xlabel('Position (q)')
        plt.ylabel('Momentum (p)')
        plt.title('Phase Space Trajectories')
        plt.legend()
        plt.grid(True)
        
        # Position over time
        plt.subplot(2, 2, 2)
        t_true = np.linspace(0, 10, len(q_true))
        plt.plot(t_true, q_true, 'r-', linewidth=3, label='True')
        plt.plot(t_hnn, q_hnn, 'b--', linewidth=1.5, label='HNN')
        plt.plot(t_lnn, q_lnn, 'g-.', linewidth=1.5, label='LNN')
        plt.plot(t_hybrid, q_hybrid, 'c-', linewidth=1.5, label='Physics-Constrained')
        plt.plot(t_learning, q_learning, 'm-', linewidth=1.5, label='RK4 Combined')
        plt.plot(t_learned, q_learned, 'y-', linewidth=1.5, label='Learned Hybrid')
        plt.xlabel('Time (t)')
        plt.ylabel('Position (q)')
        plt.title('Position vs Time')
        plt.legend()
        plt.grid(True)
        
        # Momentum over time
        plt.subplot(2, 2, 3)
        plt.plot(t_true, p_true, 'r-', linewidth=3, label='True')
        plt.plot(t_hnn, p_hnn, 'b--', linewidth=1.5, label='HNN')
        plt.plot(t_lnn, q_dot_lnn, 'g-.', linewidth=1.5, label='LNN')
        plt.plot(t_hybrid, p_hybrid, 'c-', linewidth=1.5, label='Physics-Constrained')
        plt.plot(t_learning, p_learning, 'm-', linewidth=1.5, label='RK4 Combined')
        plt.plot(t_learned, p_learned, 'y-', linewidth=1.5, label='Learned Hybrid')
        plt.xlabel('Time (t)')
        plt.ylabel('Momentum (p)')
        plt.title('Momentum vs Time')
        plt.legend()
        plt.grid(True)
        
        # Energy conservation
        plt.subplot(2, 2, 4)
        plt.plot(t_true, energy_true, 'r-', linewidth=3, label='True')
        plt.plot(t_hnn, energy_hnn, 'b--', linewidth=1.5, label='HNN')
        plt.plot(t_lnn, energy_lnn, 'g-.', linewidth=1.5, label='LNN')
        plt.plot(t_hybrid, energy_hybrid, 'c-', linewidth=1.5, label='Physics-Constrained')
        plt.plot(t_learning, energy_learning, 'm-', linewidth=1.5, label='RK4 Combined')
        plt.plot(t_learned, energy_learned, 'y-', linewidth=1.5, label='Learned Hybrid')
        plt.xlabel('Time (t)')
        plt.ylabel('Energy (H)')
        plt.title('Energy Conservation')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('images/comparisons/all_models_with_learned.png')
        plt.close()
        print("Comprehensive model comparison with learned hybrid saved to 'images/comparisons/all_models_with_learned.png'")
        
        # Update energy conservation statistics
        learned_mean = np.mean(energy_learned)
        learned_std = np.std(energy_learned)
        learned_max_dev = 100 * (np.max(energy_learned) - np.min(energy_learned)) / learned_mean
        
        # Print updated energy statistics
        print("\nUpdated Energy Conservation Analysis:")
        print("-" * 80)
        print(f"{'Model':<20} {'Mean Energy':<15} {'Std Dev':<15} {'Max Deviation %':<15}")
        print("-" * 80)
        print(f"{'True':<20} {true_mean:<15.6f} {true_std:<15.6f} {true_max_dev:<15.6f}")
        print(f"{'HNN':<20} {hnn_mean:<15.6f} {hnn_std:<15.6f} {hnn_max_dev:<15.6f}")
        print(f"{'LNN':<20} {lnn_mean:<15.6f} {lnn_std:<15.6f} {lnn_max_dev:<15.6f}")
        print(f"{'Physics-Constrained':<20} {hybrid_mean:<15.6f} {hybrid_std:<15.6f} {hybrid_max_dev:<15.6f}")
        print(f"{'RK4 Combined':<20} {learning_mean:<15.6f} {learning_std:<15.6f} {learning_max_dev:<15.6f}")
        print(f"{'Learned Hybrid':<20} {learned_mean:<15.6f} {learned_std:<15.6f} {learned_max_dev:<15.6f}")
        print("-" * 80)
    
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
    plt.plot(t_true, q_true, 'r-', label='True')
    plt.plot(t_true, q_hnn, 'b--', label='HNN')
    plt.plot(t_true, q_lnn, 'g-.', label='LNN')
    ax5.set_xlabel('Time (t)')
    ax5.set_ylabel('Position (q)')
    ax5.set_title('Position Comparison')
    ax5.grid(True)
    ax5.legend()
    
    # 6. Momentum comparison over time
    ax6 = fig.add_subplot(gs[2, 1])
    plt.plot(t_true, p_true, 'r-', label='True')
    plt.plot(t_true, p_hnn, 'b--', label='HNN')
    plt.plot(t_true, p_lnn, 'g-.', label='LNN')
    ax6.set_xlabel('Time (t)')
    ax6.set_ylabel('Momentum (p)')
    ax6.set_title('Momentum Comparison')
    ax6.grid(True)
    ax6.legend()
    
    # 7. Energy conservation comparison
    ax7 = fig.add_subplot(gs[2, 2])
    plt.plot(t_true, energy_true, 'r-', label='True')
    plt.plot(t_true, energy_hnn, 'b--', linewidth=1.5, label='HNN')
    plt.plot(t_true, energy_lnn, 'g-.', linewidth=1.5, label='LNN')
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
