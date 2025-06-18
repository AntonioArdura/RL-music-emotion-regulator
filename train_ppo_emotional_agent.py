import os
import time  # To add a timestamp to the saved models

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# Make sure UserEmotionalSimulator and SIM_CONFIG are importable
# If they are in user_emotional_simulator.py, you can import them like this:
from user_emotional_simulator import SIM_CONFIG, UserEmotionalSimulator

# EMOTION_VA_MAP and EMOTION_NAMES are not strictly necessary here if 
# UserEmotionalSimulator handles them internally for its reset.

# --- Directories for logs and models (create these folders in your project if they don't exist) ---
LOG_DIR = "ppo_tensorboard_logs/"
MODEL_DIR = "ppo_trained_models/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Training Configuration ---
# You can adjust these values. Start with a lower number to test the pipeline.
TOTAL_TRAINING_TIMESTEPS = 200000  # Example: 200k steps for an initial test run
EVAL_FREQ = 10000                 # Evaluate the agent every N agent steps
N_EVAL_EPISODES = 10              # Number of episodes to run for each evaluation
CHECKPOINT_FREQ = 25000           # Save a model checkpoint every N steps

if __name__ == '__main__':
    print("--- Starting environment and PPO agent setup ---")

    # 1. Create the simulated environment and wrap it
    # Stable Baselines3 expects a "Vectorized Environment"
    # DummyVecEnv is the simplest way to do this for a single environment
    print("Creating and wrapping the simulator environment...")
    env = DummyVecEnv([lambda: UserEmotionalSimulator(config=SIM_CONFIG)])
    print("Environment created and wrapped.")

    # 2. Callbacks (Optional but very useful)
    print("Configuring Callbacks...")
    # Periodically evaluate the agent and save the best model
    eval_callback = EvalCallback(
        env, # Evaluation environment (can be the same as the training one to start)
        best_model_save_path=os.path.join(MODEL_DIR, "best_model"), # Saves the best model here
        log_path=os.path.join(LOG_DIR, "eval_logs"),       # Saves the evaluation logs here
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True, # Use deterministic actions for evaluation
        render=False        # Do not render during evaluation
    )

    # Save model checkpoints periodically
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=os.path.join(MODEL_DIR, f"checkpoints_{timestamp}"),
        name_prefix="ppo_emotional_agent_checkpoint"
    )
    print("Callbacks configured.")

    # 3. Define the PPO agent
    # Review the PPO hyperparameters in the Stable Baselines3 documentation
    # and tune them according to your problem's complexity and computational resources.
    # The ones we defined in SIM_CONFIG were for the simulator, these are for the AGENT.
    print("Defining the PPO model...")
    model = PPO(
        "MlpPolicy",        # Policy based on a Multi-Layer Perceptron
        env,
        verbose=1,          # Prints training progress information (0=none, 1=info, 2=debug)
        learning_rate=3e-4, # Learning rate (common value for PPO)
        n_steps=2048,       # Number of steps to run for each environment per policy update
        batch_size=64,      # Minibatch size for updates
        n_epochs=10,        # Number of epochs when optimizing the surrogate loss
        gamma=0.99,         # Discount factor for future rewards
        gae_lambda=0.95,    # Factor for Generalized Advantage Estimation
        clip_range=0.2,     # PPO's clipping parameter (stabilizes training)
        tensorboard_log=LOG_DIR, # Directory for TensorBoard logs
        # Network architecture for the actor (pi) and critic (vf)
        # Two hidden layers with 128 and then 64 neurons is a good starting point
        policy_kwargs=dict(net_arch=dict(pi=[128, 64], vf=[128, 64]))
    )
    print("PPO model defined.")

    # 4. Train the model
    print(f"\n--- Starting PPO Training for {TOTAL_TRAINING_TIMESTEPS} timesteps ---")
    try:
        model.learn(
            total_timesteps=TOTAL_TRAINING_TIMESTEPS,
            callback=[eval_callback, checkpoint_callback], # List of callbacks
            progress_bar=True       # Shows a progress bar in the console
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
    finally:
        # Save the final model regardless of interruption or error
        final_model_path = os.path.join(MODEL_DIR, f"ppo_emotional_agent_final_{timestamp}_{TOTAL_TRAINING_TIMESTEPS}steps")
        print(f"\nSaving final model to: {final_model_path}")
        model.save(final_model_path)
        print(f"The best model (if a new best was found during evaluation) was saved in: {os.path.join(MODEL_DIR, 'best_model.zip')}")

    print("\n--- Training finished ---")
    env.close()
    print("Environment closed.")