import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# --- Configuration ---
SIM_CONFIG = {
    "N_ACTION_DIMS": 5,  # V_song, A_song, Tempo_song, Mode_song, Danceability_song
    "OBS_DIMS": 9,       # V_curr, A_curr, V_targ, A_targ, H_V, H_A, PlayRatio_prev, Skip_prev, Feedback_prev
    "MAX_STEPS_PER_EPISODE": 30,
    "TARGET_REACHED_THRESHOLD": 0.15,  # Euclidean distance in V-A space for success
    
    # --- Simulator Dynamics Parameters ---
    "EMOTIONAL_SUSCEPTIBILITY_BETA": 0.3, 
    "NOISE_SIGMA_V_TRANSITION": 0.05,     
    "NOISE_SIGMA_A_TRANSITION": 0.05,     
    "ALPHA_EWMA_HISTORY": 0.5,            

    # --- Simulated Spotify Song Retrieval Noise ---
    "SPOTIFY_SIM_NOISE_SIGMA_V": 0.02, 
    "SPOTIFY_SIM_NOISE_SIGMA_A": 0.02, 

    # --- Simulated User Interaction Logic Parameters ---
    "SKIP_PROB_BASE": 0.05,
    "SKIP_PROB_SONG_MISMATCH_FACTOR": 0.4, 
    "SKIP_PROB_AWAY_FROM_TARGET_FACTOR": 0.3, 
    "SKIP_MAX_PROB": 0.90, 

    "FEEDBACK_IMPROVEMENT_THRESHOLD_LIKE": 0.05,  
    "FEEDBACK_WORSENING_THRESHOLD_DISLIKE": 0.03, # Note: This is a positive value; worsening is improvement < -threshold
    "FEEDBACK_EXPLICIT_PROB": 0.7, 

    # --- Reward Weights & Constants ---
    "REWARD_WEIGHTS": {
        "w_trans": 0.6, "w_eng": 0.15, "w_feed": 0.25
    },
    "REWARD_CONSTANTS": {
        "R_target_reached_bonus": 5.0, 
        "R_step_penalty": -0.1,         
        "C_skip_penalty": -1.0,
        "C_play_reward_factor": 0.5,    
        "C_like_bonus": 1.5,
        "C_dislike_penalty": -1.5,
        "C_song_profile_discrepancy_penalty_factor": -0.2 
    }
}

# --- Emotion to V-A Mapping (Ensure these are your final, normalized values) ---
EMOTION_VA_MAP = {
    "Happy": np.array([0.80, 0.17], dtype=np.float32),
    "Sad": np.array([-0.75, -0.32], dtype=np.float32),
    "Calm": np.array([0.44, -0.63], dtype=np.float32),
    "Excited": np.array([0.63, 0.69], dtype=np.float32), # From Warriner for "excited"
    "Content": np.array([0.62, -0.32], dtype=np.float32),
    "Serene": np.array([0.58, -0.51], dtype=np.float32), 
    "Relaxed": np.array([0.50, -0.55], dtype=np.float32),
    "Tired": np.array([-0.33, -0.35], dtype=np.float32),
    "Bored": np.array([-0.55, -0.53], dtype=np.float32),
    "Frustrated": np.array([-0.63, 0.45], dtype=np.float32), 
    "Tense": np.array([-0.45, 0.54], dtype=np.float32),
    "Angry": np.array([-0.62, 0.60], dtype=np.float32),
}
EMOTION_NAMES = list(EMOTION_VA_MAP.keys())

class UserEmotionalSimulator(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, config=SIM_CONFIG, emotion_map=EMOTION_VA_MAP):
        super(UserEmotionalSimulator, self).__init__()
        self.config = config
        self.emotion_map = emotion_map
        self.emotion_names = list(emotion_map.keys())

        # Corrected Action Space: Symmetric and normalized to [-1, 1]
        action_dims = self.config["N_ACTION_DIMS"]
        action_low_bounds = np.full(action_dims, -1.0, dtype=np.float32)
        action_high_bounds = np.full(action_dims, 1.0, dtype=np.float32)
        self.action_space = spaces.Box(low=action_low_bounds, high=action_high_bounds, dtype=np.float32)

        obs_dims = self.config["OBS_DIMS"]
        obs_low_bounds = np.full(obs_dims, -1.0, dtype=np.float32)
        obs_high_bounds = np.full(obs_dims, 1.0, dtype=np.float32)
        obs_low_bounds[6] = 0.0 # PlayRatio_prev is [0,1]
        obs_low_bounds[7] = 0.0 # Skip_prev is 0 or 1 (will be float(0) or float(1))
        self.observation_space = spaces.Box(low=obs_low_bounds, high=obs_high_bounds, shape=(obs_dims,), dtype=np.float32)

        self.current_V = 0.0; self.current_A = 0.0
        self.target_V = 0.0; self.target_A = 0.0
        self.history_V_ewma = 0.0; self.history_A_ewma = 0.0
        self.prev_song_play_ratio = 0.0
        self.prev_song_skipped = 0
        self.prev_song_feedback = 0
        self.initial_distance_to_target = 1.0
        self.current_step_count = 0
        print("UserEmotionalSimulator initialized.")

    def _get_random_emotion_va(self):
        emotion_name = random.choice(self.emotion_names)
        return self.emotion_map[emotion_name]

    def _calculate_distance(self, v1, a1, v2, a2):
        return np.sqrt((v1 - v2)**2 + (a1 - a2)**2)

    def _get_observation(self):
        return np.array([
            self.current_V, self.current_A,
            self.target_V, self.target_A,
            self.history_V_ewma, self.history_A_ewma,
            self.prev_song_play_ratio,
            float(self.prev_song_skipped),
            float(self.prev_song_feedback)
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        initial_va = self._get_random_emotion_va()
        self.current_V, self.current_A = initial_va[0], initial_va[1]
        while True:
            target_va = self._get_random_emotion_va()
            if not np.array_equal(target_va, initial_va):
                self.target_V, self.target_A = target_va[0], target_va[1]
                break
        self.history_V_ewma = self.current_V
        self.history_A_ewma = self.current_A
        self.prev_song_play_ratio = 0.0
        self.prev_song_skipped = 0
        self.prev_song_feedback = 0
        self.initial_distance_to_target = self._calculate_distance(self.current_V, self.current_A, self.target_V, self.target_A)
        if self.initial_distance_to_target < 1e-6: self.initial_distance_to_target = 1e-6
        self.current_step_count = 0
        observation = self._get_observation()
        info = {"initial_V_A": (self.current_V, self.current_A), "target_V_A": (self.target_V, self.target_A)}
        return observation, info

    def step(self, action_vector_neg1_1): # Renamed to indicate it's from the agent in [-1,1]
        # print(f"\n--- Inside step() for step {self.current_step_count + 1} ---")
        # print(f"Received raw action_vector: {action_vector_neg1_1}")

        # De-normalize action components that were originally [0,1]
        # V and A from agent are already [-1,1]
        requested_song_V = action_vector_neg1_1[0]
        requested_song_A = action_vector_neg1_1[1]
        # Tempo, Mode, Danceability were scaled by agent to [-1,1], convert back to [0,1] for internal logic
        requested_song_Tempo_0_1 = (action_vector_neg1_1[2] + 1.0) / 2.0
        requested_song_Mode_0_1 = (action_vector_neg1_1[3] + 1.0) / 2.0 # Will interpret >0.5 as Major later
        requested_song_Danceability_0_1 = (action_vector_neg1_1[4] + 1.0) / 2.0
        
        # print(f"Interpreted action: V={requested_song_V:.2f}, A={requested_song_A:.2f}, T={requested_song_Tempo_0_1:.2f}, M={requested_song_Mode_0_1:.2f}, D={requested_song_Danceability_0_1:.2f}")


        # 1. Simulate Song Characteristics based on de-normalized agent's action
        sim_song_V = np.clip(requested_song_V + np.random.normal(0, self.config["SPOTIFY_SIM_NOISE_SIGMA_V"]), -1.0, 1.0)
        sim_song_A = np.clip(requested_song_A + np.random.normal(0, self.config["SPOTIFY_SIM_NOISE_SIGMA_A"]), -1.0, 1.0)
        # print(f"Requested V-A: ({requested_song_V:.2f}, {requested_song_A:.2f}) -> Sim Song V-A: ({sim_song_V:.2f}, {sim_song_A:.2f})")

        # 2. Simulate emotional transition
        prev_V, prev_A = self.current_V, self.current_A
        self.current_V = np.clip(prev_V + self.config["EMOTIONAL_SUSCEPTIBILITY_BETA"] * (sim_song_V - prev_V) +
                                 np.random.normal(0, self.config["NOISE_SIGMA_V_TRANSITION"]), -1.0, 1.0)
        self.current_A = np.clip(prev_A + self.config["EMOTIONAL_SUSCEPTIBILITY_BETA"] * (sim_song_A - prev_A) +
                                 np.random.normal(0, self.config["NOISE_SIGMA_A_TRANSITION"]), -1.0, 1.0)
        # print(f"Emotional Transition: Prev V-A: ({prev_V:.2f}, {prev_A:.2f}) -> New V-A: ({self.current_V:.2f}, {self.current_A:.2f})")
        
        self.history_V_ewma = self.config["ALPHA_EWMA_HISTORY"] * self.current_V + (1 - self.config["ALPHA_EWMA_HISTORY"]) * self.history_V_ewma
        self.history_A_ewma = self.config["ALPHA_EWMA_HISTORY"] * self.current_A + (1 - self.config["ALPHA_EWMA_HISTORY"]) * self.history_A_ewma

        # 3. Simulate user interaction
        prob_skip = self.config["SKIP_PROB_BASE"]
        dist_song_to_current_va = self._calculate_distance(sim_song_V, sim_song_A, prev_V, prev_A)
        prob_skip_increase_mismatch = dist_song_to_current_va * self.config["SKIP_PROB_SONG_MISMATCH_FACTOR"]
        prob_skip += prob_skip_increase_mismatch
        
        dist_sim_song_to_target = self._calculate_distance(sim_song_V, sim_song_A, self.target_V, self.target_A)
        dist_prev_user_state_to_target = self._calculate_distance(prev_V, prev_A, self.target_V, self.target_A)
        prob_skip_increase_away = 0.0
        if dist_sim_song_to_target > dist_prev_user_state_to_target:
             prob_skip_increase_away = self.config["SKIP_PROB_AWAY_FROM_TARGET_FACTOR"]
             prob_skip += prob_skip_increase_away
        
        sim_skipped = 1 if np.random.rand() < np.clip(prob_skip, 0, self.config["SKIP_MAX_PROB"]) else 0
        sim_play_ratio = 0.1 if sim_skipped else np.random.uniform(0.75, 1.0)
        # print(f"Skip Logic: BaseProb={self.config['SKIP_PROB_BASE']:.2f}, IncMismatch={prob_skip_increase_mismatch:.2f}, IncAway={prob_skip_increase_away:.2f} => TotalProbSkip={prob_skip:.2f} -> Skipped={sim_skipped}, PlayRatio={sim_play_ratio:.2f}")

        sim_feedback = 0
        improvement_in_dist = dist_prev_user_state_to_target - self._calculate_distance(self.current_V, self.current_A, self.target_V, self.target_A)
        # print(f"Feedback Logic: PrevDistToTarget={dist_prev_user_state_to_target:.2f}, CurrDistToTarget={self._calculate_distance(self.current_V, self.current_A, self.target_V, self.target_A):.2f} => Improvement={improvement_in_dist:.2f}")
        if np.random.rand() < self.config["FEEDBACK_EXPLICIT_PROB"]:
            if improvement_in_dist > self.config["FEEDBACK_IMPROVEMENT_THRESHOLD_LIKE"]:
                sim_feedback = 1
            elif improvement_in_dist < -self.config["FEEDBACK_WORSENING_THRESHOLD_DISLIKE"]: 
                sim_feedback = -1
        # print(f"Simulated Feedback: {sim_feedback} (Thresholds: Like>{self.config['FEEDBACK_IMPROVEMENT_THRESHOLD_LIKE']:.2f}, Dislike<-{self.config['FEEDBACK_WORSENING_THRESHOLD_DISLIKE']:.2f})")
            
        # 4. Calculate Reward
        R_trans = 0.0
        if self.initial_distance_to_target > 1e-5:
            R_trans = improvement_in_dist / self.initial_distance_to_target
        
        R_eng = self.config["REWARD_CONSTANTS"]["C_skip_penalty"] * sim_skipped + \
                self.config["REWARD_CONSTANTS"]["C_play_reward_factor"] * (1-sim_skipped) * sim_play_ratio
                
        R_feed = 0.0
        if sim_feedback == 1: R_feed = self.config["REWARD_CONSTANTS"]["C_like_bonus"]
        elif sim_feedback == -1: R_feed = self.config["REWARD_CONSTANTS"]["C_dislike_penalty"]
        
        R_step = self.config["REWARD_CONSTANTS"]["R_step_penalty"]

        R_profile_discrepancy = self.config["REWARD_CONSTANTS"]["C_song_profile_discrepancy_penalty_factor"] * \
                                self._calculate_distance(requested_song_V, requested_song_A, sim_song_V, sim_song_A)

        reward_value = (self.config["REWARD_WEIGHTS"]["w_trans"] * R_trans +
                        self.config["REWARD_WEIGHTS"]["w_eng"] * R_eng +
                        self.config["REWARD_WEIGHTS"]["w_feed"] * R_feed +
                        R_step +
                        R_profile_discrepancy)
        
        current_reward_float = float(reward_value) # Ensure it's a Python float

        # print(f"Reward Components: R_trans={R_trans:.2f}*({self.config['REWARD_WEIGHTS']['w_trans']}), R_eng={R_eng:.2f}*({self.config['REWARD_WEIGHTS']['w_eng']}), R_feed={R_feed:.2f}*({self.config['REWARD_WEIGHTS']['w_feed']}), R_step={R_step:.2f}, R_profile_disc={R_profile_discrepancy:.2f}")
        # print(f"Total Reward (before target bonus): {current_reward_float:.3f}")

        # 5. Update state for next observation & step count
        self.prev_song_play_ratio = sim_play_ratio
        self.prev_song_skipped = sim_skipped
        self.prev_song_feedback = sim_feedback
        self.current_step_count += 1

        # 6. Determine termination conditions
        current_dist_final = self._calculate_distance(self.current_V, self.current_A, self.target_V, self.target_A)
        
        terminated_condition = current_dist_final < self.config["TARGET_REACHED_THRESHOLD"]
        is_terminated = bool(terminated_condition) # Python bool

        if is_terminated:
             current_reward_float += float(self.config["REWARD_CONSTANTS"]["R_target_reached_bonus"])
             # print(f"TARGET REACHED! Bonus of {self.config['REWARD_CONSTANTS']['R_target_reached_bonus']:.2f} applied. Final Reward: {current_reward_float:.3f}")

        truncated_condition = self.current_step_count >= self.config["MAX_STEPS_PER_EPISODE"]
        is_truncated = bool(truncated_condition) # Python bool
        
        observation = self._get_observation()
        info = {"distance_to_target": current_dist_final, "is_success": is_terminated, 
                "action_taken_norm": action_vector_neg1_1, # Raw action from agent
                "sim_song_VA": (sim_song_V, sim_song_A)}
        
        # print(f"--- End of step {self.current_step_count} ---")
        return observation, current_reward_float, is_terminated, is_truncated, info

    def render(self):
        print(f"Step: {self.current_step_count:2d} | Curr:({self.current_V:6.2f},{self.current_A:6.2f})| Targ:({self.target_V:6.2f},{self.target_A:6.2f})| Dist: {self._calculate_distance(self.current_V, self.current_A, self.target_V, self.target_A):.3f} | Last Skip: {self.prev_song_skipped}, Feed: {self.prev_song_feedback}")

    def close(self):
        print("Simulator closed.")

if __name__ == '__main__':
    print("--- Testing UserEmotionalSimulator ---")
    env = UserEmotionalSimulator(config=SIM_CONFIG, emotion_map=EMOTION_VA_MAP)
    try:
        from stable_baselines3.common.env_checker import check_env
        print("\nChecking environment with Stable Baselines3 check_env...")
        check_env(env, warn=True, skip_render_check=True) 
        print("Environment check passed (or warnings issued)!")
    except ImportError:
        print("Stable Baselines3 not found, skipping check_env. Install with: pip install stable-baselines3[extra]")
    except Exception as e:
        print(f"Environment check failed: {e}")

    print("\n--- Running a sample episode with random actions (verbose prints in step() are commented out by default) ---")
    # To see detailed step prints, uncomment them inside the step() method
    obs, info = env.reset()
    env.render()
    total_ep_reward = 0
    for i in range(SIM_CONFIG["MAX_STEPS_PER_EPISODE"] + 2):
        random_action = env.action_space.sample() # Agent's raw action in [-1,1]
        # print(f"\n--- Step {i+1} ---")
        # print(f"Raw Action: {random_action}") # V,A,T,M,D all in [-1,1] if action space is symmetric
        
        obs, reward, terminated, truncated, info = env.step(random_action)
        total_ep_reward += reward
        # env.render() # Render prints current state
        # print(f"Reward for step {i+1}: {reward:.3f}, Total Ep Reward: {total_ep_reward:.3f}")
        # print(f"Info: {info}")

        if terminated or truncated:
            print(f"\nEpisode finished after {i+1} steps. Total Reward: {total_ep_reward:.3f}")
            if terminated: print("TARGET REACHED!")
            if truncated: print("MAX STEPS REACHED.")
            env.render() # Render final state
            break
    env.close()