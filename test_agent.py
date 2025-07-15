import os
import time

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# Import your simulator
from user_emotional_simulator import (EMOTION_VA_MAP, SIM_CONFIG,
                                      UserEmotionalSimulator)

# --- CONFIGURATION ---
MODEL_PATH = os.path.join("ppo_trained_models", "best_model", "best_model.zip")
# --- CAMBIO 1: Apuntar al nuevo archivo CSV ---
# Asegúrate de que el nombre del archivo del nuevo dataset sea este, o ajústalo.
SONG_DATASET_PATH = "tracks_features.csv" 

# --- FUNCIÓN CORREGIDA ---
def load_song_dataset(path):
    """ Loads the song dataset from the new CSV file. """
    print(f"Loading song dataset from {path}...")
    try:
        # --- CAMBIO 2: Nombres de las columnas actualizados ---
        # El nuevo dataset usa 'artists' y 'name' en lugar de 'artist_name' y 'track_name'.
        # También, el modo ya viene como un número (0 o 1), no como 'Major'/'Minor'.
        columns_to_read = ['artists', 'name', 'valence', 'energy', 'tempo', 'mode', 'danceability']
        df = pd.read_csv(path, usecols=columns_to_read)
        
        # Limpieza de datos: eliminar filas con valores nulos en las columnas que nos interesan
        df.dropna(subset=columns_to_read, inplace=True)
        # Limpieza de datos: el nombre del artista a veces es una lista en formato string, lo limpiamos
        # y nos quedamos solo con el primer artista para simplificar.
        df['artists'] = df['artists'].apply(lambda x: x.strip("[]").strip("'").split("',")[0].strip("'"))

        library = df.to_dict('records')
        print(f"✅ Dataset loaded successfully with {len(library)} songs.")
        return library
    except FileNotFoundError:
        print(f"❌ ERROR: Song dataset file not found at '{path}'.")
        print("   Please download 'tracks.csv' from the Kaggle dataset and place it in your project folder.")
        return None
    except Exception as e:
        print(f"❌ ERROR: Could not load or parse song dataset. Details: {e}")
        return None

# --- FUNCIÓN CORREGIDA ---
def get_song_from_dataset(agent_action_vector, library):
    """
    Finds the best matching song from the local library based on the agent's action.
    """
    if not library:
        print("--> Song library is not available. Skipping song retrieval.")
        return None

    # 1. De-normalize the agent's action to target values
    target_v = float((agent_action_vector[0] + 1.0) / 2.0)
    target_a = float((agent_action_vector[1] + 1.0) / 2.0)
    target_t_0_1 = float((agent_action_vector[2] + 1.0) / 2.0)
    target_m_0_1 = float((agent_action_vector[3] + 1.0) / 2.0)
    target_d = float((agent_action_vector[4] + 1.0) / 2.0)
    
    min_bpm, max_bpm = 50, 200
    target_t_bpm = float(min_bpm + target_t_0_1 * (max_bpm - min_bpm))
    target_m_binary = 1 if target_m_0_1 > 0.5 else 0

    print(f"--> Agent wants profile: V={target_v:.2f}, E={target_a:.2f}, T={target_t_bpm:.0f}, M={target_m_binary}, D={target_d:.2f}")

    best_song = None
    min_distance = float('inf')

    target_t_norm = (target_t_bpm - min_bpm) / (max_bpm - min_bpm)
    target_vector = np.array([target_v, target_a, target_t_norm, float(target_m_binary), target_d])

    # 2. Search for the closest song in the library
    for song in library:
        song_tempo_norm = (song['tempo'] - min_bpm) / (max_bpm - min_bpm)
        
        # --- CAMBIO 3: El modo ya es un número (0 o 1) en este dataset ---
        # Ya no necesitamos convertir de 'Major'/'Minor' a float.
        song_mode_float = float(song['mode'])
        
        song_vector = np.array([
            song['valence'], song['energy'], song_tempo_norm,
            song_mode_float,
            song['danceability']
        ])
        
        distance = np.linalg.norm(target_vector - song_vector)
        
        if distance < min_distance:
            min_distance = distance
            best_song = song

    return best_song

def test_trained_agent(model_path, start_emotion, target_emotion, library):
    """
    Loads a trained PPO model and runs one episode, finding songs from the local library.
    """
    print(f"\n--- Testing Transition: {start_emotion.upper()} -> {target_emotion.upper()} ---")
    
    try:
        model = PPO.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model not found at {model_path}.")
        return

    env = UserEmotionalSimulator(config=SIM_CONFIG)
    obs, info = env.reset()

    env.current_V, env.current_A = EMOTION_VA_MAP[start_emotion]
    env.target_V, env.target_A = EMOTION_VA_MAP[target_emotion]
    obs = env._get_observation()
    
    print("Initial State:")
    env.render()

    for i in range(SIM_CONFIG["MAX_STEPS_PER_EPISODE"]):
        action, _states = model.predict(obs, deterministic=True)
        
        print(f"\nStep {i+1} -> Agent's Action (Target Musical Profile):")
        print(f"  V={action[0]:.2f}, A={action[1]:.2f}, T={action[2]:.2f}, M={action[3]:.2f}, D={action[4]:.2f}")
        
        recommended_song = get_song_from_dataset(action, library)
        if recommended_song:
            # --- CAMBIO 4: Usar los nuevos nombres de columna para imprimir ---
            print(f"  Dataset Suggestion: '{recommended_song['name']}' by {recommended_song['artists']}")
        else:
            print("  No song suggestion was found for this step.")

        obs, reward, terminated, truncated, info = env.step(action)
        
        print("\nResulting State:")
        env.render()

        if terminated or truncated:
            print("\nEpisode finished.")
            if terminated: print("SUCCESS: Target emotion reached within the simulator!")
            if truncated: print("NOTE: Maximum steps reached.")
            break
            
    env.close()


if __name__ == '__main__':
    song_library = load_song_dataset(SONG_DATASET_PATH)
    
    if song_library:
        if not os.path.exists(MODEL_PATH):
            print("="*50)
            print(f"FATAL ERROR: The trained model was not found at '{MODEL_PATH}'")
        else:
            test_trained_agent(MODEL_PATH, start_emotion="Sad", target_emotion="Happy", library=song_library)
            test_trained_agent(MODEL_PATH, start_emotion="Tense", target_emotion="Calm", library=song_library)
