import json
import os
import time

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# Importa tu simulador y configuraciones
from user_emotional_simulator import (EMOTION_VA_MAP, SIM_CONFIG,
                                      UserEmotionalSimulator)

# --- CONFIGURACIÓN ---
MODEL_PATH = os.path.join("ppo_trained_models", "best_model", "best_model.zip")
# Apuntamos al nuevo archivo CSV descargado de Kaggle
SONG_DATASET_PATH = "tracks_features.csv"
REAL_INTERACTIONS_LOG = "real_interactions_dataset.json"

def load_song_dataset(path):
    """ Carga el dataset de canciones desde el archivo CSV. """
    print(f"Loading song dataset from {path}...")
    try:
        # Nombres de las columnas del nuevo dataset
        columns_to_read = ['artists', 'name', 'valence', 'energy', 'tempo', 'mode', 'danceability']
        df = pd.read_csv(path, usecols=columns_to_read)
        
        # Limpieza de datos
        df.dropna(subset=columns_to_read, inplace=True)
        # El nombre del artista a veces es una lista en formato string, lo limpiamos
        df['artists'] = df['artists'].apply(lambda x: str(x).strip("[]'").split("',")[0].strip("'"))

        library = df.to_dict('records')
        print(f"✅ Dataset loaded successfully with {len(library)} songs.")
        return library
    except FileNotFoundError:
        print(f"❌ ERROR: Song dataset file not found at '{path}'.")
        print("   Please download 'tracks.csv' from Kaggle and place it in your project folder.")
        return None
    except Exception as e:
        print(f"❌ ERROR: Could not load or parse song dataset. Details: {e}")
        return None

def get_song_from_dataset(agent_action_vector, library):
    """
    Encuentra la canción que mejor se ajusta en la librería local.
    """
    if not library:
        return None

    # Desnormaliza la acción del agente
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

    # Busca la canción más cercana en la librería
    for song in library:
        song_tempo_norm = (song['tempo'] - min_bpm) / (max_bpm - min_bpm)
        # El nuevo dataset ya tiene el modo como 0 o 1
        song_mode_float = float(song['mode'])
        
        song_vector = np.array([
            song['valence'], song['energy'], song_tempo_norm,
            song_mode_float, song['danceability']
        ])
        
        distance = np.linalg.norm(target_vector - song_vector)
        if distance < min_distance:
            min_distance, best_song = distance, song

    return best_song

def get_user_feedback():
    """ Pide feedback de skip y like/dislike al usuario. """
    while True:
        skip_input = input("  -> ¿Saltarías esta canción? (s/n): ").lower()
        if skip_input in ['s', 'n']:
            skipped = 1 if skip_input == 's' else 0
            play_ratio = 0.1 if skipped == 1 else 1.0
            break
        else: print("   Respuesta no válida. Por favor, introduce 's' o 'n'.")
    while True:
        feedback_input = input("  -> ¿Te ha gustado? (l: like, d: dislike, n: neutral): ").lower()
        if feedback_input in ['l', 'd', 'n']:
            feedback = 1 if feedback_input == 'l' else -1 if feedback_input == 'd' else 0
            break
        else: print("   Respuesta no válida. Por favor, introduce 'l', 'd', o 'n'.")
    return skipped, play_ratio, feedback

def get_user_progress_rating(start_emotion_name, target_emotion_name):
    """ Pide al usuario que valore su progreso en una escala de 0 a 10. """
    print(f"\nDel 0 al 10, ¿cómo de cerca te sientes de tu objetivo emocional?")
    print(f"  (0 = {start_emotion_name}, 10 = {target_emotion_name})")
    while True:
        try:
            rating = int(input("Introduce tu puntuación (0-10): "))
            if 0 <= rating <= 10:
                return rating
            else: print("   Error: El número debe estar entre 0 y 10.")
        except ValueError:
            print("   Error: Por favor, introduce un número entero válido.")

def calculate_real_reward(config, improvement, skipped, play_ratio, feedback):
    """ Calcula la recompensa basada en el feedback real del usuario. """
    R_trans = improvement
    R_eng = config["REWARD_CONSTANTS"]["C_skip_penalty"] * skipped + \
            config["REWARD_CONSTANTS"]["C_play_reward_factor"] * (1 - skipped) * play_ratio
    R_feed = 0.0
    if feedback == 1: R_feed = config["REWARD_CONSTANTS"]["C_like_bonus"]
    elif feedback == -1: R_feed = config["REWARD_CONSTANTS"]["C_dislike_penalty"]
    R_step = config["REWARD_CONSTANTS"]["R_step_penalty"]
    reward = (config["REWARD_WEIGHTS"]["w_trans"] * R_trans +
              config["REWARD_WEIGHTS"]["w_eng"] * R_eng +
              config["REWARD_WEIGHTS"]["w_feed"] * R_feed +
              R_step)
    return float(reward)

def run_interactive_session(model, library, start_emotion_name, target_emotion_name):
    """ Ejecuta una sesión interactiva usando la escala de 0-10. """
    print(f"\n--- Iniciando Sesión Interactiva: {start_emotion_name.upper()} -> {target_emotion_name.upper()} ---")
    
    start_V, start_A = EMOTION_VA_MAP[start_emotion_name]
    target_V, target_A = EMOTION_VA_MAP[target_emotion_name]
    
    current_V, current_A = start_V, start_A
    prev_play_ratio, prev_skipped, prev_feedback = 0.0, 0, 0
    history_V_ewma, history_A_ewma = current_V, current_A
    alpha_ewma = SIM_CONFIG.get("ALPHA_EWMA_HISTORY", 0.5)
    session_history = []

    for i in range(SIM_CONFIG["MAX_STEPS_PER_EPISODE"]):
        print(f"\n===== PASO {i+1} =====")
        print(f"Estado Actual: ({current_V:.2f}, {current_A:.2f}) | Objetivo: ({target_V:.2f}, {target_A:.2f})")
        
        obs = np.array([current_V, current_A, target_V, target_A,
                        history_V_ewma, history_A_ewma,
                        prev_play_ratio, float(prev_skipped), float(prev_feedback)], dtype=np.float32)

        action, _ = model.predict(obs, deterministic=True)
        recommended_song = get_song_from_dataset(action, library)
        
        if recommended_song:
            # Usar los nuevos nombres de columna para imprimir
            print(f"\nSugerencia: '{recommended_song['name']}' by {recommended_song['artists']}")
            print("Escucha la canción y responde a las siguientes preguntas...")
            skipped, play_ratio, feedback = get_user_feedback()
        else:
            print("No se encontró canción. Penalizando y continuando...")
            skipped, play_ratio, feedback = 1, 0.0, -1
            time.sleep(2)

        prev_V, prev_A = current_V, current_A
        rating = get_user_progress_rating(start_emotion_name, target_emotion_name)
        progress = rating / 10.0
        
        next_V = start_V + progress * (target_V - start_V)
        next_A = start_A + progress * (target_A - start_A)
        
        dist_prev = np.linalg.norm(np.array([prev_V, prev_A]) - np.array([target_V, target_A]))
        dist_now = np.linalg.norm(np.array([next_V, next_A]) - np.array([target_V, target_A]))
        improvement = dist_prev - dist_now
        
        real_reward = calculate_real_reward(SIM_CONFIG, improvement, skipped, play_ratio, feedback)
        print(f"Recompensa para este paso: {real_reward:.3f}")

        transition = {'state': obs.tolist(), 'action': action.tolist(), 'reward': real_reward,
                      'next_state': [float(next_V), float(next_A), float(target_V), float(target_A), 0,0,0,0,0],
                      'done': bool(rating == 10)}
        session_history.append(transition)

        current_V, current_A = next_V, next_A
        prev_play_ratio, prev_skipped, prev_feedback = play_ratio, skipped, feedback
        history_V_ewma = alpha_ewma * current_V + (1 - alpha_ewma) * history_V_ewma
        history_A_ewma = alpha_ewma * current_A + (1 - alpha_ewma) * history_A_ewma

        if transition['done']:
            print("\n¡Objetivo alcanzado! Sesión finalizada.")
            break
            
    return session_history

def save_interactions(history):
    """ Guarda las interacciones recolectadas a un archivo JSON. """
    all_interactions = []
    if os.path.exists(REAL_INTERACTIONS_LOG):
        with open(REAL_INTERACTIONS_LOG, 'r', encoding='utf-8') as f:
            all_interactions = json.load(f)
    all_interactions.extend(history)
    with open(REAL_INTERACTIONS_LOG, 'w', encoding='utf-8') as f:
        json.dump(all_interactions, f, indent=4)
    print(f"\n✅ {len(history)} nuevas interacciones guardadas en '{REAL_INTERACTIONS_LOG}'.")


if __name__ == '__main__':
    song_library = load_song_dataset(SONG_DATASET_PATH)
    
    if song_library:
        if not os.path.exists(MODEL_PATH):
            print(f"FATAL ERROR: El modelo entrenado no se encontró en '{MODEL_PATH}'")
        else:
            model = PPO.load(MODEL_PATH)
            run_interactive_session(model, song_library, "Sad", "Happy")
            run_interactive_session(model, song_library, "Tense", "Calm")