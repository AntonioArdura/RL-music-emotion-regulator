import os
import random
import time

import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from stable_baselines3 import PPO

# Import the simulator environment class from your other file
from user_emotional_simulator import (EMOTION_VA_MAP, SIM_CONFIG,
                                      UserEmotionalSimulator)

# --- CONFIGURATION ---
MODEL_PATH = os.path.join("ppo_trained_models", "best_model", "best_model.zip")

# --- SPOTIFY API CREDENTIALS ---
# Make sure these are correct and match your Spotify Developer Dashboard
SPOTIPY_CLIENT_ID='5a8e257a3dac4135932503545b86deb1'
SPOTIPY_CLIENT_SECRET='888c409317774bfd9bb2f0d5ebe8a715'
SPOTIPY_REDIRECT_URI='http://127.0.0.1:8888/callback'

def setup_spotify_client():
    """ Sets up and authenticates the Spotipy client. """
    try:
        # Pide permisos para leer las canciones más escuchadas del usuario
        scope = "user-top-read"
        auth_manager = SpotifyOAuth(
            client_id=SPOTIPY_CLIENT_ID,
            client_secret=SPOTIPY_CLIENT_SECRET,
            redirect_uri=SPOTIPY_REDIRECT_URI,
            scope=scope,
            open_browser=False # Avoids opening a new browser tab if a token is already cached
        )
        sp = spotipy.Spotify(auth_manager=auth_manager)
        
        # TOKEN TEST: Get and print the current user's display name to confirm it works
        user_info = sp.me()
        print(f"✅ Spotify client authenticated successfully for user: {user_info['display_name']}")
        return sp
        
    except Exception as e:
        print(f"\n--- Spotify Authentication Error --- \nError details: {e}\n-------------------------------------\n")
        return None

# --- NEW ROBUST VERSION OF THE FUNCTION ---
def get_song_from_spotify(agent_action_vector, sp_client, seed_info={"genres": ["pop", "rock", "chill", "acoustic", "electronic", "dance"]}):
    """
    Takes the agent's action vector and robustly retrieves a matching song from Spotify
    using the correct API format and a multi-step fallback system.
    """
    if not sp_client:
        print("--> Spotify client not available. Skipping song retrieval.")
        return None

    # 1. De-normalize the agent's action & convert to Python native types
    target_v = float((agent_action_vector[0] + 1.0) / 2.0)
    target_a = float((agent_action_vector[1] + 1.0) / 2.0)
    target_t_0_1 = float((agent_action_vector[2] + 1.0) / 2.0)
    target_m_0_1 = float((agent_action_vector[3] + 1.0) / 2.0)
    target_d = float((agent_action_vector[4] + 1.0) / 2.0)
    
    min_bpm, max_bpm = 50, 200
    target_t_bpm = float(min_bpm + target_t_0_1 * (max_bpm - min_bpm))
    target_m_binary = 1 if target_m_0_1 > 0.5 else 0

    print(f"--> Agent wants profile: V={target_v:.2f}, E={target_a:.2f}, T={target_t_bpm:.0f}, M={target_m_binary}, D={target_d:.2f}")

    # 2. Prepare the seed parameters for spotipy
    seed_artists = seed_info.get('artist_ids', [])
    seed_tracks = seed_info.get('track_ids', [])
    seed_genres = seed_info.get('genres', [])

    recs = None
    try:
        # --- PLAN A: SPECIFIC SEARCH ---
        # print("--> Attempting specific search (Plan A)...")
        recs = sp_client.recommendations(
            seed_artists=seed_artists,
            seed_genres=seed_genres,
            seed_tracks=seed_tracks,
            limit=10, 
            target_valence=target_v,
            target_energy=target_a, 
            target_tempo=target_t_bpm,
            target_mode=target_m_binary,
            target_danceability=target_d
        )
        if recs and recs['tracks']:
            print("--> Success with specific search (Plan A).")
            track = recs['tracks'][0]
            return {"name": track['name'], "artist": track['artists'][0]['name'], "id": track['id'], "url": track['external_urls']['spotify']}

        # --- PLAN B: BROAD SEARCH (V-A + SEEDS) ---
        print("--> Specific search failed. Trying broader search (Plan B)...")
        recs = sp_client.recommendations(
            seed_artists=seed_artists,
            seed_genres=seed_genres,
            seed_tracks=seed_tracks,
            limit=10, 
            target_valence=target_v, 
            target_energy=target_a
        )
        if recs and recs['tracks']:
            print("--> Success with broad search (Plan B).")
            track = recs['tracks'][0]
            return {"name": track['name'], "artist": track['artists'][0]['name'], "id": track['id'], "url": track['external_urls']['spotify']}
        
        # --- PLAN C: SAFEST SEARCH (SEEDS ONLY) ---
        print("--> Broad search failed. Trying safest search (Plan C)...")
        recs = sp_client.recommendations(
            seed_artists=seed_artists,
            seed_genres=seed_genres,
            seed_tracks=seed_tracks,
            limit=10
        )
        if recs and recs['tracks']:
            print("--> Success with safest search (Plan C).")
            track = recs['tracks'][0]
            return {"name": track['name'], "artist": track['artists'][0]['name'], "id": track['id'], "url": track['external_urls']['spotify']}

        print("--> All searches failed. No recommendations found.")
        return None

    except Exception as e:
        print(f"--> An unexpected error occurred while fetching from Spotify: {e}")
        return None


def test_trained_agent(model_path, start_emotion, target_emotion, sp_client):
    """
    Loads a trained PPO model and runs one episode, searching for real songs.
    """
    print(f"\n--- Testing Transition: {start_emotion.upper()} -> {target_emotion.upper()} ---")
    
    try:
        model = PPO.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model not found at {model_path}.")
        return

    # Get user's top tracks to use as recommendation seeds
    top_tracks = sp_client.current_user_top_tracks(limit=1, time_range='short_term')
    seed_info_for_test = {'genres': ['pop', 'rock']} # Fallback if no top tracks
    if top_tracks and top_tracks['items']:
        top_track = top_tracks['items'][0]
        # Use both the track and its artist as seeds for better results
        seed_info_for_test = {'track_ids': [top_track['id']], 'artist_ids': [top_track['artists'][0]['id']]}
        print(f"Using user's top track and artist as seeds: '{top_track['name']}' by {top_track['artists'][0]['name']}")
    else:
        print("Could not get user's top tracks, using default seed genres.")
    
    env = UserEmotionalSimulator(config=SIM_CONFIG)
    obs, info = env.reset()

    env.current_V, env.current_A = EMOTION_VA_MAP[start_emotion]
    env.target_V, env.target_A = EMOTION_VA_MAP[target_emotion]
    env.initial_distance_to_target = env._calculate_distance(env.current_V, env.current_A, env.target_V, env.target_A)
    obs = env._get_observation()
    
    print("Initial State:")
    env.render()

    for i in range(SIM_CONFIG["MAX_STEPS_PER_EPISODE"]):
        action, _states = model.predict(obs, deterministic=True)
        
        print(f"\nStep {i+1} -> Agent's Action (Target Musical Profile):")
        # --- CORRECTION APPLIED HERE ---
        # The variable was 'action_d' which is not defined. 
        # It should be action[4] to access the 5th element (Danceability).
        print(f"  V={action[0]:.2f}, A={action[1]:.2f}, T={action[2]:.2f}, M={action[3]:.2f}, D={action[4]:.2f}")
        
        recommended_song = get_song_from_spotify(action, sp_client, seed_info=seed_info_for_test)
        if recommended_song:
            print(f"  Spotify Suggestion: '{recommended_song['name']}' by {recommended_song['artist']}")
            print(f"  URL: {recommended_song['url']}")
        else:
            print("  No song suggestion was found for this step.")

        obs, reward, terminated, truncated, info = env.step(action)
        
        print("\nResulting State:")
        env.render()

        if terminated or truncated:
            print("\nEpisode finished.")
            if terminated:
                print("SUCCESS: Target emotion reached within the simulator!")
            if truncated:
                print("NOTE: Maximum steps reached.")
            break
            
    env.close()


if __name__ == '__main__':
    spotify_client = setup_spotify_client()
    
    if spotify_client:
        if not os.path.exists(MODEL_PATH):
            print("="*50)
            print(f"FATAL ERROR: The trained model was not found at '{MODEL_PATH}'")
            print("Please make sure you have run the training script and that a 'best_model.zip' was successfully saved.")
            print("="*50)
        else:
            # --- RUN TESTS ---
            test_trained_agent(MODEL_PATH, start_emotion="Sad", target_emotion="Happy", sp_client=spotify_client)
            test_trained_agent(MODEL_PATH, start_emotion="Tense", target_emotion="Calm", sp_client=spotify_client)
            test_trained_agent(MODEL_PATH, start_emotion="Happy", target_emotion="Content", sp_client=spotify_client)
