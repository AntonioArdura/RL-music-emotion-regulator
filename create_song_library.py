import json
import os

import spotipy
from spotipy.oauth2 import SpotifyOAuth

# --- YOUR CREDENTIALS ---
CLIENT_ID     = "5a8e257a3dac4135932503545b86deb1"
CLIENT_SECRET = "888c409317774bfd9bb2f0d5ebe8a715"
REDIRECT_URI  = "http://127.0.0.1:8888/callback"
# Scope needed to read a user's private and public playlists
SCOPE         = "playlist-read-private" 

OUTPUT_FILE = "song_library.json"
# How many of your playlists to use for building the library
# Change this number if you want to use more or fewer of your playlists
PLAYLIST_LIMIT = 10 

def setup_spotify_client():
    """ Authenticates the Spotipy client. """
    try:
        auth_manager = SpotifyOAuth(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            redirect_uri=REDIRECT_URI,
            scope=SCOPE,
            open_browser=False
        )
        sp = spotipy.Spotify(auth_manager=auth_manager)
        user_info = sp.me()
        print(f"✅ Authenticated successfully for user: {user_info['display_name']}")
        return sp
    except Exception as e:
        print(f"❌ Authentication Error: {e}")
        return None

def get_user_playlist_ids(sp, limit=PLAYLIST_LIMIT):
    """ Fetches the IDs of the current user's playlists. """
    playlists = sp.current_user_playlists(limit=limit)
    playlist_ids = []
    if playlists and playlists['items']:
        for playlist in playlists['items']:
            playlist_ids.append(playlist['id'])
            print(f"  -> Found playlist: '{playlist['name']}'")
    return playlist_ids

def get_playlist_track_ids(sp, playlist_id):
    """ Fetches all track IDs from a given playlist. """
    track_ids = []
    try:
        results = sp.playlist_tracks(playlist_id)
        while results:
            for item in results['items']:
                if item and item['track'] and item['track']['id']:
                    track_ids.append(item['track']['id'])
            if results['next']:
                results = sp.next(results)
            else:
                results = None
    except Exception as e:
        print(f"    -> Warning: Could not fetch tracks for playlist {playlist_id}. It might be empty or inaccessible. Error: {e}")
    return track_ids

def get_audio_features_in_batches(sp, track_ids):
    """ Fetches audio features for a list of track IDs in batches of 100. """
    all_features = []
    for i in range(0, len(track_ids), 100):
        batch = track_ids[i:i+100]
        try:
            features = sp.audio_features(batch)
            all_features.extend(f for f in features if f) # Filter out None results
        except Exception as e:
            print(f"    -> Warning: Could not fetch audio features for a batch. Error: {e}")
    return all_features

def main():
    """ Main script to build the song library from the user's own playlists. """
    sp = setup_spotify_client()
    if not sp:
        return

    print(f"\nFetching up to {PLAYLIST_LIMIT} of your personal playlists...")
    user_playlist_ids = get_user_playlist_ids(sp)
    
    if not user_playlist_ids:
        print("❌ No playlists found for your account. Please create a playlist on Spotify and add songs to it.")
        return

    print(f"\nFetching all unique tracks from your {len(user_playlist_ids)} playlists...")
    all_track_ids = set() # Use a set to avoid duplicate tracks
    for pl_id in user_playlist_ids:
        ids = get_playlist_track_ids(sp, pl_id)
        all_track_ids.update(ids)
    
    track_ids_list = list(all_track_ids)
    print(f"Found {len(track_ids_list)} unique tracks.")

    if not track_ids_list:
        print("❌ No tracks found in the selected playlists.")
        return

    print("\nFetching audio features for all tracks...")
    audio_features = get_audio_features_in_batches(sp, track_ids_list)
    print(f"Successfully fetched features for {len(audio_features)} tracks.")

    print("\nFetching track names and artists to build the library...")
    song_library = []
    for i in range(0, len(audio_features), 50):
        batch_features = audio_features[i:i+50]
        batch_ids = [f['id'] for f in batch_features]
        try:
            track_info_results = sp.tracks(batch_ids)
            for features, track_info in zip(batch_features, track_info_results['tracks']):
                if track_info and track_info['artists']:
                    song_library.append({
                        'id': features['id'],
                        'name': track_info['name'],
                        'artist': track_info['artists'][0]['name'],
                        'valence': features['valence'],
                        'energy': features['energy'],
                        'tempo': features['tempo'],
                        'mode': features['mode'],
                        'danceability': features['danceability']
                    })
        except Exception as e:
            print(f"    -> Warning: Could not fetch track info for a batch. Error: {e}")


    print(f"\nSaving library with {len(song_library)} songs to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(song_library, f, indent=4, ensure_ascii=False)
    
    print("✅ Done! Your personal song library is ready.")

if __name__ == "__main__":
    main()
