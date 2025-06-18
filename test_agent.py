import os

import spotipy
from spotipy.oauth2 import SpotifyOAuth

# ————————— CONFIG —————————
CLIENT_ID     = "5a8e257a3dac4135932503545b86deb1"
CLIENT_SECRET = "888c409317774bfd9bb2f0d5ebe8a715"
REDIRECT_URI  = "http://127.0.0.1:8888/callback"
SCOPE         = "user-top-read"

# —————— AUTENTICACIÓN ——————
sp = spotipy.Spotify(
    auth_manager=SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPE,
        open_browser=False,
        show_dialog=True
    )
)
me = sp.me()
print(f"✅ Conectado como {me['display_name']} (country={me.get('country','US')})\n")

# —————— 1) Top-5 TRACKS ——————
print("🎧 Tu Top-5 de las últimas 4 semanas:")
top5 = sp.current_user_top_tracks(limit=5, time_range='short_term')['items']
for i, tr in enumerate(top5, 1):
    print(f" {i}. {tr['name']} — {tr['artists'][0]['name']} (id: {tr['id']})")
print()

# —————— 2) RECOMMENDATIONS ——————
#    Usaremos el #1 de ese Top5 como semilla
seed_id = top5[0]['id']
print(f"🎯 Generando recomendación con seed_tracks='{seed_id}'\n")

# Parámetros de ejemplo para audio-features
params = {
    'seed_tracks': seed_id,      # ¡como STRING!
    'limit': 5,
    'market': me.get('country','US'),
    # si quieres filtrar por audio:
    # 'target_valence': 0.6,
    # 'target_energy': 0.7,
    # 'target_tempo': 120,
    # 'target_mode': 1,
    # 'target_danceability': 0.5,
}

print("→ Llamada a /v1/recommendations")
print("  Params:", params)
try:
    recs = sp.recommendations(**params)['tracks']
    if not recs:
        raise Exception("vacío")
    print("\n✅ Recomendaciones:")
    for t in recs:
        print(f" • {t['name']} — {t['artists'][0]['name']}")
except Exception as e:
    print("\n❌ ¡Nada encontrado! (404 o vacío)")
    print("   → Asegúrate de que:")
    print("      • seed_tracks sea un string, no lista.")
    print("      • El token tenga scope 'user-top-read'.")
    print("      • Si pones filtros de audio, no los llenes todos de golpe; ve probando uno a uno.")
