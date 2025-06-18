# RL-music-emotion-regulator

## Objective

This project implements a music recommendation agent that uses Reinforcement Learning (PPO)  
to guide users from a current emotional state to a target state by selecting song features  
(valence, energy, tempo, mode, danceability) and fetching matching tracks from Spotify.

## Prerequisites

- Python 3.8 or higher  
- A Spotify Developer account with a registered app (you will need your Client ID, Client Secret,  
  and Redirect URI).

## Setup

### Clone the repository 
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo 

### Create & activate a virtual environment

**macOS / Linux**

`bash
python3 -m venv .venv
source .venv/bin/activate´

**Windows (PowerShell)**

`bash
python -m venv .venv
.venv\Scripts\Activate.ps1´

### Install dependencies

`bash
pip install -r requirements.txt´

### Configure Spotify credentials
Set your credentials as environment variables (or adjust the script to read them):

`bash
export SPOTIPY_CLIENT_ID="your_client_id"
export SPOTIPY_CLIENT_SECRET="your_client_secret"
export SPOTIPY_REDIRECT_URI="http://127.0.0.1:8888/callback"´

## Training the Agent

Train a new PPO model with:

`bash
python train_ppo_emotional_agent.py´

By default, the model will log metrics to ppo_tensorboard_logs/
and save the best checkpoint under ppo_trained_models/best_model.zip.

**Optional flags** (if supported):

--timesteps N — total number of training timesteps (e.g. --timesteps 200000).
--env-config path/to/config.json – custom simulator configuration

## Testing the Agent
Once you have a trained model (e.g. ppo_trained_models/best_model.zip), run:

`bash
python test_agent.py´

This script will:

1. Authenticate with Spotify
2. Load your trained PPO model
3. Simulate emotion transitions (e.g. “Sad” → “Happy”)
4. Query Spotify for each step’s recommended track
5. Print the results in the console

## (Optional) Monitoring & Debugging
- **TensorBoard**
`bash
tensorboard --logdir ppo_tensorboard_logs´

- **Rate limits / HTTP errors**
If you hit Spotify rate limits or see 5xx errors, add retry/back-off logic in test_agent.py.
