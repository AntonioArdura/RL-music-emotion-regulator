# **RL Music Emotion Regulator**

## **Objective**

This project implements a music recommendation agent that uses Reinforcement Learning (PPO) to guide a user from an initial emotional state to a target state. The agent learns to generate ideal musical profiles (composed of valence, energy, tempo, mode, and danceability).

Due to recent Spotify API restrictions for developer-mode applications, this project has been adapted to use a large, local dataset of songs for the recommendation and testing phase, making it fully functional offline.

## **Project Structure**

The project is divided into three main Python scripts:

1. user\_emotional\_simulator.py: Implements the custom Gymnasium environment where the RL agent is trained. It simulates a user's emotional transitions and interactions.  
2. train\_ppo\_emotional\_agent.py: Handles the training of the PPO agent within the simulated environment using Stable Baselines3. It saves the trained model for later use.  
3. interactive\_test.py: The main script for testing. It loads a pre-trained agent and a local song dataset, then runs an interactive session where the agent recommends songs and collects real human feedback.

## **Prerequisites**

* Python 3.8 or higher  
* The project's song dataset (see Setup section)

## **Setup**

### **1\. Clone the repository**

git clone https://github.com/AntonioArdura/RL-music-emotion-regulator.git  
cd RL-music-emotion-regulator

### **2\. Create & activate a virtual environment**

**macOS / Linux**

python3 \-m venv .venv  
source .venv/bin/activate

**Windows (PowerShell)**

python \-m venv .venv  
.venv\\Scripts\\Activate.ps1

### **3\. Install dependencies**

This project uses pandas for data handling. Make sure to install all requirements.

pip install \-r requirements.txt

### **4\. Download the Dataset**

This project relies on a local dataset of song features.

1. Download the **"Spotify \- 1.2M Songs"** dataset from Kaggle:  
   [https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs)  
2. Unzip the file.  
3. Place the tracks.csv file into the root directory of this project.

## **Workflow**

### **Step 1: Training the Agent**

To train a new PPO model from scratch, run:

python train\_ppo\_emotional\_agent.py

This will log training metrics to the ppo\_tensorboard\_logs/ directory and save the best model found during evaluation to ppo\_trained\_models/best\_model/best\_model.zip.

### **Step 2: Testing the Agent Interactively**

Once you have a trained model, you can run an interactive session. This script will load the model and the tracks.csv dataset to provide recommendations.

python interactive\_test.py

The script will guide you through a session, asking for your feedback on the recommended songs and your emotional progress on a 0-10 scale. The results of your session will be saved to real\_interactions\_rating.json.

## **Monitoring & Analysis**

### **TensorBoard**

To visualize the agent's learning curves during or after training, run:

tensorboard \--logdir ppo\_tensorboard\_logs

### **Analyzing Interactions**

To get a summary of a completed interactive session, you can use the analysis script:

python analyze\_interactions.py

This will read the real\_interactions\_rating.json file and print a summary table of the agent's performance.

## **(Optional) Creating a Custom Song Library via Spotify API**

The project initially included scripts to create a custom song library using the Spotify API (create\_song\_library.py). Due to recent API restrictions that block access to key endpoints like /v1/audio-features for new developer apps, these scripts may not work for you.

If you have an older, established Spotify Developer App, you can try them. This requires setting your credentials as environment variables first.
